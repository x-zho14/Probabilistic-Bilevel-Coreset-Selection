import sys
import os
import argparse
import time

import torch
import numpy as np
# from torchvision.datasets import MNIST
from datasets.mnist import MNIST
from torchvision.transforms import transforms
import loss_utils
import random
import json
import models
import torch.nn.functional as F
from logging_utils.dir_manage import get_directories
from torch.utils.tensorboard import SummaryWriter
from logging_utils.tbtools import AverageMeter, ProgressMeter
import copy
from hypergrad.hypergradients_cifar_iter import update_tensor_grads
from varname import nameof
import wandb
from hypergrad.meta import MetaSGD
import math
import torch
import gc
import logging
# the proxy will always want more data, so probabily categorical sampling is a more natural choice, in that case, RL can be used to give the feedback
from noisy_label import noisify
torch.set_printoptions(precision=6,sci_mode=False)

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def debug_memory():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass

def name_print(s):
    print (nameof(s), ":", s)

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    
def get_mnist_full_train_loader(lim=1000, batch_size=1000):
    train_dataset = MNIST(root='dataset/mnist', train=True, transform=mnist_transform, download=True)
    train_dataset.data, train_dataset.targets = train_dataset.data[:lim], train_dataset.targets[:lim]
    if args.noisify:
        train_dataset.targets, actual_noise_rate = noisify(dataset='mnist', nb_classes=10, train_labels=train_dataset.targets.view(-1, 1).numpy(), noise_type=args.noise_type, noise_rate=args.noise_rate)
        train_dataset.targets = torch.tensor(train_dataset.targets).flatten()
        print(f"actual noise rate {actual_noise_rate}")
    full_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=args.num_worker)
    full_train_loader_stochastic = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=args.num_worker)
    return full_train_loader, full_train_loader_stochastic


def get_mnist_coreset_train_loader(indices, batch_size=1000):
    train_dataset = MNIST(root='dataset/mnist', train=True, transform=mnist_transform, download=True)
    train_dataset.data, train_dataset.targets = train_dataset.data[indices], train_dataset.targets[indices]
    l = {}
    for i in range(len(indices)):
        if train_dataset.targets[i].item() in l:
            l[train_dataset.targets[i].item()] += 1
        else:
            l[train_dataset.targets[i].item()] = 1
    print("label", l)
    print("sampled batch size", batch_size)
    coreset_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=args.num_worker)
    return coreset_train_loader

def get_mnist_test_loader(batch_size=10000):
    test_loader = torch.utils.data.DataLoader(MNIST('dataset/mnist', train=False, transform=mnist_transform), batch_size=10000, pin_memory=True, num_workers=4)
    return test_loader

def train_to_converge(model, x_coreset, y_coreset, outer_iter, epoch_converge=100):
    model_copy = copy.deepcopy(model)
    if args.inner_optim=="sgd":
        optimizer = torch.optim.SGD(model_copy.parameters(), lr=args.inner_lr, momentum=0.9, weight_decay=args.inner_wd)
    else:
        optimizer = torch.optim.Adam(model_copy.parameters(), lr=args.inner_lr, weight_decay=args.inner_wd)
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    data, target = x_coreset.cuda(), y_coreset.cuda()
    diverged = False
    for i in range(epoch_converge):
        assign_learning_rate(optimizer, 0.5 * (1 + np.cos(np.pi * i / epoch_converge)) * args.inner_lr)
        optimizer.zero_grad()
        output = model_copy(data)
        acc1, acc5 = loss_utils.accuracy(output, target, topk=(1, 5))
        loss = F.cross_entropy(output, target)
        losses.update(loss.item(), target.size(0))
        top1.update(acc1, target.size(0))
        top5.update(acc5, target.size(0))
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print(f"{i}th iter, inner loss {loss.item()}")
    if math.isnan(loss.item()) or loss > args.div_tol:
        diverged = True
    return model_copy, loss.item(), acc1, acc5, diverged

def get_loss_on_full_train(model, X,Y):
    loss_avg = 0
    data, target = X.cuda(), Y.cuda()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss_avg += loss.item()
    print(loss_avg)
    return loss_avg


def repass_backward(model, subnet, model_checkpoints, opt_checkpoints, outer_grads_w, loader):
    # accumulate gradients backwards to leverage hessian-vector product
    score_grads = [torch.zeros_like(subnet)]
    old_params = model_checkpoints[0]
    old_opt = opt_checkpoints[0]
    model_copy = copy.deepcopy(model)
    for batch_idx, (data, target, idx) in enumerate(loader):
        data, target = data.cuda(), target.cuda()
        old_params_, w_mapped = pseudo_updated_params(model_copy, old_params, old_opt, data, target, subnet[idx])
        grad_batch = torch.autograd.grad(w_mapped, subnet, grad_outputs=outer_grads_w, retain_graph=True)
        score_grads = [g + b for g, b in zip(score_grads, grad_batch)]
    return score_grads[0]

def pseudo_updated_params(pseudo_net, model_checkpoint, opt_checkpoint, data, target, coreset_weights):
    # perform one pseudo update to obtain the mapped weights
    for p, p_old in zip(pseudo_net.parameters(), model_checkpoint):
        p.data.copy_(p_old.cuda())
    pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=0.1)
    if args.inner_optim == "sgd":
        pseudo_optimizer.load_state_dict(opt_checkpoint)
    w_old = [p for p in pseudo_net.parameters()]
    pseudo_outputs = pseudo_net(data)
    pseudo_loss_vector = F.cross_entropy(pseudo_outputs, target.long(), reduction='none').flatten()
    pseudo_loss_vector *= coreset_weights
    pseudo_loss = torch.mean(pseudo_loss_vector)
    pseudo_grads = torch.autograd.grad(pseudo_loss, pseudo_net.parameters(), create_graph=True)
    if args.inner_optim == "adam":
        w_mapped = pseudo_optimizer.meta_step_adam(pseudo_grads, lr=opt_checkpoint['param_groups'][0]['lr'])
    else:
        w_mapped = pseudo_optimizer.meta_step(pseudo_grads, if_update=False)
    return w_old, w_mapped


def get_all_data(full_train_loader):
    datas, targets = [], []
    for data, target, _ in full_train_loader:
        datas.append(data)
        targets.append(target)
    return torch.cat(datas), torch.cat(targets)

def calculateGrad_vr(scores, fn_list, grad_list, fn_avg):
    for i in range(args.K):
        scores.grad.data += 1 / (args.K-1) * (fn_list[i]-fn_avg) * grad_list[i]

def calculateGrad(scores, fn_list, grad_list):
    for i in range(args.K):
        scores.grad.data += 1/args.K*fn_list[i]*grad_list[i]

def solve(model, full_train_loader, full_train_loader_stochastic, test_loader, writer, epoch_converge=100):
    X, Y = get_all_data(full_train_loader)
    print(X.size())
    print(args.coreset_size)
    prune_rate = args.coreset_size / args.limit

    ts = int(args.ts * args.max_outer_iter)
    te = int(args.te * args.max_outer_iter)
    pr_target = prune_rate
    pr_start = args.start_coreset_size / args.limit if args.iterative else prune_rate
    scores = torch.full([len(full_train_loader.dataset)], pr_start, dtype=torch.float, requires_grad=True, device="cuda")
    scores_opt = torch.optim.Adam([scores], lr=args.outer_lr)
    scores.grad = torch.zeros_like(scores)

    for outer_iter in range(args.max_outer_iter):
        if args.iterative:
            if outer_iter < ts:
                prune_rate = pr_start
            elif outer_iter < te:
                prune_rate = pr_target + (pr_start - pr_target) * (1 - (outer_iter - ts) / (te - ts)) ** 3
            else:
                prune_rate = pr_target
        print(args.coreset_size)
        args.coreset_size = prune_rate * args.limit
        print(args.coreset_size, prune_rate)

        print("now coreset_size", args.coreset_size)
        print(f"outer_iter {outer_iter}")
        writer.add_histogram("Scores Distribution", scores, outer_iter)
        temp = 1 / ((1 - 0.03) * (1 - outer_iter / args.max_outer_iter) + 0.03)
        assign_learning_rate(scores_opt, 0.5 * (1 + np.cos(np.pi * outer_iter / args.max_outer_iter)) * args.outer_lr)
        if args.print_score:
            print(f"scores {scores[:1000].data}")
        fn_list = []
        grad_list = []
        fn_avg = 0
        all_models = []
        for i in range(args.K):
            diverged = True
            while diverged:
                subnet, grad = obtain_mask(scores)
                grad_list.append(grad)
                subnet = subnet.detach()
                indices = torch.nonzero(subnet.squeeze())
                indices = indices.reshape(len(indices)).cpu().numpy()
                x_coreset, y_coreset = X[indices], Y[indices]
                model_copy_converged, loss, top1, top5, diverged = train_to_converge(model, x_coreset, y_coreset, outer_iter, epoch_converge=epoch_converge)
                print(f"diverged {diverged} loss {loss}")
            scores_opt.zero_grad()
            all_models.append(model_copy_converged)
            with torch.no_grad():
                # loss_on_full_train = get_loss_on_full_train(model_copy_converged, full_train_loader)
                loss_on_full_train = get_loss_on_full_train(model_copy_converged, X, Y)
            del model_copy_converged
            fn_list.append(loss_on_full_train)
            fn_avg += loss_on_full_train/args.K
            torch.cuda.empty_cache()
        with torch.no_grad():
            if args.vr:
                calculateGrad_vr(scores, fn_list, grad_list, fn_avg)
            else:
                calculateGrad(scores, fn_list, grad_list)
        torch.nn.utils.clip_grad_norm_(scores, args.clip_constant)
        scores_opt.step()
        constrainScoreByWhole(scores)

        index_min = np.argmin(fn_list)
        print(fn_list, index_min, sum(fn_list)/args.K)
        model_copy_converged = all_models[index_min]
        wandb.log({"train/avg_fn": sum(fn_list)/args.K})
        if (outer_iter+1) % args.test_freq == 0:
            acc1, acc5, loss = test(model_copy_converged, data_t, target_t)
            print(f"{outer_iter}th iteration, test acc1 {acc1}, acc5 {acc5}, loss {loss}")
    print("++++++++++++++++finished solving++++++++++++++++++++")
    sample_times = args.sample_times
    best_loss = 100
    best_indices = None
    while sample_times > 0:
        print(sample_times)
        subnet = (torch.rand_like(scores) < scores).float()
        indices = torch.nonzero(subnet.squeeze())
        indices = indices.reshape(len(indices)).cpu().numpy()
        print(f"sampled length {len(indices)}")
        x_coreset, y_coreset = X[indices], Y[indices]
        model_copy_converged, loss, top1, top5, _ = train_to_converge(model, x_coreset, y_coreset, outer_iter, epoch_converge=epoch_converge)
        acc1, acc5, loss = test2(model_copy_converged, full_train_loader)
        if loss < best_loss:
            best_indices = indices
            best_loss = loss
        sample_times -= 1
        print(f"sample {args.sample_times - sample_times + 1}th time, loss {loss}, best loss {best_loss}")
    indices = best_indices.reshape(len(best_indices))
    return indices

def solve_v_total(weight, subset):
    k = subset
    a, b = 0, 0
    b = max(b, weight.max())
    def f(v):
        s = (weight - v).clamp(0, 1).sum()
        return s - k
    if f(0) < 0:
        return 0
    itr = 0
    while (1):
        itr += 1
        v = (a + b) / 2
        obj = f(v)
        if abs(obj) < 1e-3 or itr > 20:
            break
        if obj < 0:
            b = v
        else:
            a = v
    v = max(0, v)
    return v

def constrainScoreByWhole(scores):
    with torch.no_grad():
        v = solve_v_total(scores, args.coreset_size)
        scores.sub_(v).clamp_(0, 1)

class GetMaskDiscrete(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m_cont):
        m_dis = (m_cont >= 0.5).float()
        return m_dis
    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs

def obtain_mask(scores):
    subnet = (torch.rand_like(scores) < scores).float()
    return subnet, (subnet - scores) / ((scores + 1e-20) * (1 - scores + 1e-20))

def train(model, data, target, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    acc1, acc5 = loss_utils.accuracy(output, target, topk=(1, 5))
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()
    return acc1, acc5, loss

def train_stochastic(model, loader, optimizer, nr_epochs=6, scheduler=None, return_loss = False):
    loss_fn = torch.nn.CrossEntropyLoss()
    for ep in range(nr_epochs):
        model.train()
        for batch_idx, (inputs, targets, _) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            acc1, acc5 = loss_utils.accuracy(output, targets, topk=(1, 5))
            if batch_idx % 20 == 0 or batch_idx == len(loader)-1:
                print(f"epoch {ep} full train {batch_idx} loss {loss.item()}, acc1 {acc1}, acc5 {acc5}")
        if scheduler is not None:
            assign_learning_rate(optimizer, 0.5 * (1 + np.cos(np.pi * ep / nr_epochs)) * args.inner_lr)
    if return_loss:
        return loss, acc1, acc5

def test(model, data_t, target_t):
    with torch.no_grad():
        model.eval()
        output = model(data_t)
        acc1, acc5 = loss_utils.accuracy(output, target_t, topk=(1, 5))
        loss = F.cross_entropy(output, target_t)
    return acc1, acc5, loss

def test2(model, test_loader):
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target, _) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            acc1, acc5 = loss_utils.accuracy(output, target, topk=(1, 5))
            loss = F.cross_entropy(output, target)
            losses.update(loss.item(), target.size(0))
            top1.update(acc1, target.size(0))
            top5.update(acc5, target.size(0))
    return top1.avg, top5.avg, loss.item()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MNIST summary generator')
    parser.add_argument('--seed', type=int, default=0, metavar='seed', help='random seed (default: 0)')
    parser.add_argument('--method', type=str, default="probability_1step")
    parser.add_argument('--coreset_size', default=100, type=int)
    parser.add_argument('--start_coreset_size', default=100, type=int)
    parser.add_argument('--max_prob_update', default=2000, type=int)
    parser.add_argument('--max_weight_update', default=100, type=int)
    parser.add_argument('--limit', default=1000, type=int)
    parser.add_argument('--K', default=1, type=int)
    parser.add_argument('--train_epoch', default=4000, type=int)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--outer_lr', default=5e-2, type=float)
    parser.add_argument('--inner_lr', default=0.1, type=float)
    parser.add_argument('--full_lr', default=2e-3, type=float)
    parser.add_argument('--full_wd', default=1e-3, type=float)
    parser.add_argument('--inner_wd', default=0, type=float)
    parser.add_argument('--div_tol', default=9, type=float)
    parser.add_argument('--outer_ratio', default=0.1, type=float)
    parser.add_argument('--max_outer_iter', default=2000, type=int)
    parser.add_argument('--num_worker', default=4, type=int)
    parser.add_argument('--sample_times', default=1, type=int)
    parser.add_argument('--inter_dim', default=32, type=int)
    parser.add_argument('--runs_name', default="ours", type=str)
    parser.add_argument('--inner_optim', default="sgd", type=str)
    parser.add_argument('--model', default="convnet", type=str)
    parser.add_argument('--project', default="pixel_selection", type=str)
    parser.add_argument('--scheduler', default="cosine", type=str)
    parser.add_argument('--noise_type', default="pairflip", type=str)
    parser.add_argument('--noise_rate', default=0.2, type=float)
    parser.add_argument('--epoch_converge', default=100, type=int)
    parser.add_argument("--iterative", default=False, action="store_true")
    parser.add_argument("--mean_grad", default=False, action="store_true")
    parser.add_argument("--clip_grad", default=False, action="store_true")
    parser.add_argument("--print_score", default=False, action="store_true")
    parser.add_argument("--stochastic", default=False, action="store_true")
    parser.add_argument("--noisify", default=False, action="store_true")
    parser.add_argument("--stochastic_outer", default=False, action="store_true")
    parser.add_argument('--ts', default=0.16, type=float)
    parser.add_argument('--te', default=0.6, type=float)
    parser.add_argument('--clip_constant', default=3, type=float)
    parser.add_argument('--wandb', action="store_true")
    parser.add_argument("--vr", default=False, action="store_true")
    parser.add_argument('--test_freq', default=3, type=float)



    args = parser.parse_args()
    if args.wandb:
        wandb.init(project=args.project, name=args.runs_name, config=args)
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    print(run_base_dir, ckpt_base_dir, log_base_dir)
    writer = SummaryWriter(log_dir=log_base_dir)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    full_train_loader, full_train_loader_stochastic  = get_mnist_full_train_loader(args.limit, args.batch_size)
    test_loader = get_mnist_test_loader()
    data_t, target_t, _ = next(iter(test_loader))
    data_t, target_t = data_t.cuda(), target_t.cuda()

    if args.model=="convnet":
        model_select = models.ConvNetNoDropout(10, base_hid=args.inter_dim).cuda()
    else:
        model_select = models.FNNet(28 * 28, args.inter_dim, 10).cuda()
        model_select.eval()
    print(model_select)
    start = time.time()
    indices = solve(model_select, full_train_loader, full_train_loader_stochastic, test_loader, writer, epoch_converge=args.epoch_converge)
    end = time.time()
    print(f"sampled indices {indices}")
    print(f"time spent {end - start}")
    X, Y = get_all_data(full_train_loader)
    data, target = X[indices].cuda(), Y[indices].cuda()
    # coreset_train_loader = get_mnist_coreset_train_loader(indices, args.batch_size)

    test_loss_final = AverageMeter("TestLossFinal", ":.3f", write_avg=False)
    test_top1_final = AverageMeter("TestAcc@1Final", ":6.2f", write_avg=False)
    test_top5_final = AverageMeter("TestAcc@5Final", ":6.2f", write_avg=False)
    l = [test_loss_final, test_top1_final, test_top5_final]
    progress = ProgressMeter(args.max_outer_iter, l, prefix="final test")
    acc_mean = []
    finals = []
    for i in range(3):
        if args.model=="convnet":
            model_train = models.ConvNet(10).cuda()
        else:
            model_train = models.FNNet(28 * 28, 100, 10).cuda()
        optimizer = torch.optim.Adam(model_train.parameters(), lr = args.full_lr, weight_decay= args.full_wd)
        best_acc1, best_acc5, best_train_acc1, best_train_acc5 = 0, 0, 0, 0
        # data, target, _ = next(iter(coreset_train_loader))
        # data, target = data.cuda(), target.cuda()
        data_t, target_t, _ = next(iter(test_loader))
        data_t, target_t = data_t.cuda(), target_t.cuda()
        for epoch in range(0, args.train_epoch):
            train_acc1, train_acc5, train_loss = train(model_train, data, target, optimizer)
            test_acc1, test_acc5, test_loss = test(model_train, data_t, target_t)
            is_best = test_acc1 > best_acc1
            best_acc1 = max(test_acc1, best_acc1)
            best_acc5 = max(test_acc5, best_acc5)
            best_train_acc1 = max(train_acc1, best_train_acc1)
            best_train_acc5 = max(train_acc5, best_train_acc5)
            if epoch % 500 == 0 or epoch == args.train_epoch - 1:
                print(f"epoch {epoch}, test acc1 {test_acc1}, test acc5 {test_acc5}, test loss {test_loss}")
                print(f"epoch {epoch}, train acc1 {train_acc1}, train acc5 {train_acc5}, train loss {train_loss}")
                print(f"best acc1: {best_acc1}, best acc5: {best_acc5}, best train acc1: {best_train_acc1}, best test acc5: {best_train_acc5}, ckpt_base_dir: {ckpt_base_dir}, log_base_dir: {log_base_dir}")
        acc_mean.append(best_acc1)
        finals.append(test_acc1)
        print(acc_mean)
        print(finals)
    print(f"sampled indices {indices}")
    print(f"time spent {end - start}")
    print(run_base_dir, ckpt_base_dir, log_base_dir)

