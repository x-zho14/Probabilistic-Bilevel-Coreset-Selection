import sys
import os
# os.environ['CUDA_VISIBLE_DEVICES']='3'
import argparse
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
import numpy as np
# the proxy will always want more data, so probabily categorical sampling is a more natural choice, in that case, RL can be used to give the feedback
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
    # if args.model != "convnet":
        # train_dataset.data = train_dataset.data.reshape(-1, 28*28)
    full_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=args.num_worker)
    return full_train_loader

def get_mnist_full_train_loader2(lim=1000, batch_size=60000):
    train_dataset = MNIST(root='dataset/mnist', train=True, transform=mnist_transform, download=True)
    print(len(train_dataset))
    train_dataset.data, train_dataset.targets = train_dataset.data[:lim], train_dataset.targets[:lim]
    print(train_dataset.data.size())
    # if args.model != "convnet":
        # train_dataset.data = train_dataset.data.reshape(-1, 28*28)
    full_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=args.num_worker)
    print(len(full_train_loader))
    return full_train_loader

def get_mnist_test_loader(batch_size=10000):
    test_loader = torch.utils.data.DataLoader(MNIST('dataset/mnist', train=False, transform=mnist_transform), batch_size=batch_size, pin_memory=True, num_workers=4)
    # if args.model != "convnet":
    #     test_loader.dataset.data = test_loader.dataset.data.view(-1, 28*28)
    return test_loader

def train_to_converge(model, x_coreset, y_coreset, outer_iter, epoch_converge=100):
    model_copy = copy.deepcopy(model)
    if args.inner_optim=="sgd":
        print("here")
        optimizer = torch.optim.SGD(model_copy.parameters(), lr=args.inner_lr, momentum=0.9, weight_decay=args.inner_wd)
    else:
        optimizer = torch.optim.Adam(model_copy.parameters(), lr=args.inner_lr, weight_decay=args.inner_wd)
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    # data, target, _ = next(iter(coreset_loader))
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

def get_grad_weights_on_full_train(model, full_train_loader):
    grad_weights_on_full_train = []
    for batch_idx, (data, target, _) in enumerate(full_train_loader):
        data, target = data.cuda(), target.cuda()
        output = model(data)
        if args.mean_grad:
            loss = F.cross_entropy(output, target)/len(full_train_loader)
        else:
            loss = F.cross_entropy(output, target)
        grad_weights_on_full_train_batch = torch.autograd.grad(loss, model.parameters())
        if batch_idx > 0:
            grad_weights_on_full_train = [wb+w for wb, w in zip(grad_weights_on_full_train_batch, grad_weights_on_full_train)]
        else:
            grad_weights_on_full_train = grad_weights_on_full_train_batch
    return grad_weights_on_full_train


def repass_backward(model, subnet, scores, model_checkpoints, opt_checkpoints, outer_grads_w, loader):
    # accumulate gradients backwards to leverage hessian-vector product
    score_grads = [torch.zeros_like(subnet)]
    old_params = model_checkpoints[0]
    old_opt = opt_checkpoints[0]
    for batch_idx, (data, target, idx) in enumerate(loader):
        data, target = data.cuda(), target.cuda()
        if args.model != "convnet":
            data = data.squeeze()
        old_params_, w_mapped = pseudo_updated_params(model, old_params, old_opt, data, target, subnet)
        grad_batch = torch.autograd.grad(w_mapped, scores, grad_outputs=outer_grads_w, retain_graph=True)
        score_grads = [g + b for g, b in zip(score_grads, grad_batch)]
    return score_grads[0]

def pseudo_updated_params(model, model_checkpoint, opt_checkpoint, data, target, coreset_weights):
    # perform one pseudo update to obtain the mapped weights
    pseudo_net =  copy.deepcopy(model)
    for p, p_old in zip(pseudo_net.parameters(), model_checkpoint):
        p.data.copy_(p_old.cuda())
    pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=0.1)
    if args.inner_optim == "sgd":
        pseudo_optimizer.load_state_dict(opt_checkpoint)
    w_old = [p for p in pseudo_net.parameters()]
    # data_core = data*coreset_weights
    if args.model == "convnet":
        pseudo_outputs = pseudo_net(data*(coreset_weights.expand(data.size(0),-1, -1,-1)))
    else:
        coreset_weights = coreset_weights.expand(data.size(0),-1)
        pseudo_outputs = pseudo_net(data*coreset_weights)
    pseudo_loss = F.cross_entropy(pseudo_outputs, target.long())
    pseudo_grads = torch.autograd.grad(pseudo_loss, pseudo_net.parameters(), create_graph=True)
    if args.inner_optim == "adam":
        w_mapped = pseudo_optimizer.meta_step_adam(pseudo_grads, lr=opt_checkpoint['param_groups'][0]['lr'])
    else:
        w_mapped = pseudo_optimizer.meta_step(pseudo_grads, if_update=False)
    return w_old, w_mapped


def get_loss_on_full_train(model, X,Y):
    loss_avg = 0
    data, target = X.cuda(), Y.cuda()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss_avg += loss.item()
    return loss_avg


def calculateGrad_vr(scores, fn_list, grad_list, fn_avg):
    for i in range(args.K):
        scores.grad.data += 1 / (args.K-1) * (fn_list[i]-fn_avg) * grad_list[i]

def calculateGrad(scores, fn_list, grad_list):
    for i in range(args.K):
        scores.grad.data += 1/args.K*fn_list[i]*grad_list[i]

def solve(model, full_train_loader, full_train_loader2, test_loader, writer, epoch_converge=100):
    X, Y, _ = next(iter(full_train_loader2))
    X, Y = X.cuda(), Y.cuda()
    X_test, Y_test, _ =  next(iter(test_loader))
    print(X_test.size())
    if args.model=="convnet":
        num_elements = (X.size(1)*X.size(2)*X.size(3))
    else:
        X= X.squeeze()
        X_test =  X_test.squeeze()
        num_elements = X.size(1)
    pr_target = args.coreset_size / num_elements
    prune_rate = pr_target
    ts = int(args.ts * args.max_outer_iter)
    te = int(args.te * args.max_outer_iter)
    pr_start = prune_rate if not args.iterative else args.start_coreset_size / num_elements
    scores = torch.full_like(X[0], pr_start, dtype=torch.float, requires_grad=True, device="cuda")
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
        args.coreset_size = prune_rate * num_elements
        print("now coreset_size", args.coreset_size)
        print(f"outer_iter {outer_iter}")
        writer.add_histogram("Scores Distribution", scores, outer_iter)
        temp = 1 / ((1 - 0.03) * (1 - outer_iter / args.max_outer_iter) + 0.03)
        assign_learning_rate(scores_opt, 0.5 * (1 + np.cos(np.pi * outer_iter / args.max_outer_iter)) * args.outer_lr)
        fn_list = []
        grad_list = []
        fn_avg = 0
        if args.print_score:
            print(f"scores {scores.data}")
        all_models = []
        for i in range(args.K):
            diverged = True
            while diverged:
                subnet, grad = obtain_mask(scores)
                grad_list.append(grad)
                if args.model == "convnet":
                    subnet_detached = subnet.expand(X.size(0),-1, -1,-1).detach()
                else:
                    subnet_detached = subnet.expand(X.size(0),-1).detach()
                x_coreset, y_coreset = X*subnet_detached, Y
                model_copy_converged, loss, top1, top5, diverged = train_to_converge(model, x_coreset, y_coreset, outer_iter, epoch_converge=epoch_converge)
                print(f"diverged {diverged} loss {loss}")
            scores_opt.zero_grad()
            all_models.append(model_copy_converged)
            with torch.no_grad():
                loss_on_full_train = get_loss_on_full_train(model_copy_converged, X, Y)
            print(loss_on_full_train)
            fn_list.append(loss_on_full_train)
            fn_avg += loss_on_full_train/args.K
            # del model_copy_converged
            torch.cuda.empty_cache()
        with torch.no_grad():
            if args.vr:
                calculateGrad_vr(scores, fn_list, grad_list, fn_avg)
            else:
                calculateGrad(scores, fn_list, grad_list)
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(scores, args.clip_constant)
        scores_opt.step()
        index_min = np.argmin(fn_list)
        print(fn_list, index_min, sum(fn_list)/args.K)
        model_copy_converged = all_models[index_min]
        constrainScoreByWhole(scores)
        wandb.log({"train/avg_fn": sum(fn_list)/args.K})
        if (outer_iter+1) % args.test_freq == 0:
            acc1, acc5, loss = test(model_copy_converged,X_test, Y_test)
            if args.model == "convnet":
                subnet_detached_test = subnet.expand(X_test.size(0),-1, -1,-1).detach()
            else:
                subnet_detached_test = subnet.expand(X_test.size(0),-1).detach()
            acc1_m, acc5_m, loss_m = test(model_copy_converged,X_test.cuda()*subnet_detached_test, Y_test)
            print(f"{outer_iter}th iteration, test acc1 {acc1}, acc5 {acc5}, loss {loss}")
            print(f"{outer_iter}th iteration, test masked acc1 {acc1_m}, acc5 {acc5_m}, loss {loss_m}")
    print("++++++++++++++++finished solving++++++++++++++++++++")
    # sample_times = args.sample_times
    # best_loss = 100
    # best_indices = None
    # while sample_times > 0:
    #     subnet = (torch.rand_like(scores) < scores).float()
    #     indices = torch.nonzero(subnet.squeeze())
    #     if len(indices) == args.coreset_size:
    #         indices = indices.reshape(len(indices))
    #         x_coreset, y_coreset = X[indices], Y[indices]
    #         model_copy_converged, loss, top1, top5, model_weights_cache, opt_checkpoints_cache, diverged = train_to_converge(model, x_coreset, y_coreset, outer_iter, epoch_converge=epoch_converge)
    #         acc1, acc5, loss = test2(model_copy_converged, full_train_loader)
    #         if loss < best_loss:
    #             best_indices = indices
    #             best_loss = loss
    #         sample_times -= 1
    #         print(f"sample {args.sample_times - sample_times + 1}th time, loss {loss}, best loss {best_loss}")
    #     else:
    #         print(f"sampled length {len(indices)}")
    # indices = best_indices.reshape(len(best_indices))
    subnet = (torch.rand_like(scores) < scores).float()
    return subnet

def solve_v_total(weight, subset):
    weight = weight.view(-1)
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
        data_t, target_t = data_t.cuda(), target_t.cuda()
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
    parser.add_argument('--coreset_size', default=30, type=int)
    parser.add_argument('--true_size', default=10, type=int)
    parser.add_argument('--start_coreset_size', default=100, type=int)
    parser.add_argument('--max_prob_update', default=2000, type=int)
    parser.add_argument('--max_weight_update', default=100, type=int)
    parser.add_argument('--limit', default=1000, type=int)
    parser.add_argument('--K', default=1, type=int)
    parser.add_argument('--train_epoch', default=4000, type=int)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--outer_lr', default=5e-1, type=float)
    parser.add_argument('--inner_lr', default=0.1, type=float)
    parser.add_argument('--inner_wd', default=0, type=float)
    parser.add_argument('--div_tol', default=2, type=float)
    parser.add_argument('--max_outer_iter', default=2000, type=int)
    parser.add_argument('--num_worker', default=4, type=int)
    parser.add_argument('--sample_times', default=1, type=int)
    parser.add_argument('--inter_dim', default=32, type=int)
    parser.add_argument('--runs_name', default="ours", type=str)
    parser.add_argument('--inner_optim', default="sgd", type=str)
    parser.add_argument('--model', default="convnet", choices=["convnet", "fnn"], type=str)
    parser.add_argument('--project', default="pixel_selection", type=str)
    parser.add_argument('--scheduler', default="cosine", type=str)
    parser.add_argument('--epoch_converge', default=200, type=int)
    parser.add_argument("--iterative", default=False, action="store_true")
    parser.add_argument("--mean_grad", default=False, action="store_true")
    parser.add_argument("--clip_grad", default=False, action="store_true")
    parser.add_argument("--print_score", default=False, action="store_true")
    parser.add_argument("--stochastic", default=False, action="store_true")
    parser.add_argument('--ts', default=0.16, type=float)
    parser.add_argument('--te', default=0.6, type=float)
    parser.add_argument('--clip_constant', default=3, type=float)
    parser.add_argument('--test_freq', default=1, type=int)
    parser.add_argument('--wandb', action="store_true")
    parser.add_argument('--random', action="store_true")
    parser.add_argument("--vr", default=False, action="store_true")

    args = parser.parse_args()
    if args.wandb:
        wandb.init(project=args.project, name=args.runs_name, config=args)
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    print(run_base_dir, ckpt_base_dir, log_base_dir)
    writer = SummaryWriter(log_dir=log_base_dir)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    full_train_loader = get_mnist_full_train_loader(args.limit, args.batch_size)
    full_train_loader2 = get_mnist_full_train_loader2(args.limit, args.limit)
    test_loader = get_mnist_test_loader()
    if args.model=="convnet":
        model_select = models.ConvNetNoDropout(10, base_hid=args.inter_dim).cuda()
    else:
        model_select = models.FNNet(28 * 28, args.inter_dim, 10).cuda()
        model_select.eval()
    print(model_select)
    a =next(iter(full_train_loader2))
    print(a)
    X, Y, _ = next(iter(full_train_loader2))
    if args.random:
        subnet = torch.zeros_like(X[0])
        subnet_indices = np.random.choice(list(range(28*28)), args.coreset_size, replace=False)
        subnet.flatten()[subnet_indices] = 1
        subnet = subnet.cuda()
    else:
        subnet = solve(model_select, full_train_loader, full_train_loader2, test_loader, writer, epoch_converge=args.epoch_converge)
        print(f"subnet {subnet.sum()}")
        torch.save(subnet, f"{ckpt_base_dir}/subnet.pt")
    # subnet = torch.load("results/probability_1step/10/0.1/16/checkpoints/subnet.pt")
    # coreset_train_loader = get_mnist_coreset_train_loader(indices, args.batch_size)

    test_loss_final = AverageMeter("TestLossFinal", ":.3f", write_avg=False)
    test_top1_final = AverageMeter("TestAcc@1Final", ":6.2f", write_avg=False)
    test_top5_final = AverageMeter("TestAcc@5Final", ":6.2f", write_avg=False)
    l = [test_loss_final, test_top1_final, test_top5_final]
    progress = ProgressMeter(args.max_outer_iter, l, prefix="final test")
    if args.model != "convnet":
        X= X.squeeze()
    acc_mean = []
    for i in range(3):
        if args.model=="convnet":
            model_train = models.ConvNet(10).cuda()
        else:
            model_train = models.FNNet(28 * 28, 100, 10).cuda()
        optimizer = torch.optim.Adam(model_train.parameters(), lr = 5e-3)
        best_acc1, best_acc5, best_acc1_m, best_acc5_m, best_train_acc1, best_train_acc5 = 0, 0, 0, 0, 0 ,0
        data, target = X.cuda(), Y.cuda()
        data_t, target_t, _ = next(iter(test_loader))
        data_t, target_t = data_t.cuda(), target_t.cuda()
        if args.model != "convnet":
            data_t = data_t.squeeze()
        if args.model == "convnet":
            subnet_detached = subnet.expand(data.size(0),-1, -1,-1).detach()
            subnet_detached_test = subnet.expand(data_t.size(0),-1, -1,-1).detach()
        else:
            subnet_detached = subnet.expand(data.size(0),-1).detach()
            subnet_detached_test = subnet.expand(data_t.size(0),-1).detach()

        for epoch in range(0, args.train_epoch):
            train_acc1, train_acc5, train_loss = train(model_train, data*subnet_detached, target, optimizer)
            test_acc1, test_acc5, test_loss = test(model_train, data_t, target_t)
            test_acc1_m, test_acc5_m, test_loss_m = test(model_train, data_t*subnet_detached_test, target_t)
            is_best = test_acc1 > best_acc1
            best_acc1 = max(test_acc1, best_acc1)
            best_acc5 = max(test_acc5, best_acc5)
            best_acc1_m = max(test_acc1_m, best_acc1_m)
            best_acc5_m = max(test_acc5_m, best_acc5_m)
            best_train_acc1 = max(train_acc1, best_train_acc1)
            best_train_acc5 = max(train_acc5, best_train_acc5)
            if epoch % 500 == 0 or epoch == args.train_epoch - 1:
                print(f"epoch {epoch}, test acc1 {test_acc1}, test acc5 {test_acc5}, test loss {test_loss}")
                print(f"epoch {epoch}, test acc1_m {test_acc1_m}, test acc5_m {test_acc5_m}, test loss_m {test_loss_m}")
                print(f"epoch {epoch}, train acc1 {train_acc1}, train acc5 {train_acc5}, train loss {train_loss}")
                print(f"best acc1: {best_acc1}, best acc5: {best_acc5}, best acc1_m: {best_acc1_m}, best acc5_m: {best_acc5_m}, best train acc1: {best_train_acc1}, best test acc5: {best_train_acc5}, ckpt_base_dir: {ckpt_base_dir}, log_base_dir: {log_base_dir}")
        acc_mean.append((best_acc1,best_acc1_m))
        print(acc_mean)
    print(run_base_dir, ckpt_base_dir, log_base_dir)

