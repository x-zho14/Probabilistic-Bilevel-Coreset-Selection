import torch
import torch.nn as nn
import math
import numpy as np
from functools import partial



class HypernetProb(nn.Module):
    def __init__(self, weight_decay_type = 'per_param', weight_decay_init = 5e-4, sigma_init = 1e-3, if_tune_reg=1, freeze_bn_reg=False, reg_type='L2', normalize_type='exp', scale=(1, 2e-3), if_normalize=False, bn_only=False, reg_names=None, inverse_sigma=False):
        super(HypernetProb, self).__init__()
        self.weight_decay_type = weight_decay_type
        self.weight_decay_init = weight_decay_init
        self.if_tune_reg = if_tune_reg
        self.reg_type = reg_type
        self.tune_bn = not freeze_bn_reg
        self.requires_grad = False if self.if_tune_reg == 0 else True
        self.normalize_type = normalize_type
        self.if_normalize = if_normalize
        self.bn_only = bn_only
        self.init_normalize_func(scale)
        self.reg_names = reg_names
        self.inverse_sigma = inverse_sigma
        self.sigma_init = sigma_init

    def extract_feat(self, img):
        raise NotImplementedError
    def predict(self, feat):
        raise NotImplementedError
    def forward(self, img, return_feat=False):
        feat = self.extract_feat(img)
        out = self.predict(feat)
        if return_feat:
            return out, feat
        else:
            return out

    def init_wdecay(self, weight_decay_type, weight_decay_init):
        if weight_decay_type == "single":
            requires_grad = self.requires_grad
            weight_decay = torch.tensor([self.normalize_inv(weight_decay_init)], requires_grad=requires_grad, device="cuda")
            self.__setattr__(f"weight_decay", weight_decay)

        elif weight_decay_type == "per_param":
            i=0
            for name, p in self.named_parameters():
                if (not self.tune_bn and 'bn' in name) or (self.bn_only and 'bn' not in name):
                    requires_grad = False
                else:
                    requires_grad = self.requires_grad
                self.__setattr__(f"weight_decay_{i}_mu", torch.full_like(p, self.weight_decay_init, requires_grad=requires_grad, device="cuda"))
                if not self.inverse_sigma:
                    self.__setattr__(f"weight_decay_{i}_sigma", torch.full_like(p, self.sigma_init, requires_grad=requires_grad, device="cuda"))
                else:
                    self.__setattr__(f"weight_decay_{i}_sigma", torch.full_like(p, 1/(self.sigma_init+1e-20), requires_grad=requires_grad, device="cuda"))
                i += 1
        elif weight_decay_type == "classifier":
            i=0
            for name, p in self.named_parameters():
                if 'classifier' not in name:
                    requires_grad = False
                else:
                    requires_grad = self.requires_grad
                self.__setattr__(f"weight_decay_{i}", torch.full_like(p, self.normalize_inv(weight_decay_init), requires_grad=requires_grad, device="cuda"))
                i += 1
        elif weight_decay_type == "per_layer":
            i=0
            for name, p in self.named_parameters():
                if not self.tune_bn and 'bn' in name:
                    requires_grad = False
                else:
                    requires_grad = self.requires_grad
                self.__setattr__(f"weight_decay_{i}_mu", torch.tensor(self.weight_decay_init, requires_grad=requires_grad, device="cuda"))
                self.__setattr__(f"weight_decay_{i}_sigma", torch.tensor(self.sigma_init, requires_grad=requires_grad, device="cuda"))
                i += 1
        elif weight_decay_type == "per_channel":
            i=0
            for name, p in self.named_parameters():
                # if 'fc' in name or (not self.tune_bn and 'bn' in name):
                if not self.tune_bn and 'bn' in name:
                    requires_grad = False
                else:
                    requires_grad = self.requires_grad
                self.__setattr__(f"weight_decay_{i}", torch.full((p.shape[0],1),self.normalize_inv(weight_decay_init) , requires_grad=requires_grad, device="cuda"))
                i += 1

    def L2_loss(self):
        if self.weight_decay_type == "single":
            return self.single_L2_loss()
        elif self.weight_decay_type == "per_param" or self.weight_decay_type == "per_bn" or self.weight_decay_type == "classifier" or self.weight_decay_type == "anyin":
            return self.all_L2_loss()
        elif self.weight_decay_type == "per_layer":
            return self.layer_L2_loss()
        elif self.weight_decay_type == "per_channel":
            return self.channel_L2_loss()
        elif self.weight_decay_type == "per_filter":
            return self.filter_L2_loss()
        elif self.weight_decay_type == "none":
            return self.no_L2_loss()
        else:
            raise NotImplementedError("not implemented")

    def single_L2_loss(self):
        loss = 0
        for name, p in self.named_parameters():
            if not self.tune_bn:
                if 'bais' not in name and 'bn' not in name:
                    loss += torch.sum(self.normalize_func(self.weight_decay) * self.reg_term(p))
                else:
                    loss += torch.sum(self.weight_decay_init * self.reg_term(p))
            else:
                loss += torch.sum(self.normalize_func(self.weight_decay) * self.reg_term(p))
        return loss
        # return loss * (torch.exp(self.weight_decay))

    def sample_weight_decay(self, wd_upper_lim=100, reduce=False):
        sampled_wds = []
        grads_mu = []
        grads_sigma = []
        eps = 1e-20
        for i, p in enumerate(self.parameters()):
            mean = getattr(self, f"weight_decay_{i}_mu")
            sigma = getattr(self, f"weight_decay_{i}_sigma")
            # noise = torch.randn_like(mean)
            with torch.no_grad():
                # sampled_wd = noise*sigma + mean
                if not self.inverse_sigma:
                    sampled_wd = torch.normal(mean, torch.sqrt(sigma))
                    if not reduce:
                        grads_mu.append(-(1/(sigma + eps))*(mean-sampled_wd))
                        grads_sigma.append(-(1/(2*sigma + eps)) + 1/(2*sigma**2 + eps)*(sampled_wd - mean)**2)
                    else:
                        grads_mu.append(-(mean-sampled_wd))
                        grads_sigma.append(0)
                    sampled_wd = torch.clamp(sampled_wd, min=0, max=wd_upper_lim)
                    sampled_wds.append(sampled_wd)
                else:
                    sampled_wd = torch.normal(mean, torch.sqrt(1/sigma))
                    g_mu = -sigma*(mean-sampled_wd)
                    g_sigma = 1/(2*sigma + 1e-20) - 1/2*(sampled_wd - mean)**2
                    grads_mu.append(g_mu/g_mu.norm(2, dim=0).clamp(min=eps))
                    grads_sigma.append(g_sigma/g_sigma.norm(2, dim=0).clamp(min=eps))
                    sampled_wd = torch.clamp(sampled_wd, min=0, max=wd_upper_lim)
                    sampled_wds.append(sampled_wd)
            self.__setattr__(f"sampled_weight_decay_{i}", sampled_wd)

        return sampled_wds, grads_mu, grads_sigma

    def use_mean_weight_decay(self):
        for i, p in enumerate(self.parameters()):
            mean = getattr(self, f"weight_decay_{i}_mu")
            self.__setattr__(f"sampled_weight_decay_{i}", mean)



    def get_sampled_weight_decay(self):
        decays = []
        for i, p in enumerate(self.parameters()):
            decays.append(getattr(self,f"sampled_weight_decay_{i}"))
        return decays

    def all_L2_loss(self):
        loss = 0
        for i, p in enumerate(self.parameters()):
            loss += torch.sum(self.normalize_func(getattr(self, f"sampled_weight_decay_{i}")) * self.reg_term(p))
        return loss

    def layer_L2_loss(self):
        loss = 0
        for i, p in enumerate(self.parameters()):
            loss += torch.sum(self.normalize_func(getattr(self, f"sampled_weight_decay_{i}")) * self.reg_term(p))
        return loss

    def channel_L2_loss(self):
        loss = 0
        for i, p in enumerate(self.parameters()):
            loss += torch.sum(self.normalize_func(getattr(self, f"weight_decay_{i}")) * self.reg_term(p).reshape(p.shape[0], -1))
        return loss

    def filter_L2_loss(self):
        loss = 0
        for i, p in enumerate(self.parameters()):
            loss += torch.sum(self.normalize_func(getattr(self, f"weight_decay_{i}")) * self.reg_term(p))
        return loss

    def no_L2_loss(self):
        return 0

    def reg_term(self, params):
        if self.reg_type == "L2":
            return params**2
        elif self.reg_type == "L1":
            return torch.abs(params)

    def get_reg_params(self):
        if self.weight_decay_type == "single":
            return [self.weight_decay]
        elif self.weight_decay_type == "per_param":
            dict = {'mu': [], 'sigma':[]}
            for i in range(self.num_params):
                dict['mu'].append(getattr(self, f"weight_decay_{i}_mu"))
                dict['sigma'].append(getattr(self, f"weight_decay_{i}_sigma"))
            return dict
        else:
            dict = {'mu': [], 'sigma':[]}
            for i in range(self.num_params):
                dict['mu'].append(getattr(self, f"weight_decay_{i}_mu"))
                dict['sigma'].append(getattr(self, f"weight_decay_{i}_sigma"))
            return dict

    def load_reg_params_prob(self, dict):
        for i in range(self.num_params):
            self.__setattr__(f"weight_decay_{i}_mu", dict['mu'][i])

    def get_numels(self):
        if self.weight_decay_type == "per_layer":
            numels_tune = []
            numels_all = [p.numel() for p in self.parameters()]
            for i in range(self.num_params):
                if getattr(self, f"weight_decay_{i}").requires_grad == True:
                    numels_tune.append(numels_all[i])
        else:
            numels_tune = None

        return numels_tune

    def get_reg_params_all(self):
        if self.weight_decay_type == "single":
            return [self.weight_decay]
        else:
            return [getattr(self, f"weight_decay_{i}") for i in range(self.num_params)]

    def load_reg_params(self, reg_path):
        regs = torch.load(reg_path)['reg']
        for i, p in enumerate(self.parameters()):
            regs[i].requires_grad = False
            self.__setattr__(f"weight_decay_{i}", regs[i])



    def init_normalize_func(self, scale):
        if self.normalize_type == 'exp':
            self.normalize_func = torch.exp
            self.normalize_inv = math.log
        elif self.normalize_type == 'sigmoid':
            self.normalize_func = partial(sigmoid, scale=scale)
            self.normalize_inv = partial(sigmoid_inv, scale=scale)
        elif self.normalize_type == 'relu':
            self.normalize_func = torch.relu
            self.normalize_inv = linear
        elif self.normalize_type == 'linear':
            self.normalize_func = linear
            self.normalize_inv = linear
        elif self.normalize_type == 'tanh':
            self.normalize_func = torch.tanh
            self.normalize_inv = math.atanh
        elif self.normalize_type == 'abs':
            self.normalize_func = torch.abs
            self.normalize_inv = linear
        elif self.normalize_type == 'square':
            self.normalize_func = torch.square
            self.normalize_inv = math.sqrt
        elif self.normalize_type == 'softplus':
            self.normalize_func = softplus
            self.normalize_inv = softplusinv
        elif self.normalize_type == 'hard_sigmoid':
            self.normalize_func = partial(hard_sigmoid, scale=scale)
            self.normalize_inv = partial(hard_sigmoid_inv, scale=scale)

def sigmoid_inv(x, scale=(1, 2e-3)):
    return -math.log((scale[1] / (x)) - 1)/scale[0]
def sigmoid(x, scale=(1, 2e-3)):
    return (scale[1])/(1+torch.exp(-scale[0]*x))
def linear(x):
    return x
def softplus(x):
    return torch.log(torch.ones_like(x)+torch.exp(x))
def softplusinv(x):
    return math.log(math.exp(x)-1.)

def hard_sigmoid(x, scale=(10, 2e-3)):
    thresh1=-scale[0]
    thresh2=scale[0]
    upper=scale[1]
    k = upper/(2*thresh2)
    m1 = (x >= thresh2)
    m2 = ((x > thresh1) & (x < thresh2))
    m3 = (x <= thresh1)
    out1 = upper*m1
    out2 = (k * x + upper/2) * m2
    out3 = 0 * m3
    return out1+out2+out3

def hard_sigmoid_inv(x, scale=(10, 2e-3)):
    thresh1= -scale[0]
    thresh2= scale[0]
    upper=scale[1]
    k = upper/(2*thresh2)
    if x > upper:
        raise ValueError("greater than bound")
    elif x < 0:
        raise ValueError("smaller than bound")
    else:
        return x/k - upper/(2*k)

if __name__ == "__main__":
    print(sigmoid(torch.tensor(sigmoid_inv(1e-4, scale=(1, 1))), scale=(1, 1)).item())
    print(torch.exp(torch.tensor(math.log(1e-4))).item())