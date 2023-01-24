import torch

def calculateGrad_vr(reg_param, fn_list, fn_avg, mu_grad_list, sigma_grad_list, K):
    mus = reg_param['mu']
    sigmas = reg_param['sigma']
    for i in range(K):
        for j in range(len(mus)):
            mus[j].grad.data += 1 / (K-1) * (fn_list[i]-fn_avg) * mu_grad_list[i][j]
            sigmas[j].grad.data += 1 / (K-1) * (fn_list[i]-fn_avg) * sigma_grad_list[i][j]

def calculateGrad(reg_params, fn_list, mu_grad_list, sigma_grad_list, K):
    mus = reg_params['mu']
    sigmas = reg_params['sigma']
    for i in range(K):
        for j in range(len(mus)):
            mus[j].grad.data += 1/K*fn_list[i]*mu_grad_list[i][j]
            sigmas[j].grad.data += 1/K*fn_list[i]*sigma_grad_list[i][j]