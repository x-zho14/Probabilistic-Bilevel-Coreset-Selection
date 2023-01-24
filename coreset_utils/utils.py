import torch

def obtain_mask(scores, temp, eps=1e-20):
    uniform0 = torch.rand_like(scores)
    uniform1 = torch.rand_like(scores)
    noise = -torch.log(torch.log(uniform0 + eps) / torch.log(uniform1 + eps) + eps)
    return GetMaskDiscrete.apply(torch.sigmoid((torch.log(scores + eps) - torch.log(1.0 - scores + eps) + noise) * temp))

def constrainScoreByWhole(scores, coreset_size):
    with torch.no_grad():
        v = solve_v_total(scores, coreset_size)
        scores.sub_(v).clamp_(0, 1)

class GetMaskDiscrete(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m_cont):
        m_dis = (m_cont >= 0.5).float()
        return m_dis
    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs

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