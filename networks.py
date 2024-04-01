import torch
from torch import nn
import torch.nn.functional as F
import torch.linalg as LA

class SimpleRegProb(nn.Module):

    def __init__(self, d_in, d_out, l1=0, l2=0):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.proj = nn.Sequential(
            nn.Linear(self.d_in, self.d_out),
            nn.LeakyReLU()
        )
        self.l1 = l1
        self.l2 = l2

    def forward(self, x):
        # x(B, d_in)
        y = self.proj(x)
        return y

    def similarity_log_gaussian_kernel(self, x1, x2):
        return - LA.norm(x1 - x2, dim=-1)

    def cal_sim(self, h):
        # h (N, 3, l)
        s12 = (h[:,0] * h[:,1]).sum(dim=-1) # (N, )
        s13 = (h[:,0] * h[:,2]).sum(dim=-1)
        s23 = (h[:,1] * h[:,2]).sum(dim=-1)

        S = torch.stack((s23, s13, s12), dim=1) # (N, 3)
        return S, S.argmax(dim=-1)==2
    
    def l_CE(self, S):
        # maximize the option 3 (s12)
        target = 2 * torch.ones(S.shape[0], device=S.device, dtype=torch.int64)
        l_ce = F.cross_entropy(S, target)
        return l_ce

    def l2_loss(self):
        l2_loss = self.proj[0].weight.square().mean()
        return self.l2 * l2_loss
    
    def loss_func(self, h, tr, return_acc=True, return_loss=True):
        # tr (B, 3)
        # h (N, d_in)
        x = tr @ h # (B, 3, d_in)
        B, _, _ = x.shape
        y = self(x.view(-1, self.d_in)).view(B, 3, self.d_out) # (B, 3, d_out)
        output = []
        output.append(y)

        l1_loss = self.l1 * y.abs().mean()
        if return_acc:
            S, acc = self.cal_sim(y)
            output.append(acc)
        if return_loss:
            l = self.l_CE(S)
            l += self.l2_loss() + l1_loss
            output.append(l)
        return output