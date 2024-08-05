import torch
import einops
import numpy as np
import os
# The attacker is parameterized by a low-rank MLP (not used by default)
class LowRankAdversary(torch.nn.Module):
    
    def __init__(self, dim, rank, device, bias=False, zero_init=True):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.device = device
        self.lora_A = torch.nn.Linear(dim, rank, bias=False).to(device)
        self.lora_B = torch.nn.Linear(rank, dim, bias=bias).to(device)
        if zero_init:
            self.lora_B.weight.data.zero_()
    
    def forward(self, x):
        return self.lora_B(self.lora_A(x)) + x


# The attacker is parameterized by a full-rank MLP (not used by default)
class FullRankAdversary(torch.nn.Module):
    
    def __init__(self, dim, device, bias=False):
        super().__init__()
        self.dim = dim
        self.device = device
        self.m = torch.nn.Linear(dim, dim, bias=bias).to(device)

        self.m.weight.data.zero_()
    
    def forward(self, x):
        return self.m(x) + x
def get_next_available_path(base_path, base_name, extension=""):
    """
    获取下一个可用的文件或目录路径。
    
    :param base_path: 文件或目录的基础路径
    :param base_name: 文件或目录的基础名称
    :param extension: 文件的扩展名（如果是文件）
    :return: 下一个可用的文件或目录路径
    """
    index = 1
    while True:
        path = os.path.join(base_path, f"{base_name}_{index}{extension}.npy")
        # print(path)
        if not os.path.exists(path):
            return path
        index += 1

def get_new_path():
    base_path = "/root/autodl-tmp/at/uap"
    base_name = "temp_attack"
    new_path = get_next_available_path(base_path, base_name)
    return new_path


# Standard projected gradient attack (used by default)
class GDAdversary(torch.nn.Module):
    
    def __init__(self, dim, epsilon, attack_mask, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.epsilon = epsilon

        
        self.attack_mask = attack_mask

        if dtype:
            self.attack = torch.nn.Parameter(torch.zeros(
                attack_mask.shape[0], attack_mask.shape[1], dim, device=self.device, dtype=dtype))
        else:
            self.attack = torch.nn.Parameter(torch.zeros(
                attack_mask.shape[0], attack_mask.shape[1], dim, device=self.device))
        # print("This is the shape for GDadversary", self.attack.shape)
        uap_path = "/root/autodl-tmp/at/uap/temp_attack.npy"
        if os.path.exists(uap_path):
            #print('load in uap')
            #print(self.attack_mask)
            uap_tensor = torch.tensor(np.load(uap_path), device=self.device, dtype=self.attack.dtype)
            # Create a new tensor with the required values
            new_attack = self.attack.clone()
            new_attack[self.attack_mask] = uap_tensor
            
            # Assign the new tensor to self.attack to avoid in-place operation
            self.attack = torch.nn.Parameter(new_attack)    
        self.clip_attack()
        
    def forward(self, x):
        #print(x.shape)
        if x.shape[1] == 1 and self.attack.shape[1] != 1:  # generation mode (perturbation already applied)
            #print("pass")
            return x
        else:
            #np.save('x2', x.to(torch.float32).detach().cpu().numpy())
            if self.device is None or self.device != x.device:
                with torch.no_grad():
                    self.device = x.device
                    self.attack.data = self.attack.data.to(self.device)
                    self.attack_mask = self.attack_mask.to(self.device)
            #print("This is x shape", x.shape)
            #print(x.shape[1])
            #print("This is attack mask shape",self.attack_mask.shape)
            if x.shape[1] == self.attack_mask.shape[1]: # training model
                perturbed_acts =  self.attack[self.attack_mask[:, :x.shape[1]]].to(x.dtype)
                x[self.attack_mask[:, :x.shape[1]]] = perturbed_acts
            elif x.shape[1] != self.attack_mask.shape[1] and x.shape[1] != 1: # evaluation model
                print("run only on suffix")
                temp_mask = self.attack_mask[:, :x.shape[1]]
                temp_attack = self.attack[:, :x.shape[1], :]
                np.save('/root/autodl-tmp/at/uap/temp_attack', temp_attack[temp_mask[:, :x.shape[1]]].to(torch.float32).detach().cpu().numpy())
                temp_path = get_new_path()
                np.save(temp_path, temp_attack[temp_mask[:, :x.shape[1]]].to(torch.float32).detach().cpu().numpy())
                perturbed_acts =  temp_attack[temp_mask[:, :x.shape[1]]].to(x.dtype)
                #print(temp_mask[:, :x.shape[1]])
                x[temp_mask[:, :x.shape[1]]] = perturbed_acts
                #np.save('perturb_x2', x.to(torch.float32).detach().cpu().numpy())
            #print("This is the part that we changed")
            #print(self.attack_mask[:, :x.shape[1]])
            return x
    
    def clip_attack(self):
        with torch.no_grad():
            # clip attack norm to eps
            norms = torch.norm(self.attack, dim=-1, keepdim=True)
            scale = torch.clamp(norms / self.epsilon, min=1)
            self.attack.div_(scale)

            norms = torch.norm(self.attack, dim=-1)


# Whitened adversaries train perturbations in a whitened space (not used by default)
class WhitenedGDAdversary(torch.nn.Module):

    def __init__(self, dim, device, epsilon, attack_mask, proj=None, inv_proj=None):
        super().__init__()
        self.attack = None
        self.device = device
        self.epsilon = epsilon
        self.proj = proj  # proj is a projection matrix (e.g. one obtained using PCA) to whiten the attack

        if inv_proj is None:
            self.inv_proj = torch.inverse(proj)
        else:
            self.inv_proj = inv_proj

        self.attack = torch.nn.Parameter(torch.randn(
            attack_mask.shape[0], attack_mask.shape[1], dim, device=self.device))
        self.clip_attack()
        self.attack_mask = attack_mask
    
    def forward(self, x):
        unprojected_attack = torch.einsum("n d, batch seq n-> batch seq d", self.inv_proj, self.attack) # n is whitened dimension, d is original hidden size (technically same here)
        x[self.attack_mask[:, :x.shape[1]]] = (x + unprojected_attack.to(x.dtype))[self.attack_mask[:, :x.shape[1]]]
        return x

    
    def clip_attack(self):
        with torch.no_grad():
            # clip attack norm to eps
            norms = torch.norm(self.attack, dim=-1, keepdim=True)
            scale = torch.clamp(norms / self.epsilon, min=1)
            self.attack.div_(scale)

            norms = torch.norm(self.attack, dim=-1)