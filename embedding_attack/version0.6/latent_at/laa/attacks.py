import torch
import einops
import numpy as np
import os
from torch import nn
import torch.nn.functional as F

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
def compute_toward_away_loss_adversary(
    model,
    towards_tokens,
    towards_labels_mask,
    away_tokens,
    away_labels_mask,
    towards_labels=None,
    away_labels=None,
    coefs={"toward": 1.0, "away": 1.0},
    accelerator=None,
):
    losses = {"total": 0}

    if towards_tokens is not None:
        with torch.no_grad():
            logits = model(input_ids=towards_tokens.cuda()).logits
            final_logits = logits[:, :-1][towards_labels_mask[:, 1:]]
            if towards_labels is None:
                towards_labels = towards_tokens[:, 1:][towards_labels_mask[:, 1:]].cuda()

            toward_loss = F.cross_entropy(final_logits, towards_labels)
            losses["toward"] = toward_loss.item()
            losses["total"] += toward_loss.item()

    if away_tokens is not None:
        with torch.no_grad():
            logits = model(input_ids=away_tokens.cuda()).logits
            final_logits = logits[:, :-1][away_labels_mask[:, 1:]]
            if away_labels is None:
                away_labels = away_tokens[:, 1:][away_labels_mask[:, 1:]].cuda()
            away_loss = F.cross_entropy(F.log_softmax(final_logits, dim=-1), away_labels)

            losses["away"] = away_loss.item()
            losses["total"] += away_loss.item()

    return losses

# Standard projected gradient attack (used by default)
class GDAdversary(torch.nn.Module):
    
    def __init__(self, dim, epsilon, attack_mask, batch,embedding_space=None, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.epsilon = epsilon
        self.batch = batch
        self.attack_mask = attack_mask

        if dtype:
            self.attack = torch.nn.Parameter(torch.zeros(
                attack_mask.shape[0], attack_mask.shape[1], dim, device=self.device, dtype=dtype))
        else:
            self.attack = torch.nn.Parameter(torch.zeros(
                attack_mask.shape[0], attack_mask.shape[1], dim, device=self.device))
        self.embedding_space = torch.tensor(embedding_space, device=self.device, dtype=self.attack.dtype)
        random_indices = torch.randint(0, self.embedding_space.shape[0], (1, attack_mask.shape[1]))
        initial_attack = self.embedding_space[random_indices].to(device)
        self.attack = torch.nn.Parameter(initial_attack) 
        # print("This is the shape for GDadversary", self.attack.shape)
        # uap_path = "/root/autodl-tmp/at/uap/temp_attack.npy"
        # if os.path.exists(uap_path):
        #     #print('load in uap')
        #     #print(self.attack_mask)
        #     uap_tensor = torch.tensor(np.load(uap_path), device=self.device, dtype=self.attack.dtype)
        #     # Create a new tensor with the required values
        #     new_attack = self.attack.clone()
        #     new_attack[self.attack_mask] = uap_tensor
            
        #     # Assign the new tensor to self.attack to avoid in-place operation
        #     self.attack = torch.nn.Parameter(new_attack)    

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
    def save_uap(self, path):
        print("save uap")
        temp_mask = self.attack_mask[:, :]
        temp_attack = self.attack[:, :, :]
        np.save(path, self.attack[self.attack_mask[:, :]].to(torch.float32).detach().cpu().numpy())
    
    def calculate_top_k_embedding_loss(self, k=10):
        loss = 0.0
        for i in range(self.attack.shape[1]):
            # 计算 self.attack 与 embedding_space 中最相似的前 k 个向量的距离
            distances = torch.norm(self.embedding_space - self.attack[0, i], dim=-1)
            top_k_indices = torch.topk(distances, k, largest=False).indices
            top_k_embeddings = self.embedding_space[top_k_indices]
            
            # 计算目标向量与 top_k_embeddings 的余弦相似度损失
            cosine_similarity = nn.CosineSimilarity(dim=-1)
            similarity_scores = cosine_similarity(self.attack[0, i].unsqueeze(0), top_k_embeddings)
            
            # 最小化相似度的负值（即最大化相似度）
            top_k_loss = -torch.mean(similarity_scores)
            loss += top_k_loss    
        return loss
    def _compute_losses(self, candidate_embeddings,model,index):
        original_attack = self.attack.detach().clone()
        self.attack[0, index] = candidate_embeddings.to(self.device)
        with torch.no_grad():
            losses = compute_toward_away_loss_adversary(
                model,
                self.batch["adv_padd_prompt_tokens"],
                self.batch["adv_padd_labels_mask"],
                self.batch["def_padd_prompt_tokens"],
                self.batch["def_padd_labels_mask"],
                self.batch.get("adv_labels"),
                self.batch.get("def_labels"),
                {"toward": 1.0, "away": 1.0},
            )
            total_loss = losses["total"]

        self.attack[0] = original_attack
        del original_attack

        return total_loss

    def select_best_embedding(self, k=10, model=None):
        with torch.no_grad():
            best_embeddings = self.attack.clone()
            attack_indices = self.attack_mask[0].nonzero(as_tuple=True)[0]
            for i in attack_indices:
                print(i)
                distances = torch.norm(self.embedding_space - self.attack[0, i], dim=-1)
                top_k_indices = torch.topk(distances, k, largest=False).indices
                top_k_embeddings = self.embedding_space[top_k_indices]
                # 计算每个候选嵌入的总损失
                total_losses = torch.tensor([
                    self._compute_losses(candidate.unsqueeze(0), model=model, index = i) for candidate in top_k_embeddings
                ], device=self.device)
                
                best_idx = torch.argmin(total_losses)
                best_embeddings[0, i] = top_k_embeddings[best_idx]
                del distances, top_k_indices, top_k_embeddings, total_losses
            
            self.attack.copy_(best_embeddings)
            del best_embeddings

    def compute_away_loss(self,model):
        away_tokens = self.batch["def_tokens"].to(self.device)
        away_labels_mask = self.batch["def_labels_mask"].to(self.device)
        logits = model(input_ids=away_tokens).logits
        final_logits = logits[:, :-1][away_labels_mask[:, 1:].to(self.device)]
        away_labels = away_tokens[:, 1:][away_labels_mask[:, 1:].to(self.device)]
        loss = nn.CrossEntropyLoss()(final_logits, away_labels)
        del final_logits, away_labels, logits
        return loss.item()

    def compute_toward_loss(self, model):
        toward_tokens = self.batch["adv_tokens"].to(self.device)
        toward_labels_mask = self.batch["adv_labels_mask"].to(self.device)
        logits = model(input_ids=toward_tokens).logits
        final_logits = logits[:, :-1][toward_labels_mask[:, 1:].to(self.device)]
        toward_labels = toward_tokens[:, 1:][toward_labels_mask[:, 1:].to(self.device)]
        loss = nn.CrossEntropyLoss()(final_logits, toward_labels)
        del final_logits, toward_labels, logits

        return loss.item()

    def clip_attack(self,select_best=False, model=None):
        if select_best and model != None:
            print("new clip attack")
            self.select_best_embedding(k=10, model=model)
        else:
            if self.embedding_space == None:
                with torch.no_grad():
                    # clip attack norm to eps
                    norms = torch.norm(self.attack, dim=-1, keepdim=True)
                    scale = torch.clamp(norms / self.epsilon, min=1)
                    self.attack.div_(scale)

                    norms = torch.norm(self.attack, dim=-1)
            else:
                print("start clipping")
                with torch.no_grad():
                # 将 self.attack 的值限制为 embedding_space 中的值
                    closest_embeddings = torch.zeros_like(self.attack)
                    for i in range(self.attack.shape[1]):
                        distances = torch.norm(self.embedding_space - self.attack[0, i], dim=-1)
                        closest_idx = torch.argmin(distances)
                        closest_embeddings[0, i] = self.embedding_space[closest_idx]
                    self.attack.copy_(closest_embeddings)

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