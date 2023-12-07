import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, noise_text_ratio=0.0, normalize=False):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.noise_text_ratio = noise_text_ratio
        self.normalize = normalize




    def forward(self, patch_embedding, img_embedding, gt_text_embedding_map,text_prompt_mask, noise_text_embeddings, gt_density):

        
        text_prompt_mask_=torch.stack(text_prompt_mask, dim=1) # [B, 10]
        gt_text_embedding_map_=torch.stack(gt_text_embedding_map, dim=1) # [B, 10, 1, 512]
        

        gt_text_embedding_map_=gt_text_embedding_map_.squeeze(2) # [B, 10, 512]
        # text_prompt_mask_modified = text_prompt_mask_.clone()
        # text_prompt_mask_modified[:, 1:] = text_prompt_mask_modified[:, 1:]| text_prompt_mask_modified[:, :-1] # 注意最后一个如果是1，它左边的没变1，是个缺陷，要改进
        n_pos = torch.sum(text_prompt_mask_, dim=-1)
        sim_map = F.cosine_similarity(img_embedding, gt_text_embedding_map_ , dim=-1) # (B, 10)
        sim_map = torch.exp(sim_map / self.temperature)
        text_prompt_mask_=text_prompt_mask_.bool()
        pos_sum = torch.sum(torch.where(text_prompt_mask_, sim_map, torch.zeros_like(sim_map)), dim=-1) + 1e-5
        neg_sum = torch.sum(torch.where(~text_prompt_mask_, sim_map, torch.zeros_like(sim_map)), dim=-1) + 1e-5
        loss = -torch.log(pos_sum / (pos_sum + neg_sum))
        if self.normalize:
            loss = loss / n_pos            
        return loss.mean()
