import torch

import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        
        self.num_blocks = d_k // 2
        k_idx = torch.arange(0, self.num_blocks, dtype=torch.float32)
        exponents = -(k_idx) / self.num_blocks
        freqs = theta**(exponents)
        positions = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        theta_i_k = freqs * positions
        
        cos_theta = torch.cos(theta_i_k)  # 形状(seq_len, num_blocks)
        sin_theta = torch.sin(theta_i_k)  # 形状(seq_len, num_blocks)
        
        self.register_buffer("cos_cached", cos_theta, persistent=False)
        self.register_buffer("sin_cached", sin_theta, persistent=False)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        d_k = x.shape[-1]
        d_pair = x.shape[-1] // 2
        seq_len = x.shape[-2]
        # 关键修正：获取前置维度，构造新形状
        prefix_dims = x.shape[:-2]  # 获取除了最后两维之外的所有维度（比如batch、heads等）
        new_shape = (*prefix_dims, seq_len, d_pair, 2)  # 组合成新形状：(前置维度, seq_len, d_pair, 2)
        x_pair = x.view(new_shape)  # 用新形状元组调用view()
        x, y = x_pair.unbind(-1)
        
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        cos = cos.unsqueeze(0).unsqueeze(-1)  # 增加batch维度和最后一维
        sin = sin.unsqueeze(0).unsqueeze(-1)
        
        # 计算旋转后的x'和y'
        x_rot = x * cos.squeeze(-1) - y * sin.squeeze(-1)  # squeeze(-1)去掉最后一维1
        y_rot = x * sin.squeeze(-1) + y * cos.squeeze(-1)
        
        q_rot = torch.stack((x_rot, y_rot), dim=-1)
        ans = q_rot.view(*prefix_dims, seq_len, d_k)
        return ans.to(in_type)
