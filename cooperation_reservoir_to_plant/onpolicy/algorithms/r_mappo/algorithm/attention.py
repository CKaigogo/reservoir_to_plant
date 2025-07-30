import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x, mask=None):
        # x shape: [batch_size, num_agents, hidden_size]
        attended, _ = self.attention(x, x, x, attn_mask=mask)
        return self.norm(attended + x)  # 残差连接

class WaterAttentionPolicy(nn.Module):
    def __init__(self, obs_dim, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU()
        )
        self.attention = AttentionLayer(hidden_size)
        self.decoder = nn.Linear(hidden_size, 1)  # 动作空间
        
    def forward(self, obs):
        # 编码每个智能体的观察
        encoded = self.encoder(obs)  # [batch_size, num_agents, hidden_size]
        
        # 应用注意力机制
        attended = self.attention(encoded)
        
        # 解码得到动作
        actions = self.decoder(attended)
        return actions 