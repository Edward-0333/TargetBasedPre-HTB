import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAgentEncoder(nn.Module):
    def __init__(self, in_dim=128, hid=256, out_dim=256, T=12, n_layers=2):
        super().__init__()
        # 用 Transformer 生成 T 个“未来查询”的隐状态；也可换 TCN/GRU
        self.T = T
        self.q_time = nn.Parameter(torch.randn(T, out_dim))  # [T, D]
        self.proj_in = nn.Linear(in_dim, out_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=out_dim, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, A):  # A: [B,N,128]
        B,N,_ = A.shape
        x = self.proj_in(A)                       # [B,N,D]
        # 让每个 agent 与 T 个“时间查询”交互：简单做法是把时间查询加到 x 再堆叠
        # 这里用一种轻量 trick：复制 N 次的时间查询并与 x 融合（也可以做 cross-attn）
        q = self.q_time.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)  # [B,N,T,D]
        x = x.unsqueeze(2).expand(B, N, self.T, -1) + q                 # [B,N,T,D]
        # 合并维度过编码器
        x = self.encoder(x.reshape(B*N, self.T, -1))                    # [B*N,T,D]
        return x.reshape(B, N, self.T, -1)                              # [B,N,T,D]

class LaneEncoder(nn.Module):
    def __init__(self, in_dim=128, out_dim=256, n_layers=2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, out_dim), nn.ReLU(), nn.LayerNorm(out_dim)]
            d = out_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, Bx):  # Bx: [B,K,128]
        return self.mlp(Bx) # [B,K,D]

class BilinearScorer(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d, d) * (1.0/d**0.5))
        self.ua = nn.Parameter(torch.zeros(d))
        self.ul = nn.Parameter(torch.zeros(d))
        self.c  = nn.Parameter(torch.zeros(1))

    def forward(self, Ha, Hl):  # Ha:[B,N,T,D], Hl:[B,K,D]
        # logits[b,n,t,k] = Ha[b,n,t] W Hl[b,k] + biases
        B,N,T,D = Ha.shape
        K = Hl.shape[1]
        # (Ha @ W): [B,N,T,D]
        HW = torch.einsum('bntd,df->bntf', Ha, self.W)
        # scores core: [B,N,T,K]
        core = torch.einsum('bntf,bkf->bntk', HW, Hl)
        # add linear terms
        core = core + torch.einsum('d,bntd->bnt', self.ua, Ha).unsqueeze(-1)
        core = core + torch.einsum('d,bkd->bk', self.ul, Hl).unsqueeze(1).unsqueeze(1)
        core = core + self.c
        return core  # logits: [B,N,T,K]

class LinearScorerLayer(nn.Module):
    def __init__(self, T=12, d=256):
        super().__init__()
        self.agent_enc = TemporalAgentEncoder(in_dim=64, out_dim=d, T=T)
        self.lane_enc  = LaneEncoder(in_dim=64, out_dim=d)
        self.scorer    = BilinearScorer(d=d)

    def forward(self, A, L, agent_mask=None, lane_mask=None, time_mask=None):
        # A:[B,N,64], L:[B,K,64]
        Ha = self.agent_enc(A)  # [B,N,T,D]
        Hl = self.lane_enc(L)  # [B,K,D]
        logits = self.scorer(Ha, Hl)  # [B,N,T,K]

        # lane 屏蔽（防止无效 lane 参与 softmax）
        temp = (lane_mask).unsqueeze(1).unsqueeze(1)
        if lane_mask is not None:  # lane_mask:[B,K] bool
            logits = logits.masked_fill((lane_mask).unsqueeze(1).unsqueeze(1), float('-inf'))

        # agent 屏蔽
        if agent_mask is not None:  # agent_mask:[B,N, T] bool
            invalid_agents = (agent_mask).unsqueeze(-1)  # [B,N,1,1]
            logits = logits + invalid_agents.float() * (-1e9)  # [B,N,T,K]

        probs = logits.softmax(dim=-1)  # [B,N,T,K]

        # # 如果希望在输出的 probs 中直接把无效 agent 的概率置 0（不强制归一）
        # if agent_mask is not None:
        #     temp2 = (~agent_mask).unsqueeze(-1).unsqueeze(-1).float()
        #     probs = probs * (~agent_mask).unsqueeze(-1).unsqueeze(-1).float()  # [B,N,T,K]
        return logits, probs

