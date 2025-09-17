"""
Paper-informed Mamba implementation for CPU-only environments.
Renamed from mock_mamba.py to paper_mamba.py for clarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperInformedMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=32):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=4, padding=2, groups=d_model)

        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.dt_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.A_log, mean=-2.0, std=0.1)
        nn.init.normal_(self.B_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.C_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.dt_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)

    def forward(self, x):
        batch_size, L, D = x.shape
        residual = x

        x_and_z = self.in_proj(x)
        x, z = x_and_z.chunk(2, dim=-1)

        x_conv = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        x = F.silu(x_conv)

        A_log_clipped = torch.clamp(self.A_log.float(), min=-5.0, max=2.0)
        A = -torch.exp(A_log_clipped)
        B = self.B_proj(x)
        C = self.C_proj(x)

        states = torch.zeros((batch_size, D, self.d_state), device=x.device, dtype=x.dtype)
        outputs = []

        for i in range(L):
            state_update = states * A.unsqueeze(0) + B[:, i].unsqueeze(1) * x[:, i].unsqueeze(-1)
            states = torch.clamp(state_update, min=-10.0, max=10.0)
            y = torch.sum(C[:, i].unsqueeze(1) * states, dim=-1)
            outputs.append(y)

        scan_output = torch.stack(outputs, dim=1)
        scan_output = scan_output * F.silu(z)
        output = self.out_proj(scan_output)
        output = self.norm(output + residual)
        return output


class PaperInformedMambaLMHeadModel(nn.Module):
    def __init__(self, d_model, n_layer, vocab_size, ssm_cfg=None):
        super().__init__()

        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size

        d_state = 32
        if ssm_cfg and "d_state" in ssm_cfg:
            d_state = ssm_cfg["d_state"]

        print(f"Using paper-informed Mamba with d_state={d_state}")

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([PaperInformedMambaBlock(d_model, d_state) for _ in range(n_layer)])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids, **kwargs):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(x)
        return (logits,)

    def generate(self, *args, **kwargs):
        raise NotImplementedError("Generation not implemented for paper-informed Mamba")


# Friendlier alias
PaperMambaLMHeadModel = PaperInformedMambaLMHeadModel

