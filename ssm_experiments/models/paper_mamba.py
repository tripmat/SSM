"""
Paper-informed Mamba implementation for CPU-only environments.
Renamed from mock_mamba.py to paper_mamba.py for clarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperInformedMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=32, dt_min=1e-3, dt_max=0.2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, d_model * 2)
        # Causal local mixer: no built-in padding; we will apply left-only padding in forward
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=4, padding=0, groups=d_model)

        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        # Depthwise 1x1 convs for per-channel B and C (parameter-efficient)
        self.B_proj = nn.Conv1d(d_model, d_model * d_state, kernel_size=1, groups=d_model, bias=True)
        self.C_proj = nn.Conv1d(d_model, d_model * d_state, kernel_size=1, groups=d_model, bias=True)
        # Per-channel learned timestep (Δt) projection used to discretize the SSM
        self.dt_proj = nn.Linear(d_model, d_model)
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)

        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.A_log, mean=-2.0, std=0.1)
        if hasattr(self.B_proj, 'weight'):
            nn.init.normal_(self.B_proj.weight, mean=0.0, std=0.02)
        if hasattr(self.C_proj, 'weight'):
            nn.init.normal_(self.C_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.dt_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)

    def forward(self, x):
        batch_size, L, D = x.shape
        residual = x

        x_and_z = self.in_proj(x)
        x, z = x_and_z.chunk(2, dim=-1)

        # Left-only pad to enforce causality and keep length: pad (kernel_size-1) on the left
        xt = x.transpose(1, 2)
        k = self.conv1d.kernel_size[0]
        xt = F.pad(xt, (k - 1, 0))
        x_conv = self.conv1d(xt).transpose(1, 2)
        x = F.silu(x_conv)

        A_log_clipped = torch.clamp(self.A_log.float(), min=-5.0, max=2.0)
        # Continuous-time negative rates (stable)
        A = -torch.exp(A_log_clipped)

        # Compute per-channel B and C efficiently via depthwise 1x1 convs
        xt = x.transpose(1, 2)  # (B, D, L)
        B_all = self.B_proj(xt)  # (B, D*d_state, L)
        C_all = self.C_proj(xt)  # (B, D*d_state, L)
        B = B_all.transpose(1, 2).reshape(batch_size, L, D, self.d_state)
        C = C_all.transpose(1, 2).reshape(batch_size, L, D, self.d_state)

        # Learn per-token, per-channel timestep and discretize
        dt = F.softplus(self.dt_proj(x))
        dt = torch.clamp(dt, min=self.dt_min, max=self.dt_max)  # (B, L, D)

        states = torch.zeros((batch_size, D, self.d_state), device=x.device, dtype=x.dtype)
        outputs = []

        for i in range(L):
            dt_i = dt[:, i]  # (B, D)
            # Discretize dynamics: A_bar = exp(clamp(dt * A, [-10, 0])) for stability
            exp_arg = dt_i.unsqueeze(-1) * A.unsqueeze(0)
            A_bar = torch.exp(torch.clamp(exp_arg, min=-10.0, max=0.0))   # (B, D, d_state)
            B_i = B[:, i]                                                # (B, D, d_state)
            C_i = C[:, i]                                                # (B, D, d_state)
            B_bar = dt_i.unsqueeze(-1) * B_i                              # (B, D, d_state)
            state_update = states * A_bar + B_bar * x[:, i].unsqueeze(-1) # (B, D, d_state)
            states = torch.clamp(state_update, min=-10.0, max=10.0)
            y = torch.sum(C_i * states, dim=-1)
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
        dt_min = (ssm_cfg or {}).get("dt_min", 1e-3)
        dt_max = (ssm_cfg or {}).get("dt_max", 0.2)

        print(f"Using paper-informed Mamba with d_state={d_state}")

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            PaperInformedMambaBlock(d_model, d_state, dt_min=dt_min, dt_max=dt_max)
            for _ in range(n_layer)
        ])
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
