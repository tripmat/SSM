"""
Minimal Mamba implementation adapted from mamba-minimal repository.
Integrated into SSM experiments framework with proper interface compatibility.

Original implementation: https://github.com/johnma2006/mamba-minimal
Based on "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
by Albert Gu and Tri Dao (https://arxiv.org/abs/2312.00752)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Union
from einops import rearrange, repeat, einsum


@dataclass
class MinimalMambaArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    dt_min: float = 0.001
    dt_max: float = 0.1

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


class MinimalMamba(nn.Module):
    def __init__(self, args: MinimalMambaArgs):
        """Full Mamba model following the minimal implementation."""
        super().__init__()
        self.args = args

        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)

        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights

    def forward(self, input_ids):
        """
        Args:
            input_ids (long tensor): shape (b, l)

        Returns:
            logits: shape (b, l, vocab_size)
        """
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        return logits


class ResidualBlock(nn.Module):
    def __init__(self, args: MinimalMambaArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)

        Returns:
            output: shape (b, l, d)
        """
        output = self.mixer(self.norm(x)) + x
        return output


class MambaBlock(nn.Module):
    def __init__(self, args: MinimalMambaArgs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper."""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        """Mamba block forward following Figure 3 in the Mamba paper.

        Args:
            x: shape (b, l, d)

        Returns:
            output: shape (b, l, d)
        """
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')

        x = F.silu(x)

        y = self.ssm(x)

        y = y * F.silu(res)

        output = self.out_proj(y)

        return output

    def ssm(self, x):
        """Runs the SSM following Algorithm 2 in Section 3.2 in the Mamba paper.

        Args:
            x: shape (b, l, d_in)

        Returns:
            output: shape (b, l, d_in)
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        delta = delta.clamp(min=self.args.dt_min, max=self.args.dt_max)

        y = self.selective_scan(x, delta, A, B, C, D)

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm following Section 2 and Algorithm 2 in the Mamba paper.

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C are dependent on the input x(t).

        Args:
            u: shape (b, l, d_in)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)

        Returns:
            output: shape (b, l, d_in)
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # Discretize continuous parameters (A, B)
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        # Perform selective scan (sequential implementation for CPU compatibility)
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D

        return y


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


class MinimalMambaLMHeadModel(nn.Module):
    """
    Adapter class to match the interface expected by the SSM experiments framework.
    Wraps the MinimalMamba implementation with the required interface.
    """
    def __init__(self, d_model, n_layer, vocab_size, ssm_cfg=None):
        super().__init__()

        # Extract configuration parameters
        d_state = (ssm_cfg or {}).get("d_state", 16)
        dt_min = (ssm_cfg or {}).get("dt_min", 0.001)
        dt_max = (ssm_cfg or {}).get("dt_max", 0.1)
        expand = (ssm_cfg or {}).get("expand", 2)
        d_conv = (ssm_cfg or {}).get("d_conv", 4)

        print(f"Using minimal Mamba with d_state={d_state}, expand={expand}, dt_range=[{dt_min}, {dt_max}]")

        # Create MinimalMambaArgs
        args = MinimalMambaArgs(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            dt_min=dt_min,
            dt_max=dt_max,
        )

        # Create the actual Mamba model
        self.model = MinimalMamba(args)

        # Store config for compatibility
        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size

    def forward(self, input_ids, **kwargs):
        """Forward pass compatible with the SSM experiments framework."""
        logits = self.model(input_ids)
        return (logits,)

    def generate(self, *args, **kwargs):
        """Generation method (not implemented for experiments)."""
        raise NotImplementedError("Generation not implemented for minimal Mamba")