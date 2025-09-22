"""
fixed_optimized_mamba.py - Maintains learning dynamics while optimizing performance
Key fixes: A matrix is learnable, proper residual connections, same math as paper version
Now with Windows Triton support!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


class FastPaperMambaBlock(nn.Module):
    """Fast version that preserves the exact learning dynamics of paper version"""
    
    def __init__(self, d_model, d_state=32):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Same projections as paper version
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=4, padding=2, groups=d_model)

        # CRITICAL: A must be learnable parameter, not buffer!
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.dt_proj = nn.Linear(d_model, d_model)
        self.dt_min = 1e-3
        self.dt_max = 0.2

        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()
        
        # Try to compile the scan function if possible
        self._compiled_scan = None
        self._compile_attempted = False

    def _init_weights(self):
        # Exact same initialization as paper version
        nn.init.normal_(self.A_log, mean=-2.0, std=0.1)
        nn.init.normal_(self.B_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.C_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.dt_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)

    def forward(self, x):
        batch_size, L, D = x.shape
        residual = x

        # Same forward pass structure as paper version
        x_and_z = self.in_proj(x)
        x, z = x_and_z.chunk(2, dim=-1)

        x_conv = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        x = F.silu(x_conv)

        # Same A matrix computation
        A_log_clipped = torch.clamp(self.A_log.float(), min=-5.0, max=2.0)
        A = -torch.exp(A_log_clipped)
        B = self.B_proj(x)
        C = self.C_proj(x)
        dt = F.softplus(self.dt_proj(x))
        dt = torch.clamp(dt, min=self.dt_min, max=self.dt_max)

        # Compile for sequences >= 10 tokens (your minimum training length)
        if L >= 10 and not self._compile_attempted:
            self._try_compile()
        
        if self._compiled_scan is not None and L >= 10:
            try:
                scan_output = self._compiled_scan(x, dt, A, B, C, batch_size, L, D)
            except Exception as e:
                # Fallback if compilation fails at runtime
                warnings.warn(f"Compiled scan failed, falling back to simple scan: {e}")
                self._compiled_scan = None
                scan_output = self._simple_scan(x, dt, A, B, C, batch_size, L, D)
        else:
            scan_output = self._simple_scan(x, dt, A, B, C, batch_size, L, D)

        # Same output computation
        scan_output = scan_output * F.silu(z)
        output = self.out_proj(scan_output)
        output = self.norm(output + residual)  # Same residual + norm
        return output

    def _try_compile(self):
        """Attempt to compile the scan function once"""
        self._compile_attempted = True
        try:
            if hasattr(torch, 'compile'):
                # Try importing triton to check if it's available
                try:
                    import triton
                    print("Triton detected, attempting to compile scan function...")
                except ImportError:
                    try:
                        import triton_windows as triton
                        print("Triton-windows detected, attempting to compile scan function...")
                    except ImportError:
                        print("Triton not available, using uncompiled scan")
                        return
                
                # Create compiled version
                self._compiled_scan = torch.compile(self._simple_scan, mode='default')
                print("Successfully compiled scan function!")
        except Exception as e:
            warnings.warn(f"Failed to compile scan function: {e}")
            self._compiled_scan = None

    def _simple_scan(self, x, dt, A, B, C, batch_size, L, D):
        """Exact same scan as paper version - preserves learning dynamics"""
        states = torch.zeros((batch_size, D, self.d_state), device=x.device, dtype=x.dtype)
        outputs = []

        for i in range(L):
            dt_i = dt[:, i]  # (B, D)
            A_bar = torch.exp(dt_i.unsqueeze(-1) * A.unsqueeze(0))        # (B, D, d_state)
            B_bar = dt_i.unsqueeze(-1) * B[:, i].unsqueeze(1)             # (B, D, d_state)
            state_update = states * A_bar + B_bar * x[:, i].unsqueeze(-1)  # (B, D, d_state)
            states = torch.clamp(state_update, min=-10.0, max=10.0)
            y = torch.sum(C[:, i].unsqueeze(1) * states, dim=-1)
            outputs.append(y)

        return torch.stack(outputs, dim=1)


class FastPaperMambaLMHeadModel(nn.Module):
    """Fast version preserving exact paper dynamics"""
    
    def __init__(self, d_model, n_layer, vocab_size, d_state=32, **kwargs):
        super().__init__()

        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size

        print(f"Using FAST paper-informed Mamba with d_state={d_state}")
        print(f"d_model={d_model}, n_layer={n_layer}")

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            FastPaperMambaBlock(d_model, d_state) 
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
        raise NotImplementedError("Generation not implemented")


# Alternative optimization approaches for longer sequences
class ChunkedPaperMambaBlock(FastPaperMambaBlock):
    """Chunked processing for memory efficiency on very long sequences"""
    
    def forward(self, x):
        batch_size, L, D = x.shape
        
        # For very long sequences, process in chunks to save memory
        if L > 512:
            return self._chunked_forward(x)
        else:
            return super().forward(x)
    
    def _chunked_forward(self, x, chunk_size=256):
        """Process in chunks, maintaining state across chunks"""
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
        dt = F.softplus(self.dt_proj(x))
        dt = torch.clamp(dt, min=self.dt_min, max=self.dt_max)
        
        # Process in chunks, carrying state forward
        states = torch.zeros((batch_size, D, self.d_state), device=x.device, dtype=x.dtype)
        all_outputs = []
        
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            chunk_x = x[:, start:end]
            chunk_dt = dt[:, start:end]
            chunk_B = B[:, start:end]
            chunk_C = C[:, start:end]
            
            # Process chunk
            chunk_outputs = []
            for i in range(end - start):
                dt_i = chunk_dt[:, i]
                A_bar = torch.exp(dt_i.unsqueeze(-1) * A.unsqueeze(0))
                B_bar = dt_i.unsqueeze(-1) * chunk_B[:, i].unsqueeze(1)
                states = states * A_bar + B_bar * chunk_x[:, i].unsqueeze(-1)
                states = torch.clamp(states, min=-10.0, max=10.0)
                y = torch.sum(chunk_C[:, i].unsqueeze(1) * states, dim=-1)
                chunk_outputs.append(y)
            
            all_outputs.extend(chunk_outputs)
        
        scan_output = torch.stack(all_outputs, dim=1)
        scan_output = scan_output * F.silu(z)
        output = self.out_proj(scan_output)
        output = self.norm(output + residual)
        return output


# Export all expected names - match your original optimized_paper_mamba.py
FastMambaLMHeadModel = FastPaperMambaLMHeadModel
MambaForCopying30 = FastPaperMambaLMHeadModel
OptimizedMambaLMHeadModel = FastPaperMambaLMHeadModel
MambaLMHeadModel = FastPaperMambaLMHeadModel
PaperMambaLMHeadModel = FastPaperMambaLMHeadModel

__all__ = [
    'FastPaperMambaLMHeadModel',
    'MambaForCopying30',
    'FastMambaLMHeadModel', 
    'OptimizedMambaLMHeadModel',
    'MambaLMHeadModel',
    'PaperMambaLMHeadModel',
    'ChunkedPaperMambaBlock'
]