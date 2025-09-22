from transformers import GPTNeoXForCausalLM, GPTNeoXConfig
from .hard_alibi_model import GPTNeoXHardAlibiForCausalLM
from .alibi_model import GPTNeoXAlibiForCausalLM
from .nope import GPTNeoXNoPEForCausalLM
from .optimized_paper_mamba import MambaForCopying30
from .paper_mamba import PaperMambaLMHeadModel


def get_model(args, tokenizer):
    """Factory that returns a model based on args.model

    Supported models in the slim package:
    - "T_rope": GPT-NeoX with RoPE (built-in transformers)
    - "mamba": Transformers Mamba implementation
    - "paper_mamba": Optimized paper-informed CPU Mamba
    
    """

    if args.model in ["T_rope", "T_nope", "T_alibi"]:
        config = GPTNeoXConfig(
            bos_token_id=0,
            eos_token_id=0,
            hidden_size=args.hidden_size,
            intermediate_size=args.hidden_size * 4,
            num_attention_heads=args.heads,
            num_hidden_layers=args.layers,
            vocab_size=len(tokenizer),
        )
    elif args.model == "T_hard_alibi":
        config = GPTNeoXConfig(
            bos_token_id=0,
            eos_token_id=0,
            hidden_size=args.hidden_size,
            intermediate_size=args.hidden_size * 4,
            num_attention_heads=args.heads,
            num_hidden_layers=args.layers,
            num_masked_heads=getattr(args, "num_masked_heads", 4),
            min_window_size=getattr(args, "min_window_size", 100),
            vocab_size=len(tokenizer),
        )

    if args.model == "T_rope":
        model = GPTNeoXForCausalLM(config)
    elif args.model == "T_nope":
        model = GPTNeoXNoPEForCausalLM(config)
    elif args.model == "T_alibi":
        model = GPTNeoXAlibiForCausalLM(config)
    elif args.model == "T_hard_alibi":
        model = GPTNeoXHardAlibiForCausalLM(config)

    elif args.model == "mamba":
        # Alias to optimized paper-informed implementation to avoid real mamba_ssm dependency
        model = MambaForCopying30(
            d_model=args.hidden_size,
            n_layer=args.layers,
            d_state=getattr(args, "state_dim", 16),
            vocab_size=len(tokenizer),
        )
        print("Using OPTIMIZED paper-informed Mamba (alias for 'mamba')")

    elif args.model == "paper_mamba":
        model = PaperMambaLMHeadModel(
            d_model=args.hidden_size,
            n_layer=args.layers,
            ssm_cfg={"d_state": getattr(args, "state_dim", 16)},
            vocab_size=len(tokenizer),
        )
        print("Using paper-informed Mamba (CPU-friendly implementation)")

    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    return model

