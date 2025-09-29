from transformers import GPTNeoXForCausalLM, GPTNeoXConfig
from .hard_alibi_model import GPTNeoXHardAlibiForCausalLM
from .hard_alibi_rope_model import GPTNeoXHardAlibiRopeForCausalLM
from .alibi_model import GPTNeoXAlibiForCausalLM
from .nope import GPTNeoXNoPEForCausalLM
from .paper_mamba import PaperMambaLMHeadModel
from .minimal_mamba import MinimalMambaLMHeadModel


def get_model(args, tokenizer):
    """Factory that returns a model based on args.model

    Supported models in the slim package:
    - "T_rope": GPT-NeoX with RoPE (built-in transformers)
    - "paper_mamba": Optimized paper-informed CPU Mamba
    - "minimal_mamba": Reference Mamba implementation from mamba-minimal

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
            rotary_pct=0.0,
            vocab_size=len(tokenizer),
        )
    elif args.model in ["T_hard_alibi_rope", "transformer_hard_alibi_rope"]:
        config = GPTNeoXConfig(
            bos_token_id=0,
            eos_token_id=0,
            hidden_size=args.hidden_size,
            intermediate_size=args.hidden_size * 4,
            num_attention_heads=args.heads,
            num_hidden_layers=args.layers,
            num_masked_heads=getattr(args, "num_masked_heads", 4),
            rotary_pct=getattr(args, "rotary_pct", 1.0),
            rotary_emb_base=getattr(args, "rotary_emb_base", 10000),
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
    elif args.model in ["T_hard_alibi_rope", "transformer_hard_alibi_rope"]:
        model = GPTNeoXHardAlibiRopeForCausalLM(config)

    elif args.model == "paper_mamba":
        model = PaperMambaLMHeadModel(
            d_model=args.hidden_size,
            n_layer=args.layers,
            ssm_cfg={
                "d_state": getattr(args, "state_dim", 16),
                "dt_min": getattr(args, "dt_min", 1e-3),
                "dt_max": getattr(args, "dt_max", 0.2),
            },
            vocab_size=len(tokenizer),
        )
        print("Using paper-informed Mamba (CPU-friendly implementation)")

    elif args.model == "minimal_mamba":
        model = MinimalMambaLMHeadModel(
            d_model=args.hidden_size,
            n_layer=args.layers,
            ssm_cfg={
                "d_state": getattr(args, "state_dim", 16),
                "dt_min": getattr(args, "dt_min", 1e-3),
                "dt_max": getattr(args, "dt_max", 0.1),
                "expand": getattr(args, "expand", 2),
                "d_conv": getattr(args, "d_conv", 4),
            },
            vocab_size=len(tokenizer),
        )
        print("Using minimal Mamba (reference implementation from mamba-minimal)")

    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    return model

