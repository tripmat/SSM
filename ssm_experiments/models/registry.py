from transformers import GPTNeoXForCausalLM, GPTNeoXConfig

from .lstm import LSTM


def get_model(args, tokenizer):
    """Factory that returns a model based on args.model

    Supported models in the slim package:
    - "T_rope": GPT-NeoX with RoPE (built-in transformers)
    - "mamba": real Mamba if installed, otherwise paper-informed CPU Mamba
    - "lstm": simple LSTM baseline (optional)
    """

    if args.model == "T_rope":
        config = GPTNeoXConfig(
            bos_token_id=0,
            eos_token_id=0,
            hidden_size=args.hidden_size,
            intermediate_size=args.hidden_size * 4,
            num_attention_heads=args.heads,
            num_hidden_layers=args.layers,
            vocab_size=len(tokenizer),
        )
        model = GPTNeoXForCausalLM(config)

    elif args.model == "mamba":
        try:
            from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
            print("Using real Mamba implementation")
        except ImportError:
            from .paper_mamba import PaperMambaLMHeadModel as MambaLMHeadModel
            print("Using paper-informed Mamba implementation (CPU-only)")

        model = MambaLMHeadModel(
            d_model=args.hidden_size,
            n_layer=args.layers,
            ssm_cfg={"d_state": getattr(args, "state_dim", 32)},
            vocab_size=len(tokenizer),
        )

    elif args.model == "lstm":
        model = LSTM(
            embedding_dim=args.hidden_size,
            vocab_size=len(tokenizer),
            num_layers=args.layers,
            dropout_rate=0.65,
        )

    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    return model

