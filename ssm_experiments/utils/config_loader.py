import os
import json
import importlib.util


def load_config(path):
    """Load an experiment config from a Python file only.

    The Python module must define one of:
      - CONFIG: dict
      - get_config(): returns dict
      - module-level dicts: global/globals and models

    Returns a plain dict suitable for JSON serialization (used for logging).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".py":
        spec = importlib.util.spec_from_file_location("_ssm_user_config", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to import Python config: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        cfg = None
        if hasattr(module, "CONFIG"):
            cfg = getattr(module, "CONFIG")
        elif hasattr(module, "get_config"):
            candidate = module.get_config()
            cfg = candidate
        else:
            # Fallback: collect module-level dicts
            cfg = {}
            g = getattr(module, "global", None)
            if g is None:
                g = getattr(module, "globals", None)
            m = getattr(module, "models", None)
            if g is not None:
                cfg["global"] = g
            if m is not None:
                cfg["models"] = m

        if not isinstance(cfg, dict):
            raise TypeError("Python config must define CONFIG dict, get_config() returning dict, or module-level dicts 'global'/'models'.")

        # Best-effort ensure JSON-serializable (raises if not)
        try:
            json.dumps(cfg)
        except TypeError as e:
            raise TypeError(f"Config contains non-serializable types: {e}")
        return cfg

    raise ValueError(f"Unsupported config type '{ext}'. Only .py is supported now.")
