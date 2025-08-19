## slime API Reference

This section documents the public APIs, functions, and components in slime. It is organized by module so you can quickly find what you need, with runnable examples and usage notes.

- See the Usage Guide for CLI-based workflows: docs/en/usage.md
- For training flow, refer to train.py and docs in this API reference:
  - Ray integration: docs/en/api/ray.md
  - Rollout components: docs/en/api/rollout.md
  - Utilities: docs/en/api/utils.md
  - Backends (Megatron + SGLang): docs/en/api/backends.md
  - Plugins (Buffer, Models, Bridges): docs/en/api/plugins.md
  - Tools and conversion scripts: docs/en/api/tools.md

### Minimal programmatic example

```python
from slime.utils.arguments import parse_args
from train import train

if __name__ == "__main__":
    args = parse_args()
    train(args)
```

This uses the same arguments and runtime as the CLI. To customize rollout behavior, set `--rollout-function-path` or `--custom-generate-function-path`. For all arguments, see slime/utils/arguments.py.

