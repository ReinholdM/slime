## Plugins

### Rollout Buffer Server (slime_plugins/rollout_buffer)

An external HTTP buffer service that groups and returns batched rollout items from arbitrary generators.

Endpoints:
- POST /buffer/write {instance_id: str, prompt: ..., response: ..., reward: float, ...}
- POST /get_rollout_data -> {data: [...], meta_info: {...}}
- POST /start_rollout -> starts background data generator

Core classes:
- BufferQueue(group_size, task_type, transform_group_func=None, is_valid_group_func=None, get_group_data_meta_info_func=None)
  - append(item), get() -> grouped items and meta_info

- RolloutBuffer(group_size, task_type, ...)
  - write(data), read()

Generator discovery:
- Files in `slime_plugins/rollout_buffer/generator/*.py` with constants/functions:
  - TASK_TYPE = "math" (example)
  - run_rollout(data)
  - Optional: transform_group, is_valid_group, get_group_data_meta_info

Quick start:
```bash
python -m slime_plugins.rollout_buffer.buffer
# then POST /start_rollout with a JSON payload containing {"task_type": "math", "num_repeat_per_sample": 4, ...}
```

### mbridge Model Bridges (slime_plugins/mbridge)

Custom mbridge implementations mapping Megatron weights to HF for specific models.

- GLM4Bridge: registers as "glm4".

### Model specs (slime_plugins/models)

- get_glm_spec(args): returns transformer layer spec for Megatron.

