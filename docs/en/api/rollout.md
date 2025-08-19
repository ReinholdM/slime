## Rollout generation APIs

Rollout functions generate samples used for training. You can plug in your own generator via `--rollout-function-path`.

### slime.rollout.sglang_rollout

Exports: `generate_rollout(args, rollout_id, data_buffer, evaluation=False)`

High-level flow:
- Uses SGLang servers managed by RolloutManager
- Asynchronously generates responses and rewards, supports group RM and partial rollouts
- Returns `list[list[Sample]]` with length `args.rollout_batch_size`; each group has `args.n_samples_per_prompt`

Important helpers:
- GenerateState: caches tokenizer, concurrency semaphore, and sampling parameters
- generate(args, sample, sampling_params) -> Sample: single request via router
- generate_and_rm(args, sample, sampling_params, evaluation=False)
- generate_rollout_async(args, rollout_id, data_source) -> (completed, aborted)

Minimal custom generate function
```python
async def my_generate(args, sample, sampling_params):
    # modify sampling params per sample if needed
    return await generate(args, sample, sampling_params)
```

Use it via:
```bash
--custom-generate-function-path my_pkg.my_mod.my_generate
```

Evaluation helpers:
- eval_rollout(args, rollout_id) aggregates metrics per dataset

### slime.rollout.sft_rollout

Exports: `generate_rollout(args, rollout_id, data_buffer, evaluation=False)`

For SFT workflows, builds loss masks from chat messages.

```12:48:slime/rollout/sft_rollout.py
def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    ...
    samples = data_buffer.get_samples(args.rollout_batch_size)
    for sample in samples:
        (sample,) = sample
        messages = sample.prompt
        token_ids, loss_mask = MASK_GENERATOR.get_loss_mask(messages)
        response_length = MASK_GENERATOR.get_response_lengths([loss_mask])[0]
        sample.tokens = token_ids
        sample.response_length = response_length
        sample.reward = 0
        sample.loss_mask = loss_mask[-response_length:]
    return samples
```

### Rollout filters (dynamic/over-sampling)

- dynamic_sampling_filters.check_reward_nonzero_std(args, samples) -> bool
  - Example condition used to keep groups with non-zero reward std.

- over_sampling_filters.sort_by_reward_std(args, samples) -> list[list[Sample]]
  - Sorts groups by reward std (desc) and returns top-k.

Usage:
```bash
--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std \
--over-sampling-filter-path slime.rollout.filter_hub.over_sampling_filters.sort_by_reward_std
```

### Rollout data source

`slime.ray.rollout_data_source`

- class RolloutDataSource(args)
  - get_samples(num_samples) -> list[list[Sample]]
  - save(rollout_id), load(rollout_id): persist dataset position

- class RolloutDataSourceWithBuffer(RolloutDataSource)
  - add_samples(samples)
  - get_buffer_length()
  - buffer filter via `--buffer-filter-path` (default FIFO)

Programmatic example:
```python
from slime.ray.rollout_data_source import RolloutDataSourceWithBuffer

ds = RolloutDataSourceWithBuffer(args)
groups = ds.get_samples(args.rollout_batch_size)
ds.add_samples(groups[:4])
```

