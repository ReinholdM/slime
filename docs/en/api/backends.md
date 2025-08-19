## Backends

Integration with Megatron (training) and SGLang (serving).

### slime.backends.megatron_utils

Exports in __init__:
- parse_args, validate_args, set_default_megatron_args
- load_checkpoint, save_checkpoint
- init, initialize_model_and_optimizer
- MegatronTrainRayActor

Key components:

- initialize.init(args)
  - Initializes distributed groups, tokenizer, seeds, and microbatch calculator.

- model.initialize_model_and_optimizer(args)
  - Builds Megatron model, wraps in DDP/FSDP as configured, creates optimizer and lr scheduler, loads checkpoint.

- model.forward_only(args, model, data_iterator, num_microbatches, store_prefix="") -> dict
  - Runs forward passes to compute log probs and entropy per sample.

- model.train(rollout_id, model, optimizer, opt_param_scheduler, data_iterator, num_microbatches)
  - Full train loop over steps per rollout, overlapping grad reduce/param gather if configured; logs metrics.

- update_weight_utils.UpdateWeightFromTensor / UpdateWeightFromDistributed
  - Broadcast updated HF-compatible tensors to SGLang engines; supports TP/PP/EP.

- actor.MegatronTrainRayActor(TrainRayActor)
  - init(args, role, wandb_run_id, with_ref=False) -> start_rollout_id
  - train(rollout_id, rollout_data_ref)
  - update_weights()
  - save_model(iteration, with_optimizer=True)

Example: update weights each iteration
```python
ray.get(actor_model.async_update_weights())
```

### slime.backends.sglang_utils

- sglang_engine.SGLangEngine(RayActor)
  - init(dist_init_addr, port, nccl_port)
  - update_weights_from_tensor(serialized_named_tensors, load_format=None, flush_cache=False)
  - update_weights_from_distributed(names, dtypes, shapes, group_name, flush_cache=False)
  - flush_cache(), pause_generation(), continue_generation()
  - release_memory_occupation(), resume_memory_occupation(tags=None)

### Arguments mapping

- add_sglang_arguments(parser)
  - Adds all `ServerArgs` as CLI flags under `--sglang-` prefix, skipping runtime-provided values (port, tp_size, etc.).
- validate_args(args)
  - Derives sglang tp/dp/pp/ep sizes from slime arguments and enforces constraints.

