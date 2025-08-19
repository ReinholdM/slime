## Ray integration

Public classes and functions that orchestrate training and rollout engines using Ray.

### slime.ray.placement_group

- create_placement_groups(args) -> dict
  - Creates a Ray placement group with GPU bundles for actor (training) and rollout (inference) components. Returns a dict with keys `actor` and `rollout`, each as `(pg, reordered_bundle_indices)`.

- create_actor_group(args, pg, wandb_run_id) -> RayTrainGroup
  - Allocates training actors (Megatron) into the given placement group.

- create_rollout_manager(args, pg, wandb_run_id) -> RolloutManager
  - Creates and initializes rollout engines and controller.

Example:
```python
from slime.ray.placement_group import create_placement_groups, create_actor_group, create_rollout_manager

pgs = create_placement_groups(args)
actor_model = create_actor_group(args, pgs["actor"], wandb_run_id)
rollout_manager = create_rollout_manager(args, pgs["rollout"], wandb_run_id)
```

### slime.ray.actor_group

```13:136:slime/ray/actor_group.py
class RayTrainGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs

    Args:
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
    """
```

Key methods (all return Ray object refs when async):
- async_init(args, role, with_ref=False)
- async_init_weight_update_connections(rollout)
- async_train(rollout_id, rollout_data_ref)
- async_save_model(step_id)
- async_update_weights()
- async_offload()

Usage:
```python
start_rollout_ids = ray.get(actor_model.async_init(args, role="actor", with_ref=args.kl_coef != 0 or args.use_kl_loss))
ray.get(actor_model.async_update_weights())
ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
```

### slime.ray.rollout

- create_rollout_engines(args, pg) -> list[ray.ActorHandle]
  - Starts SGLangEngine actors with small fractional GPU allocations; wires ports and NCCL addresses.

- class RolloutManager
  - async_generate(rollout_id) -> ObjectRef
  - async_eval(rollout_id) -> ObjectRef
  - async_offload() -> list[ObjectRef]
  - async_onload(tags: list[str] | None) -> list[ObjectRef]

```145:174:slime/ray/rollout.py
class RolloutManager:
    def __init__(self, args, pg, wandb_run_id):
        ...
    def async_generate(self, rollout_id):
        return self.controller.generate.remote(rollout_id)
    def async_eval(self, rollout_id):
        return self.controller.eval.remote(rollout_id)
    def async_offload(self):
        return [engine.release_memory_occupation.remote() for engine in self.rollout_engines]
    def async_onload(self, tags: List[str] = None):
        return [engine.resume_memory_occupation.remote(tags=tags) for engine in self.rollout_engines]
```

### slime.ray.buffer

- @ray.remote class RolloutController
  - generate(rollout_id) -> Box(ObjectRef)
  - eval(rollout_id) -> None (logs eval metrics)
  - save(rollout_id) / load(rollout_id) for dataset state

Example to fetch rollout data within training loop:
```python
rollout_data_ref = ray.get(rollout_manager.async_generate(rollout_id))
ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
```

### slime.ray.ppo_actor

Abstract base actor defining training lifecycle on Ray workers.

Methods to implement:
- sleep(tags)
- wake_up(tags)
- connect_rollout_engines(rollout_engines, rollout_engine_lock)
- train(rollout_id, rollout_data_ref)
- eval(rollout_id, rollout_data_ref)
- save_model(iteration, with_optimizer=True)
- update_weights()

### slime.ray.utils

- NOSET_VISIBLE_DEVICES_ENV_VARS_LIST: env keys to prevent Ray from overriding device visibility
- ray_noset_visible_devices(env=os.environ) -> bool
- get_physical_gpu_id() -> str (GPU UUID)
- @ray.remote class Lock: acquire()/release() for engine broadcast synchronization

