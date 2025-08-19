## Utilities

### slime.utils.types

- dataclass Sample
  - Fields: index, prompt, tokens, response, response_length, label, reward, loss_mask, status, metadata
  - Methods:
    - to_dict()/from_dict()
    - get_reward_value(args) -> float

```1:44:slime/utils/types.py
@dataclass
class Sample:
    """The sample generated"""
    ...
    def get_reward_value(self, args) -> float:
        return self.reward if not args.reward_key else self.reward[args.reward_key]
```

### slime.utils.data

- class Dataset(path, tokenizer, max_length, prompt_key="text", label_key=None, tool_key=None, metadata_key="metadata", seed=42, apply_chat_template=False)
  - Loads json/jsonl/parquet into list[Sample], supports chat template and length filtering.
  - shuffle(epoch) to reshuffle deterministically.

Example:
```python
from transformers import AutoTokenizer
from slime.utils.data import Dataset

tok = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
ds = Dataset(args.prompt_data, tokenizer=tok, max_length=args.rollout_max_prompt_len,
             prompt_key=args.input_key, label_key=args.label_key, metadata_key=args.metadata_key,
             tool_key=args.tool_key, apply_chat_template=args.apply_chat_template)
```

### slime.utils.mask_utils

- class MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen")
  - get_loss_mask(messages) -> (token_ids, loss_mask)
  - get_response_lengths(loss_masks) -> list[int]
  - get_text_from_loss_mask(token_ids, loss_mask) -> list[str]

Usage:
```python
from transformers import AutoTokenizer
from slime.utils.mask_utils import MultiTurnLossMaskGenerator

tok = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
gen = MultiTurnLossMaskGenerator(tok, tokenizer_type=args.loss_mask_type)
token_ids, loss_mask = gen.get_loss_mask(messages)
```

### slime.utils.async_utils

- run(coro): Run coroutine on a dedicated background event loop thread.

### slime.utils.timer

- class Timer (singleton)
- timer decorator/context manager.

Example:
```python
from slime.utils.timer import timer

@timer
def train_step():
    ...

with timer("io"):
    ...
```

### slime.utils.http_utils

- find_available_port(base_port) -> int
- is_port_available(port) -> bool
- get_host_info() -> (hostname, ip)
- post(url, payload, use_http2=False, max_retries=60) -> dict|str (async)
- get(url, use_http2=False) -> dict (async)

### slime.utils.wandb_utils

- init_wandb_primary(args) -> run_id|None
- init_wandb_secondary(args, run_id)
- is_wandb_offline() / get_wandb_offline_dir(args=None)

### slime.utils.arguments

Entrypoint for all CLI flags used by slime. Use programmatically via:

```python
from slime.utils.arguments import parse_args
args = parse_args()
```

Key groups include cluster, rollout generation, data, eval, algorithm (PPO/GRPO/REINFORCE++), W&B, network, reward model, rollout buffer, and custom Megatron hooks. Refer to the source for exhaustive argument details.

### slime.utils.ppo_utils

Policy gradient helpers used in PPO/GRPO/REINFORCE++:
- compute_approx_kl(log_probs, log_probs_base, kl_loss_type)
- compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c=None)
- compute_log_probs(logits, tokens, process_group)
- compute_entropy_from_logits(logits, process_group)
- get_grpo_returns(rewards, kl)
- get_reinforce_plus_plus_returns(rewards, kl, loss_masks, response_lengths, total_lengths, kl_coef, gamma)
- get_reinforce_plus_plus_baseline_advantages(rewards, kl, loss_masks, kl_coef)

Example:
```python
from slime.utils.ppo_utils import compute_approx_kl, compute_policy_loss
ppo_kl = compute_approx_kl(log_probs, log_probs_base, kl_loss_type="kl")
losses, clipfrac = compute_policy_loss(ppo_kl, advantages, eps_clip=0.2, eps_clip_high=0.2)
```

