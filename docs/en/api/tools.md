## Tools and scripts

Helper utilities for FP8 conversion and checkpoint format transforms.

### FP8 utilities

- tools/convert_hf_to_fp8.py
  - quant_fp8(weight, strategy, block_size=None)
  - convert_fp8(input_path, output_path, strategy, block_size=None, max_workers=4)

- tools/fp8_cast_bf16.py
  - weight_dequant(x, s, block_size=128)

### Checkpoint conversion

- tools/convert_hf_to_torch_dist.py
  - CLI to convert HF checkpoint to Megatron torch_dist, with support for distributed conversion.

- tools/convert_torch_dist_to_hf.py
  - CLI to convert Megatron torch_dist checkpoint back to HF.

- tools/convert_to_hf.py
  - Reuses training pipeline to export any Megatron checkpoint to HF using actor weight update paths.

Example: run HF -> torch_dist
```bash
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  --hf-checkpoint /path/to/hf \
  --save /path/to/torch_dist \
  ... # model args
```

