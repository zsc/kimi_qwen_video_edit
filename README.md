# Qwen Video Edit with TTLG

Video editing using Qwen-Image-Edit-2509 with Test-Time Loss Guidance (TTLG).

## Features

- **Frame-by-frame generation**: Uses first frame as style anchor + edge sequence as structure control
- **Test-Time Loss Guidance (TTLG)**: Gradient-based guidance during inference without training
  - Edge loss: Match generated frames to target edge maps
  - Gram style loss: Maintain style consistency with init frame (via VGG features)
  - Temporal loss (optional): Encourage consistency with previous frame
- **Streaming inference**: Memory-efficient frame-by-frame processing
- **Metrics tracking**: Per-frame edge/gram/LPIPS metrics for quality assessment

## Installation

```bash
pip install -r requirements.txt
```

### Model Location

The model is expected at:
```
/.autodl-model/data/Qwen/Qwen-Image-Edit-2509
```

Or specify with `--model_id` flag.

## Usage

### Basic Usage

```bash
python scripts/qwen_video_edit.py \
  --init_frame ./data/init.png \
  --edge_frames_dir ./data/edge_frames \
  --out_frames_dir ./outputs \
  --seed 0
```

### With TTLG (Recommended)

```bash
python scripts/qwen_video_edit.py \
  --init_frame ./data/init.png \
  --edge_frames_dir ./data/edge_frames \
  --out_frames_dir ./outputs \
  --ttlg_edge_scale 1.0 \
  --ttlg_gram_scale 0.2 \
  --ttlg_lr 0.05 \
  --ttlg_last_steps 10 \
  --seed 0 \
  --metrics_out ./outputs/metrics.csv
```

### Self-Test

Verify TTLG components without loading the full model:

```bash
python scripts/qwen_video_edit.py --self_test
```

Or test the TTLG module directly:

```bash
python scripts/qwen_video_edit_ttlg_guidance.py
```

## CLI Arguments

### Input/Output
- `--init_frame`: Path to init frame (required)
- `--edge_frames_dir`: Directory containing edge frames (required)
- `--out_frames_dir`: Output directory for generated frames (required)
- `--metrics_out`: Path to write metrics CSV (optional)

### Model Settings
- `--model_id`: Model path or HF repo (default: `/.autodl-model/data/Qwen/Qwen-Image-Edit-2509`)
- `--prompt`: Generation prompt
- `--negative_prompt`: Negative prompt
- `--num_inference_steps`: Denoising steps (default: 50)
- `--true_cfg_scale`: True CFG scale (default: 4.0)
- `--seed`: Random seed (default: 0)
- `--device`: Device (default: cuda)
- `--dtype`: Data type - fp16/bf16/fp32 (default: bf16)
- `--max_frames`: Maximum frames to generate (optional)

### TTLG Settings
- `--ttlg_edge_scale`: Edge loss scale (0 = disabled)
- `--ttlg_gram_scale`: Gram style loss scale (0 = disabled)
- `--ttlg_temporal_scale`: Temporal loss scale (0 = disabled)
- `--ttlg_lr`: Latent gradient step size (default: 0.05)
- `--ttlg_every`: Apply TTLG every N steps (default: 1)
- `--ttlg_start_step` / `--ttlg_end_step`: Step range for TTLG
- `--ttlg_last_steps`: Only apply TTLG in last N steps (recommended: 10-20)
- `--ttlg_edge_size`: Resize for edge loss computation (optional, e.g., 256)
- `--ttlg_gram_size`: Resize for Gram loss computation (optional, e.g., 224)

## TTLG Tips

1. **Start conservative**: Use `--ttlg_last_steps 10` to only apply guidance in final steps
2. **Edge guidance**: Start with `--ttlg_edge_scale 1.0` and adjust based on results
3. **Style guidance**: Use `--ttlg_gram_scale 0.1~0.5` for subtle style consistency
4. **Learning rate**: Default `0.05` works well; increase for stronger guidance, decrease for stability

## Metrics

The `--metrics_out` CSV contains per-frame metrics:
- `edge_loss`: L1 loss between generated edge and target edge
- `gram_loss`: L1 Gram matrix loss vs init frame
- `lpips_to_init`: Perceptual distance to init frame
- `lpips_to_prev`: Perceptual distance to previous frame

## Project Structure

```
scripts/
├── qwen_video_edit.py              # Main CLI entry
└── qwen_video_edit_ttlg_guidance.py # TTLG utilities & metrics
```

## Algorithm

1. Load init frame and edge sequence
2. Copy init frame to output (frame_0000.png)
3. For each edge frame:
   - Prepare Qwen multi-image input: `[edge_img, prev_rgb]`
   - Run denoising with optional TTLG callback
   - Save output and compute metrics
   - Update `prev_rgb` for next iteration

## Citation

Based on Qwen-Image-Edit-2509 from Alibaba Cloud.
