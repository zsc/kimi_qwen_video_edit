#!/usr/bin/env python3
"""
Qwen Video Edit - Video editing using Qwen-Image-Edit-2509 with Test-Time Loss Guidance (TTLG).

This script performs frame-by-frame video generation using:
- First frame as style/identity anchor
- Edge sequence as structure control
- Optional TTLG (Edge loss + Gram style loss + Temporal loss)

Usage:
    python scripts/qwen_video_edit.py \
        --init_frame ./data/init.png \
        --edge_frames_dir ./data/edge_frames \
        --out_frames_dir ./outputs \
        --ttlg_edge_scale 1.0 \
        --ttlg_gram_scale 0.2
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image

# Import TTLG utilities
from qwen_video_edit_ttlg_guidance import (
    LossGuidanceConfig,
    MetricsComputer,
    VGGStyleExtractor,
    make_ttlg_callback,
    pil_to_tensor,
    tensor_to_pil,
    resize_edge_to_match,
    self_test as ttlg_self_test,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Qwen Video Edit - Video editing with TTLG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output - Frame directory mode
    parser.add_argument("--init_frame", type=str, required=True,
                        help="Path to init frame (frame_rgb[0])")
    parser.add_argument("--edge_frames_dir", type=str, required=True,
                        help="Directory containing edge frames")
    parser.add_argument("--out_frames_dir", type=str, required=True,
                        help="Output directory for generated frames")
    
    # Video file mode (optional)
    parser.add_argument("--rgb_video", type=str, default=None,
                        help="[Optional] Input RGB video (only first frame used as init)")
    parser.add_argument("--edge_video", type=str, default=None,
                        help="[Optional] Input edge video")
    parser.add_argument("--out_video", type=str, default=None,
                        help="[Optional] Output video path")
    
    # Model settings
    parser.add_argument("--model_id", type=str,
                        default="/.autodl-model/data/Qwen/Qwen-Image-Edit-2509",
                        help="Model identifier or local path")
    parser.add_argument("--prompt", type=str,
                        default="将图 2 按图 1 所勾勒出的精致形状进行变形，生成一张图像输出",
                        help="Prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default=" ",
                        help="Negative prompt")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps")
    parser.add_argument("--true_cfg_scale", type=float, default=4.0,
                        help="True CFG scale for Qwen pipeline")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="Guidance scale")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["fp16", "bf16", "fp32"],
                        help="Data type for model")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum number of frames to generate (including init)")
    
    # TTLG settings
    parser.add_argument("--ttlg_edge_scale", type=float, default=0.0,
                        help="Edge loss guidance scale (0 = disabled)")
    parser.add_argument("--ttlg_gram_scale", type=float, default=0.0,
                        help="Gram style loss guidance scale (0 = disabled)")
    parser.add_argument("--ttlg_temporal_scale", type=float, default=0.0,
                        help="Temporal consistency loss scale (0 = disabled)")
    parser.add_argument("--ttlg_lr", type=float, default=0.05,
                        help="Latent gradient step size")
    parser.add_argument("--ttlg_every", type=int, default=1,
                        help="Apply TTLG every N steps")
    parser.add_argument("--ttlg_start_step", type=int, default=None,
                        help="Start TTLG from this step")
    parser.add_argument("--ttlg_end_step", type=int, default=None,
                        help="End TTLG at this step")
    parser.add_argument("--ttlg_last_steps", type=int, default=None,
                        help="Only apply TTLG in last N steps (overrides start/end)")
    parser.add_argument("--ttlg_edge_size", type=int, default=None,
                        help="Resize to this size for edge loss computation")
    parser.add_argument("--ttlg_gram_size", type=int, default=None,
                        help="Resize to this size for Gram loss computation")
    
    # Output settings
    parser.add_argument("--metrics_out", type=str, default=None,
                        help="Path to write metrics CSV")
    parser.add_argument("--save_debug_dir", type=str, default=None,
                        help="Directory to save debug visualizations")
    parser.add_argument("--edge_resize_mode", type=str, default="nearest",
                        choices=["nearest", "bilinear"],
                        help="Interpolation mode for resizing edges")
    
    # Other
    parser.add_argument("--self_test", action="store_true",
                        help="Run self-test and exit")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (currently only 1 is supported)")
    
    return parser.parse_args()


def load_init_frame(path: str) -> Image.Image:
    """Load and validate init frame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Init frame not found: {path}")
    
    img = Image.open(path).convert("RGB")
    print(f"Loaded init frame: {img.size} from {path}")
    return img


def load_edge_frames(
    edge_dir: str,
    max_frames: Optional[int] = None
) -> List[Image.Image]:
    """
    Load edge frames from directory.
    
    Returns:
        List of PIL Images sorted by filename
    """
    if not os.path.isdir(edge_dir):
        raise NotADirectoryError(f"Edge frames directory not found: {edge_dir}")
    
    # Get all image files
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    files = [f for f in os.listdir(edge_dir) if f.lower().endswith(valid_exts)]
    files.sort()
    
    if len(files) == 0:
        raise ValueError(f"No edge frames found in {edge_dir}")
    
    # Load frames
    frames = []
    for fname in files:
        fpath = os.path.join(edge_dir, fname)
        img = Image.open(fpath)
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")
        frames.append(img)
    
    # Limit frames if specified
    if max_frames is not None:
        # max_frames includes init, so we can have max_frames-1 edge frames
        edge_limit = max_frames - 1
        if len(frames) > edge_limit:
            frames = frames[:edge_limit]
            print(f"Limited to {len(frames)} edge frames (max_frames={max_frames})")
    
    print(f"Loaded {len(frames)} edge frames from {edge_dir}")
    return frames


def create_output_dir(out_dir: str) -> None:
    """Create output directory if it doesn't exist."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")


def get_dtype(dtype_str: str):
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    return dtype_map[dtype_str]


def save_frame(img: Image.Image, path: str) -> None:
    """Save frame to disk."""
    img.save(path)


def save_metrics_csv(metrics_list: List[dict], path: str) -> None:
    """Save metrics to CSV file."""
    if not metrics_list:
        return
    
    keys = ["frame_idx", "edge_loss", "gram_loss", "lpips_to_init", "lpips_to_prev"]
    
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for m in metrics_list:
            writer.writerow({k: m.get(k, 0.0) for k in keys})
    
    print(f"Metrics saved to: {path}")


def validate_args(args: argparse.Namespace) -> None:
    """Validate arguments and check dependencies."""
    # Check inference mode
    if torch.is_inference_mode_enabled():
        warnings.warn(
            "torch.inference_mode() is enabled! This will prevent TTLG from working. "
            "The script will attempt to disable it during generation."
        )
    
    # Check torchvision if gram guidance is enabled
    if args.ttlg_gram_scale > 0:
        try:
            import torchvision
        except ImportError:
            raise ImportError(
                "torchvision is required for Gram style guidance (--ttlg_gram_scale > 0). "
                "Install with: pip install torchvision"
            )
    
    # Check diffusers
    try:
        from diffusers import QwenImageEditPlusPipeline
    except ImportError:
        raise ImportError(
            "diffusers with QwenImageEditPlusPipeline is required. "
            "Install with: pip install diffusers"
        )
    
    # Validate TTLG parameters
    if args.ttlg_last_steps is not None and args.ttlg_last_steps <= 0:
        raise ValueError("--ttlg_last_steps must be positive")
    
    if args.ttlg_edge_scale < 0 or args.ttlg_gram_scale < 0 or args.ttlg_temporal_scale < 0:
        raise ValueError("TTLG scales must be non-negative")


def generate_frame(
    pipe,
    edge_img: Image.Image,
    prev_rgb_img: Image.Image,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    true_cfg_scale: float,
    guidance_scale: float,
    generator: torch.Generator,
    config: LossGuidanceConfig,
    metrics_computer: Optional[MetricsComputer],
    edge_target_tensor: Optional[torch.Tensor],
    gram_ref: Optional[dict],
    prev_rgb_tensor: Optional[torch.Tensor],
    device: str,
) -> Tuple[Image.Image, Optional[dict]]:
    """
    Generate a single frame using Qwen pipeline with optional TTLG.
    
    Returns:
        (generated_image, metrics_dict or None)
    """
    from diffusers import QwenImageEditPlusPipeline
    
    # Prepare inputs
    inputs = {
        "image": [edge_img, prev_rgb_img],
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "true_cfg_scale": true_cfg_scale,
        "guidance_scale": guidance_scale,
        "generator": generator,
        "num_images_per_prompt": 1,
    }
    
    # Add callback if TTLG is enabled
    if config.is_enabled():
        callback = make_ttlg_callback(
            pipe=pipe,
            config=config,
            edge_target=edge_target_tensor,
            gram_ref=gram_ref,
            prev_rgb=prev_rgb_tensor,
            num_inference_steps=num_inference_steps,
        )
        inputs["callback_on_step_end"] = callback
    
    # Run generation (without inference_mode to allow gradients)
    # For TTLG, we need to avoid torch.inference_mode() context
    if config.is_enabled():
        # Just run without any special context manager
        output = pipe(**inputs)
        output_image = output.images[0]
    else:
        with torch.no_grad():
            output = pipe(**inputs)
            output_image = output.images[0]
    
    # Compute metrics if requested
    metrics = None
    if metrics_computer is not None:
        pred_tensor = pil_to_tensor(output_image).to(device)
        
        with torch.no_grad():
            metrics = metrics_computer.compute_all_metrics(
                pred=pred_tensor,
                target_edge=edge_target_tensor,
                ref_grams=gram_ref,
                init_frame=metrics_computer._init_grams is not None,
                prev_frame=prev_rgb_tensor,
            )
    
    return output_image, metrics


def main():
    """Main entry point."""
    args = parse_args()
    
    # Run self-test if requested
    if args.self_test:
        print("=" * 60)
        print("Running self-test mode")
        print("=" * 60)
        success = ttlg_self_test()
        sys.exit(0 if success else 1)
    
    # Validate arguments
    validate_args(args)
    
    print("=" * 60)
    print("Qwen Video Edit with TTLG")
    print("=" * 60)
    
    # Setup
    device = args.device
    dtype = get_dtype(args.dtype)
    
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Dtype: {args.dtype}")
    print(f"  Model: {args.model_id}")
    print(f"  Steps: {args.num_inference_steps}")
    print(f"  Seed: {args.seed}")
    
    # Load init frame
    print("\n[1/5] Loading init frame...")
    init_frame = load_init_frame(args.init_frame)
    init_size = init_frame.size  # (W, H)
    
    # Load edge frames
    print("\n[2/5] Loading edge frames...")
    edge_frames = load_edge_frames(args.edge_frames_dir, args.max_frames)
    
    if len(edge_frames) == 0:
        raise ValueError("No edge frames loaded!")
    
    total_output_frames = len(edge_frames) + 1  # +1 for init frame
    print(f"  Total output frames: {total_output_frames}")
    
    # Create output directory
    create_output_dir(args.out_frames_dir)
    
    # Save init frame as frame_0000.png
    init_save_path = os.path.join(args.out_frames_dir, "frame_0000.png")
    save_frame(init_frame, init_save_path)
    print(f"  Saved init frame to: {init_save_path}")
    
    # Load model
    print("\n[3/5] Loading Qwen pipeline...")
    from diffusers import QwenImageEditPlusPipeline
    
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    
    # Freeze all model parameters for TTLG
    # Note: QwenImageEditPlusPipeline doesn't have unet; it uses transformer
    if hasattr(pipe, "transformer"):
        for param in pipe.transformer.parameters():
            param.requires_grad = False
    if hasattr(pipe, "vae"):
        for param in pipe.vae.parameters():
            param.requires_grad = False
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        for param in pipe.text_encoder.parameters():
            param.requires_grad = False
    
    print("  Pipeline loaded and frozen for TTLG")
    
    # Setup TTLG config
    config = LossGuidanceConfig(
        ttlg_edge_scale=args.ttlg_edge_scale,
        ttlg_gram_scale=args.ttlg_gram_scale,
        ttlg_temporal_scale=args.ttlg_temporal_scale,
        ttlg_lr=args.ttlg_lr,
        ttlg_every=args.ttlg_every,
        ttlg_start_step=args.ttlg_start_step,
        ttlg_end_step=args.ttlg_end_step,
        ttlg_last_steps=args.ttlg_last_steps,
        ttlg_edge_size=args.ttlg_edge_size,
        ttlg_gram_size=args.ttlg_gram_size,
    )
    
    print("\n[TTLG Configuration]")
    print(f"  Enabled: {config.is_enabled()}")
    if config.is_enabled():
        print(f"  Edge scale: {config.ttlg_edge_scale}")
        print(f"  Gram scale: {config.ttlg_gram_scale}")
        print(f"  Temporal scale: {config.ttlg_temporal_scale}")
        print(f"  Learning rate: {config.ttlg_lr}")
        print(f"  Apply every: {config.ttlg_every} steps")
        if config.ttlg_last_steps:
            print(f"  Last steps only: {config.ttlg_last_steps}")
        else:
            print(f"  Step range: {start_step if (start_step := config.ttlg_start_step) is not None else 0} - {end_step if (end_step := config.ttlg_end_step) is not None else args.num_inference_steps}")
    
    # Setup metrics
    metrics_computer = None
    if args.metrics_out:
        print("\n[Metrics] Enabled")
        metrics_computer = MetricsComputer(device=device)
        init_tensor = pil_to_tensor(init_frame).to(device)
        metrics_computer.set_init_frame(init_tensor)
    
    # Pre-compute Gram matrices for init frame if Gram guidance is enabled
    gram_ref = None
    if config.ttlg_gram_scale > 0:
        print("\n[Gram Guidance] Pre-computing Gram matrices for init frame...")
        vgg = VGGStyleExtractor(device=device)
        init_tensor = pil_to_tensor(init_frame).to(device)
        gram_ref = vgg.compute_gram_matrices(init_tensor)
        print("  Done")
    
    # Setup generator
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    # Initialize prev_rgb with init_frame
    prev_rgb = init_frame
    
    # Metrics list
    all_metrics = []
    
    # Add init frame metrics (all zeros as reference)
    all_metrics.append({
        "frame_idx": 0,
        "edge_loss": 0.0,
        "gram_loss": 0.0,
        "lpips_to_init": 0.0,
        "lpips_to_prev": 0.0,
    })
    
    # Generate frames
    print("\n[4/5] Generating frames...")
    print("-" * 60)
    
    for frame_idx, edge_img in enumerate(edge_frames, start=1):
        print(f"\nFrame {frame_idx}/{total_output_frames-1}:")
        
        # Resize edge to match init frame if needed
        if edge_img.size != init_size:
            edge_img = resize_edge_to_match(
                edge_img,
                (init_size[1], init_size[0]),  # (H, W)
                mode=args.edge_resize_mode
            )
            print(f"  Resized edge to {init_size}")
        
        # Prepare tensors for TTLG
        edge_target_tensor = None
        prev_rgb_tensor = None
        
        if config.ttlg_edge_scale > 0:
            edge_target_tensor = pil_to_tensor(edge_img).to(device)
        
        if config.ttlg_temporal_scale > 0 or metrics_computer is not None:
            prev_rgb_tensor = pil_to_tensor(prev_rgb).to(device)
        
        # Generate
        output_img, metrics = generate_frame(
            pipe=pipe,
            edge_img=edge_img,
            prev_rgb_img=prev_rgb,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            true_cfg_scale=args.true_cfg_scale,
            guidance_scale=args.guidance_scale,
            generator=generator,
            config=config,
            metrics_computer=metrics_computer,
            edge_target_tensor=edge_target_tensor,
            gram_ref=gram_ref,
            prev_rgb_tensor=prev_rgb_tensor,
            device=device,
        )
        
        # Save output
        out_path = os.path.join(args.out_frames_dir, f"frame_{frame_idx:04d}.png")
        save_frame(output_img, out_path)
        print(f"  Saved to: {out_path}")
        
        # Print metrics
        if metrics:
            print(f"  Metrics: edge={metrics['edge_loss']:.4f}, "
                  f"gram={metrics['gram_loss']:.4f}, "
                  f"lpips_init={metrics['lpips_to_init']:.4f}, "
                  f"lpips_prev={metrics['lpips_to_prev']:.4f}")
            metrics["frame_idx"] = frame_idx
            all_metrics.append(metrics)
        
        # Update prev_rgb for next frame
        prev_rgb = output_img
        
        # Clear cache periodically
        if frame_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print(f"[5/5] Generation complete!")
    print(f"  Output frames: {total_output_frames}")
    print(f"  Location: {args.out_frames_dir}")
    
    # Save metrics
    if args.metrics_out and all_metrics:
        save_metrics_csv(all_metrics, args.metrics_out)
    
    print("=" * 60)


if __name__ == "__main__":
    main()
