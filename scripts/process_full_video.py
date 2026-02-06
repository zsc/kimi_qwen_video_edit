#!/usr/bin/env python3
"""
Process full video with Qwen Video Edit + TTLG.
Resumes from where it left off.
"""

import os
import sys
import argparse
import csv
import torch
from PIL import Image
from tqdm import tqdm

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.qwen_video_edit_ttlg_guidance import (
    LossGuidanceConfig, MetricsComputer, VGGStyleExtractor,
    make_ttlg_callback, pil_to_tensor
)


def process_video(
    init_frame_path: str,
    edge_frames_dir: str,
    out_frames_dir: str,
    num_inference_steps: int = 10,
    ttlg_edge_scale: float = 1.0,
    ttlg_gram_scale: float = 0.3,
    ttlg_lr: float = 0.05,
    ttlg_last_steps: int = 5,
    seed: int = 42,
    dtype_str: str = "bf16",
    metrics_out: str = None,
):
    """Process video frames with resume capability."""
    
    from diffusers import QwenImageEditPlusPipeline
    
    # Setup
    device = "cuda"
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype_str]
    
    os.makedirs(out_frames_dir, exist_ok=True)
    
    # Load init frame
    init_frame = Image.open(init_frame_path).convert("RGB")
    init_size = init_frame.size
    print(f"Init frame: {init_size}")
    
    # Load edge frames
    edge_files = sorted([f for f in os.listdir(edge_frames_dir) if f.endswith('.png')])
    print(f"Edge frames: {len(edge_files)}")
    
    # Check already processed frames
    existing_files = [f for f in os.listdir(out_frames_dir) if f.startswith('frame_') and f.endswith('.png')]
    start_frame = len(existing_files)
    
    if start_frame == 0:
        # Save init frame
        init_save_path = os.path.join(out_frames_dir, "frame_0000.png")
        init_frame.save(init_save_path)
        print(f"Saved init frame to: {init_save_path}")
        start_frame = 1
    else:
        print(f"Resuming from frame {start_frame}")
    
    # Load model
    print("Loading Qwen pipeline...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "/.autodl-model/data/Qwen/Qwen-Image-Edit-2509",
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)
    
    # Freeze parameters
    if hasattr(pipe, "transformer"):
        for param in pipe.transformer.parameters():
            param.requires_grad = False
    if hasattr(pipe, "vae"):
        for param in pipe.vae.parameters():
            param.requires_grad = False
    
    # Setup TTLG
    config = LossGuidanceConfig(
        ttlg_edge_scale=ttlg_edge_scale,
        ttlg_gram_scale=ttlg_gram_scale,
        ttlg_lr=ttlg_lr,
        ttlg_last_steps=ttlg_last_steps,
    )
    
    # Setup metrics
    metrics_computer = None
    if metrics_out:
        metrics_computer = MetricsComputer(device=device)
        init_tensor = pil_to_tensor(init_frame).to(device)
        metrics_computer.set_init_frame(init_tensor)
    
    # Pre-compute Gram matrices
    gram_ref = None
    if config.ttlg_gram_scale > 0:
        print("Pre-computing Gram matrices...")
        vgg = VGGStyleExtractor(device=device)
        init_tensor = pil_to_tensor(init_frame).to(device)
        gram_ref = vgg.compute_gram_matrices(init_tensor)
    
    # Generator
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Load prev_rgb for resume
    if start_frame > 1:
        prev_rgb = Image.open(os.path.join(out_frames_dir, f"frame_{start_frame-1:04d}.png"))
    else:
        prev_rgb = init_frame
    
    # Metrics list
    all_metrics = []
    
    # Process frames
    total_frames = len(edge_files) + 1
    prompt = "将图 2 按图 1 所勾勒出的精致形状进行变形，生成一张图像输出"
    negative_prompt = " "
    
    for frame_idx in tqdm(range(start_frame, total_frames), desc="Generating frames"):
        edge_idx = frame_idx - 1
        
        # Load edge
        edge_path = os.path.join(edge_frames_dir, edge_files[edge_idx])
        edge_img = Image.open(edge_path)
        if edge_img.size != init_size:
            edge_img = edge_img.resize(init_size, Image.NEAREST)
        
        # Prepare inputs
        inputs = {
            "image": [edge_img, prev_rgb],
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "true_cfg_scale": 4.0,
            "guidance_scale": 1.0,
            "generator": generator,
            "num_images_per_prompt": 1,
        }
        
        # Add callback if TTLG enabled
        if config.is_enabled():
            edge_target = pil_to_tensor(edge_img).to(device)
            prev_rgb_tensor = pil_to_tensor(prev_rgb).to(device) if config.ttlg_temporal_scale > 0 else None
            
            callback = make_ttlg_callback(
                pipe=pipe,
                config=config,
                edge_target=edge_target,
                gram_ref=gram_ref,
                prev_rgb=prev_rgb_tensor,
                num_inference_steps=num_inference_steps,
            )
            inputs["callback_on_step_end"] = callback
        
        # Generate
        if config.is_enabled():
            output = pipe(**inputs)
        else:
            with torch.no_grad():
                output = pipe(**inputs)
        
        output_img = output.images[0]
        
        # Compute metrics
        if metrics_computer:
            pred_tensor = pil_to_tensor(output_img).to(device)
            with torch.no_grad():
                metrics = metrics_computer.compute_all_metrics(
                    pred=pred_tensor,
                    target_edge=pil_to_tensor(edge_img).to(device),
                    ref_grams=gram_ref,
                    init_frame=pil_to_tensor(init_frame).to(device),
                    prev_frame=pil_to_tensor(prev_rgb).to(device),
                )
                metrics["frame_idx"] = frame_idx
                all_metrics.append(metrics)
                tqdm.write(f"  edge={metrics['edge_loss']:.4f}, gram={metrics['gram_loss']:.4f}")
        
        # Save
        out_path = os.path.join(out_frames_dir, f"frame_{frame_idx:04d}.png")
        output_img.save(out_path)
        
        # Update prev_rgb
        prev_rgb = output_img
        
        # Clear cache periodically
        if frame_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    # Save metrics
    if metrics_out and all_metrics:
        with open(metrics_out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["frame_idx", "edge_loss", "gram_loss", "lpips_to_init", "lpips_to_prev"])
            writer.writeheader()
            for m in all_metrics:
                writer.writerow(m)
        print(f"\nMetrics saved to: {metrics_out}")
    
    print(f"\n✅ Complete! Generated {total_frames} frames in {out_frames_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_frame", default="data_full/init/init.png")
    parser.add_argument("--edge_frames_dir", default="data_full/edges")
    parser.add_argument("--out_frames_dir", default="outputs_full")
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--ttlg_edge_scale", type=float, default=1.0)
    parser.add_argument("--ttlg_gram_scale", type=float, default=0.3)
    parser.add_argument("--ttlg_lr", type=float, default=0.05)
    parser.add_argument("--ttlg_last_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--metrics_out", default="outputs_full/metrics.csv")
    args = parser.parse_args()
    
    process_video(
        init_frame_path=args.init_frame,
        edge_frames_dir=args.edge_frames_dir,
        out_frames_dir=args.out_frames_dir,
        num_inference_steps=args.num_inference_steps,
        ttlg_edge_scale=args.ttlg_edge_scale,
        ttlg_gram_scale=args.ttlg_gram_scale,
        ttlg_lr=args.ttlg_lr,
        ttlg_last_steps=args.ttlg_last_steps,
        seed=args.seed,
        dtype_str=args.dtype,
        metrics_out=args.metrics_out,
    )
