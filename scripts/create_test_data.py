#!/usr/bin/env python3
"""Create test data for Qwen Video Edit experiment."""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_init_frame(path: str, size: tuple = (512, 512)):
    """Create a colorful init frame with gradient and shapes."""
    w, h = size
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Create gradient background
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            arr[y, x, 0] = int(255 * (x / w))  # Red gradient
            arr[y, x, 1] = int(255 * (y / h))  # Green gradient
            arr[y, x, 2] = 128  # Blue constant
    
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    
    # Draw some shapes
    draw.ellipse([w//4, h//4, 3*w//4, 3*h//4], outline='white', width=5)
    draw.rectangle([w//3, h//3, 2*w//3, 2*h//3], outline='yellow', width=3)
    
    img.save(path)
    print(f"Created init frame: {path} ({size})")
    return img


def create_edge_frame(path: str, init_img: Image.Image, frame_idx: int, deformation: float = 0.1):
    """Create an edge frame by deforming the init frame edges."""
    w, h = init_img.size
    
    # Convert to grayscale for edge extraction simulation
    gray = init_img.convert('L')
    arr = np.array(gray).astype(np.float32) / 255.0
    
    # Apply Sobel-like edge detection
    from scipy import ndimage
    dx = ndimage.sobel(arr, axis=1)
    dy = ndimage.sobel(arr, axis=0)
    edge = np.sqrt(dx**2 + dy**2)
    
    # Normalize to 0-255
    edge = (edge / edge.max() * 255).clip(0, 255).astype(np.uint8)
    
    # Add some deformation based on frame index
    from scipy.ndimage import affine_transform
    offset = deformation * frame_idx * 20
    matrix = np.array([[1, 0, offset], [0, 1, offset/2]])
    edge_deformed = affine_transform(edge, matrix[:2, :2], offset=matrix[:, 2])
    
    edge_img = Image.fromarray(edge_deformed.astype(np.uint8), mode='L').convert('RGB')
    edge_img.save(path)
    print(f"Created edge frame {frame_idx}: {path}")
    return edge_img


def create_simple_edge_frames(edge_dir: str, init_img: Image.Image, num_frames: int = 2):
    """Create simple synthetic edge frames without scipy dependency."""
    w, h = init_img.size
    
    for i in range(num_frames):
        # Create a simple edge pattern that changes per frame
        edge = Image.new('L', (w, h), color=0)
        draw = ImageDraw.Draw(edge)
        
        # Draw some lines/shapes that shift per frame
        offset = i * 30
        
        # Draw a circle that moves
        cx, cy = w//2 + offset, h//2
        radius = min(w, h) // 4
        draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius], outline=255, width=3)
        
        # Draw some lines
        draw.line([(0, h//2 + offset), (w, h//2 + offset)], fill=200, width=2)
        draw.line([(w//2 - offset, 0), (w//2 - offset, h)], fill=200, width=2)
        
        # Convert to RGB
        edge_rgb = edge.convert('RGB')
        
        path = os.path.join(edge_dir, f"edge_{i+1:04d}.png")
        edge_rgb.save(path)
        print(f"Created edge frame {i+1}: {path}")


def main():
    import sys
    
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/edge_frames", exist_ok=True)
    
    # Create init frame at 1/4 resolution (512x512 instead of 1024x1024)
    init_path = "data/init.png"
    init_img = create_init_frame(init_path, size=(512, 512))
    
    # Create 2 edge frames (for total of 3 output frames: init + 2 generated)
    create_simple_edge_frames("data/edge_frames", init_img, num_frames=2)
    
    print("\n✅ Test data created successfully!")
    print(f"  Init frame: {init_path}")
    print(f"  Edge frames: data/edge_frames/ (2 frames)")
    print(f"  Expected output: 3 frames total")


if __name__ == "__main__":
    main()
