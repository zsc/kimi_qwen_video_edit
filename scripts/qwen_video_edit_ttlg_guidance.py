"""
TTLG (Test-Time Loss Guidance) utilities for Qwen Video Edit.

This module provides:
- Differentiable edge extraction (Sobel, Laplacian)
- Gram matrix style loss via VGG16
- TTLG callback factory for diffusers pipelines
- Metrics computation (edge, gram, LPIPS)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image

if TYPE_CHECKING:
    from diffusers import QwenImageEditPlusPipeline


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LossGuidanceConfig:
    """Configuration for TTLG (Test-Time Loss Guidance)."""
    
    # Scales (0.0 = disabled)
    ttlg_edge_scale: float = 0.0
    ttlg_gram_scale: float = 0.0
    ttlg_temporal_scale: float = 0.0
    
    # Optimization params
    ttlg_lr: float = 0.05
    ttlg_every: int = 1
    ttlg_start_step: Optional[int] = None
    ttlg_end_step: Optional[int] = None
    ttlg_last_steps: Optional[int] = None  # Priority over start/end
    
    # Input sizes for efficiency
    ttlg_edge_size: Optional[int] = None  # e.g., 256
    ttlg_gram_size: Optional[int] = None  # e.g., 224 or 256
    
    def is_enabled(self) -> bool:
        """Check if any TTLG guidance is enabled."""
        return (
            self.ttlg_edge_scale > 0
            or self.ttlg_gram_scale > 0
            or self.ttlg_temporal_scale > 0
        )


# ---------------------------------------------------------------------------
# Edge Extraction (Differentiable)
# ---------------------------------------------------------------------------

def sobel_edges(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Differentiable Sobel edge detection.
    
    Args:
        x: Input tensor [B, 3, H, W] in range [0, 1]
        eps: Small constant for numerical stability
        
    Returns:
        Edge magnitude [B, 1, H, W] in range [0, ~1.4]
    """
    # Convert to grayscale using standard weights
    gray = (
        0.2989 * x[:, 0:1] +
        0.5870 * x[:, 1:2] +
        0.1140 * x[:, 2:3]
    )
    
    # Sobel kernels
    sobel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    
    sobel_y = torch.tensor([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    
    # Apply convolution
    dx = F.conv2d(gray, sobel_x, padding=1)
    dy = F.conv2d(gray, sobel_y, padding=1)
    
    # Magnitude
    edge = torch.sqrt(dx ** 2 + dy ** 2 + eps)
    
    return edge


def laplacian_edges(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Differentiable Laplacian edge detection.
    
    Args:
        x: Input tensor [B, 3, H, W] in range [0, 1]
        eps: Small constant for numerical stability
        
    Returns:
        Edge magnitude [B, 1, H, W]
    """
    # Convert to grayscale
    gray = (
        0.2989 * x[:, 0:1] +
        0.5870 * x[:, 1:2] +
        0.1140 * x[:, 2:3]
    )
    
    # Laplacian kernel
    laplacian_kernel = torch.tensor([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    
    # Apply convolution
    edge = F.conv2d(gray, laplacian_kernel, padding=1)
    edge = torch.abs(edge)
    
    return edge


def compute_edge_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    edge_fn: Callable = sobel_edges,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute L1 edge loss between prediction and target.
    
    Args:
        pred: Predicted image [B, 3, H, W] in [0, 1]
        target: Target edge image [B, 1 or 3, H, W] in [0, 1]
        edge_fn: Edge extraction function
        reduction: "mean" or "none"
        
    Returns:
        Scalar loss or per-sample losses
    """
    # Extract edges from prediction
    pred_edge = edge_fn(pred)
    
    # Ensure target is compatible
    if target.dim() == 3:
        target = target.unsqueeze(1)
    if target.shape[1] == 3:
        # Convert RGB target to grayscale edge
        target = target[:, 0:1] * 0.2989 + target[:, 1:2] * 0.5870 + target[:, 2:3] * 0.1140
    
    # Resize target if needed
    if target.shape[-2:] != pred_edge.shape[-2:]:
        target = F.interpolate(
            target, size=pred_edge.shape[-2:], mode="bilinear", align_corners=False
        )
    
    loss = F.l1_loss(pred_edge, target, reduction=reduction)
    return loss


# ---------------------------------------------------------------------------
# Gram Matrix Style Loss
# ---------------------------------------------------------------------------

def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """
    Compute Gram matrix for style loss.
    
    Args:
        feat: Feature tensor [B, C, H, W]
        
    Returns:
        Gram matrix [B, C, C]
    """
    b, c, h, w = feat.shape
    feat_flat = feat.view(b, c, h * w)
    gram = torch.bmm(feat_flat, feat_flat.transpose(1, 2))
    gram = gram / (c * h * w)
    return gram


class VGGStyleExtractor(torch.nn.Module):
    """
    VGG16-based style feature extractor.
    Extracts features from multiple layers for Gram matrix computation.
    """
    
    # Default layers for style transfer
    DEFAULT_LAYERS = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
    
    def __init__(
        self,
        layers: Optional[List[str]] = None,
        device: str = "cuda",
        weights: str = "DEFAULT"
    ):
        super().__init__()
        
        try:
            import torchvision.models as models
        except ImportError:
            raise ImportError(
                "torchvision is required for Gram style guidance. "
                "Install with: pip install torchvision"
            )
        
        self.layers = layers or self.DEFAULT_LAYERS
        self.device = device
        
        # Load pretrained VGG16
        vgg = models.vgg16(weights=weights).features
        vgg = vgg.to(device).eval()
        
        # Freeze all parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Build feature extractor
        self.layer_names = []
        self.feature_extractor = torch.nn.Sequential()
        
        layer_mapping = {
            "relu1_2": 4,   # After conv1_2
            "relu2_2": 9,   # After conv2_2
            "relu3_3": 16,  # After conv3_3
            "relu4_3": 23,  # After conv4_3
        }
        
        # Collect layers up to the last needed one
        max_idx = max(layer_mapping[l] for l in self.layers)
        
        for i in range(max_idx + 1):
            self.feature_extractor.add_module(str(i), vgg[i])
            
        # Store indices for extraction
        self.layer_indices = [layer_mapping[l] for l in self.layers]
        
        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize to ImageNet statistics."""
        # x is in [0, 1], convert to VGG input format
        x = (x - self.mean) / self.std
        return x
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extract features from specified layers.
        
        Args:
            x: Input image [B, 3, H, W] in [0, 1]
            
        Returns:
            Dictionary of features for each layer
        """
        x = self.normalize(x)
        
        features = {}
        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            if i in self.layer_indices:
                layer_name = self.layers[self.layer_indices.index(i)]
                features[layer_name] = x
                
        return features
    
    def compute_gram_loss(
        self,
        pred: torch.Tensor,
        ref_grams: dict[str, torch.Tensor],
        weights: Optional[dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Compute Gram matrix loss between prediction and reference.
        
        Args:
            pred: Predicted image [B, 3, H, W] in [0, 1]
            ref_grams: Pre-computed Gram matrices for reference image
            weights: Optional per-layer weights
            
        Returns:
            Scalar loss
        """
        pred_features = self.forward(pred)
        
        if weights is None:
            weights = {layer: 1.0 for layer in self.layers}
        
        total_loss = 0.0
        total_weight = 0.0
        
        for layer_name in self.layers:
            if layer_name not in pred_features or layer_name not in ref_grams:
                continue
                
            pred_gram = gram_matrix(pred_features[layer_name])
            ref_gram = ref_grams[layer_name]
            
            layer_loss = F.l1_loss(pred_gram, ref_gram)
            weight = weights.get(layer_name, 1.0)
            
            total_loss += weight * layer_loss
            total_weight += weight
        
        if total_weight > 0:
            total_loss = total_loss / total_weight
            
        return total_loss
    
    def compute_gram_matrices(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Pre-compute Gram matrices for a reference image."""
        features = self.forward(x)
        return {name: gram_matrix(feat) for name, feat in features.items()}


# ---------------------------------------------------------------------------
# TTLG Callback Factory
# ---------------------------------------------------------------------------

def make_ttlg_callback(
    pipe: "QwenImageEditPlusPipeline",
    config: LossGuidanceConfig,
    edge_target: Optional[torch.Tensor] = None,
    gram_ref: Optional[dict[str, torch.Tensor]] = None,
    prev_rgb: Optional[torch.Tensor] = None,
    num_inference_steps: int = 50,
) -> Callable:
    """
    Create a callback_on_step_end function for TTLG.
    
    Args:
        pipe: The Qwen pipeline instance
        config: TTLG configuration
        edge_target: Target edge map [1, 1, H, W] or [1, 3, H, W]
        gram_ref: Pre-computed Gram matrices for style reference
        prev_rgb: Previous frame RGB for temporal consistency [1, 3, H, W]
        num_inference_steps: Total number of inference steps
        
    Returns:
        Callback function for callback_on_step_end
    """
    
    # Determine TTLG step range
    start_step = config.ttlg_start_step or 0
    end_step = config.ttlg_end_step or num_inference_steps
    
    if config.ttlg_last_steps is not None:
        start_step = max(0, num_inference_steps - config.ttlg_last_steps)
        end_step = num_inference_steps
    
    # Prepare VGG if needed
    vgg_extractor = None
    if config.ttlg_gram_scale > 0 and gram_ref is not None:
        try:
            vgg_extractor = VGGStyleExtractor(device=pipe.device)
        except ImportError:
            warnings.warn("torchvision not available, disabling Gram guidance")
            config.ttlg_gram_scale = 0.0
    
    # Prepare edge target
    if edge_target is not None and config.ttlg_edge_scale > 0:
        if edge_target.dim() == 3:
            edge_target = edge_target.unsqueeze(0)
        edge_target = edge_target.to(pipe.device)
    
    # Prepare prev_rgb for temporal loss
    if prev_rgb is not None and config.ttlg_temporal_scale > 0:
        prev_rgb = prev_rgb.to(pipe.device)
    
    def callback_on_step_end(
        pipeline,
        i: int,
        t: torch.Tensor,
        callback_kwargs: dict
    ) -> dict:
        """
        Callback function called at each denoising step end.
        
        Args:
            pipeline: The pipeline instance
            i: Current step index (0-indexed)
            t: Current timestep
            callback_kwargs: Contains latents and possibly pred_original_sample
            
        Returns:
            Modified callback_kwargs with updated latents
        """
        # Check if TTLG should run this step
        if not config.is_enabled():
            return callback_kwargs
        
        if i % config.ttlg_every != 0:
            return callback_kwargs
        
        if i < start_step or i >= end_step:
            return callback_kwargs
        
        # Get latents from callback
        latents = callback_kwargs.get("latents")
        if latents is None:
            return callback_kwargs
        
        original_dtype = latents.dtype
        
        # Enable gradients for latents
        with torch.enable_grad():
            # Create gradient-enabled copy
            latents_g = latents.detach().float().requires_grad_(True)
            
            # Determine what to decode (prioritize pred_x0 if available)
            latent_to_decode = callback_kwargs.get("pred_original_sample", latents_g)
            if latent_to_decode is None:
                latent_to_decode = latents_g
            
            # Decode latents to image (differentiable)
            # VAE scaling factor
            scaling_factor = getattr(pipe.vae.config, "scaling_factor", 0.18215)
            latents_vae = latent_to_decode / scaling_factor
            
            # Decode
            img_pred = pipe.vae.decode(latents_vae).sample
            # img_pred is in [-1, 1], convert to [0, 1]
            img_pred = (img_pred / 2 + 0.5).clamp(0, 1)
            
            # Resize for efficiency if specified
            if config.ttlg_edge_size is not None and config.ttlg_edge_scale > 0:
                img_for_edge = F.interpolate(
                    img_pred,
                    size=(config.ttlg_edge_size, config.ttlg_edge_size),
                    mode="bilinear",
                    align_corners=False
                )
            else:
                img_for_edge = img_pred
            
            if config.ttlg_gram_size is not None and config.ttlg_gram_scale > 0:
                img_for_gram = F.interpolate(
                    img_pred,
                    size=(config.ttlg_gram_size, config.ttlg_gram_size),
                    mode="bilinear",
                    align_corners=False
                )
            else:
                img_for_gram = img_pred
            
            # Compute losses
            total_loss = 0.0
            
            # Edge loss
            if config.ttlg_edge_scale > 0 and edge_target is not None:
                # Resize edge target to match
                target_resized = edge_target
                if target_resized.shape[-2:] != img_for_edge.shape[-2:]:
                    target_resized = F.interpolate(
                        target_resized,
                        size=img_for_edge.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                
                edge_loss = compute_edge_loss(
                    img_for_edge,
                    target_resized,
                    edge_fn=sobel_edges,
                    reduction="mean"
                )
                total_loss += config.ttlg_edge_scale * edge_loss
            
            # Gram loss
            if config.ttlg_gram_scale > 0 and gram_ref is not None and vgg_extractor is not None:
                gram_loss = vgg_extractor.compute_gram_loss(img_for_gram, gram_ref)
                total_loss += config.ttlg_gram_scale * gram_loss
            
            # Temporal loss
            if config.ttlg_temporal_scale > 0 and prev_rgb is not None:
                # Resize prev_rgb to match
                prev_resized = prev_rgb
                if prev_resized.shape[-2:] != img_pred.shape[-2:]:
                    prev_resized = F.interpolate(
                        prev_resized,
                        size=img_pred.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                temporal_loss = F.l1_loss(img_pred, prev_resized)
                total_loss += config.ttlg_temporal_scale * temporal_loss
            
            # Compute gradient and update
            if total_loss > 0:
                grad = torch.autograd.grad(total_loss, latents_g)[0]
                
                # Normalize gradient (per sample)
                grad_norm = grad.norm(dim=(1, 2, 3), keepdim=True) + 1e-8
                grad = grad / grad_norm
                
                # Update latents
                latents_updated = (latents_g - config.ttlg_lr * grad).to(original_dtype).detach()
                callback_kwargs["latents"] = latents_updated
        
        return callback_kwargs
    
    return callback_on_step_end


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------

class MetricsComputer:
    """Computes various metrics for video frame quality assessment."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._vgg_extractor = None
        self._lpips_model = None
        
        # Cache for reference features
        self._init_grams = None
        self._init_vgg_features = None
    
    def _get_vgg_extractor(self) -> Optional[VGGStyleExtractor]:
        """Lazy initialization of VGG extractor."""
        if self._vgg_extractor is None:
            try:
                self._vgg_extractor = VGGStyleExtractor(device=self.device)
            except ImportError:
                pass
        return self._vgg_extractor
    
    def _get_lpips_model(self):
        """Lazy initialization of LPIPS model."""
        if self._lpips_model is None:
            try:
                import lpips
                self._lpips_model = lpips.LPIPS(net="vgg").to(self.device)
                self._lpips_model.eval()
                for param in self._lpips_model.parameters():
                    param.requires_grad = False
            except ImportError:
                pass
        return self._lpips_model
    
    def set_init_frame(self, init_frame: torch.Tensor):
        """
        Set and cache features for the init frame.
        
        Args:
            init_frame: [1, 3, H, W] in [0, 1]
        """
        init_frame = init_frame.to(self.device)
        
        # Cache Gram matrices
        vgg = self._get_vgg_extractor()
        if vgg is not None:
            self._init_grams = vgg.compute_gram_matrices(init_frame)
            self._init_vgg_features = vgg.forward(init_frame)
    
    def compute_edge_loss(
        self,
        pred: torch.Tensor,
        target_edge: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute edge loss.
        
        Args:
            pred: [1, 3, H, W] in [0, 1]
            target_edge: [1, 1 or 3, H, W] in [0, 1]
            
        Returns:
            L1 edge loss value
        """
        pred = pred.to(self.device)
        
        with torch.no_grad():
            pred_edge = sobel_edges(pred)
            
            if target_edge is not None:
                target_edge = target_edge.to(self.device)
                if target_edge.dim() == 3:
                    target_edge = target_edge.unsqueeze(0)
                if target_edge.shape[1] == 3:
                    target_edge = (
                        target_edge[:, 0:1] * 0.2989 +
                        target_edge[:, 1:2] * 0.5870 +
                        target_edge[:, 2:3] * 0.1140
                    )
                
                # Resize if needed
                if target_edge.shape[-2:] != pred_edge.shape[-2:]:
                    target_edge = F.interpolate(
                        target_edge,
                        size=pred_edge.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                
                loss = F.l1_loss(pred_edge, target_edge).item()
            else:
                # No target, return mean edge magnitude
                loss = pred_edge.mean().item()
        
        return loss
    
    def compute_gram_loss(
        self,
        pred: torch.Tensor,
        ref_grams: Optional[dict] = None
    ) -> float:
        """
        Compute Gram matrix loss.
        
        Args:
            pred: [1, 3, H, W] in [0, 1]
            ref_grams: Optional reference Gram matrices (uses init frame if not provided)
            
        Returns:
            L1 Gram loss value
        """
        pred = pred.to(self.device)
        vgg = self._get_vgg_extractor()
        
        if vgg is None:
            return 0.0
        
        grams = ref_grams if ref_grams is not None else self._init_grams
        if grams is None:
            return 0.0
        
        with torch.no_grad():
            loss = vgg.compute_gram_loss(pred, grams).item()
        
        return loss
    
    def compute_lpips(
        self,
        pred: torch.Tensor,
        ref: torch.Tensor,
        size: int = 256
    ) -> float:
        """
        Compute LPIPS distance.
        
        Args:
            pred: [1, 3, H, W] in [0, 1]
            ref: [1, 3, H, W] in [0, 1]
            size: Resize to this size for computation
            
        Returns:
            LPIPS distance
        """
        lpips_model = self._get_lpips_model()
        if lpips_model is None:
            return 0.0
        
        pred = pred.to(self.device)
        ref = ref.to(self.device)
        
        # Resize to common size
        if pred.shape[-2:] != (size, size):
            pred = F.interpolate(
                pred, size=(size, size), mode="bilinear", align_corners=False
            )
        if ref.shape[-2:] != (size, size):
            ref = F.interpolate(
                ref, size=(size, size), mode="bilinear", align_corners=False
            )
        
        # LPIPS expects [-1, 1]
        pred_lpips = pred * 2 - 1
        ref_lpips = ref * 2 - 1
        
        with torch.no_grad():
            dist = lpips_model(pred_lpips, ref_lpips).item()
        
        return dist
    
    def compute_all_metrics(
        self,
        pred: torch.Tensor,
        target_edge: Optional[torch.Tensor] = None,
        ref_grams: Optional[dict] = None,
        init_frame: Optional[torch.Tensor] = None,
        prev_frame: Optional[torch.Tensor] = None
    ) -> dict[str, float]:
        """
        Compute all available metrics.
        
        Returns:
            Dictionary with keys: edge_loss, gram_loss, lpips_to_init, lpips_to_prev
        """
        metrics = {}
        
        # Edge loss
        metrics["edge_loss"] = self.compute_edge_loss(pred, target_edge)
        
        # Gram loss
        metrics["gram_loss"] = self.compute_gram_loss(pred, ref_grams)
        
        # LPIPS to init
        if init_frame is not None:
            metrics["lpips_to_init"] = self.compute_lpips(pred, init_frame)
        else:
            metrics["lpips_to_init"] = 0.0
        
        # LPIPS to prev
        if prev_frame is not None:
            metrics["lpips_to_prev"] = self.compute_lpips(pred, prev_frame)
        else:
            metrics["lpips_to_prev"] = 0.0
        
        return metrics


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL Image to torch tensor [1, 3, H, W] in [0, 1]."""
    import numpy as np
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert torch tensor [1, 3, H, W] in [0, 1] to PIL Image."""
    import numpy as np
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def resize_edge_to_match(
    edge: Union[Image.Image, torch.Tensor],
    target_size: Tuple[int, int],
    mode: str = "nearest"
) -> Union[Image.Image, torch.Tensor]:
    """
    Resize edge image to match target size.
    
    Args:
        edge: Edge image (PIL or tensor)
        target_size: (H, W) tuple
        mode: Interpolation mode for PIL (nearest, bilinear) or tensor
        
    Returns:
        Resized edge image
    """
    if isinstance(edge, Image.Image):
        return edge.resize((target_size[1], target_size[0]), Image.NEAREST if mode == "nearest" else Image.BILINEAR)
    else:
        # Tensor [B, C, H, W]
        align_corners = False if mode != "nearest" else None
        if mode == "nearest":
            return F.interpolate(edge, size=target_size, mode="nearest")
        else:
            return F.interpolate(edge, size=target_size, mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# Self-Test
# ---------------------------------------------------------------------------

def self_test():
    """Run self-tests for TTLG components."""
    print("Running TTLG self-test...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Test Sobel edges
    print("\n1. Testing Sobel edges (differentiable)...")
    x = torch.rand(2, 3, 256, 256, requires_grad=True)
    edge = sobel_edges(x)
    loss = edge.mean()
    loss.backward()
    assert x.grad is not None, "Sobel edges should allow gradient flow"
    print(f"   ✓ Sobel edges: output shape {edge.shape}, grad shape {x.grad.shape}")
    
    # Test Laplacian edges
    print("\n2. Testing Laplacian edges (differentiable)...")
    x = torch.rand(2, 3, 256, 256, requires_grad=True)
    edge = laplacian_edges(x)
    loss = edge.mean()
    loss.backward()
    assert x.grad is not None, "Laplacian edges should allow gradient flow"
    print(f"   ✓ Laplacian edges: output shape {edge.shape}")
    
    # Test Gram matrix
    print("\n3. Testing Gram matrix...")
    feat = torch.rand(2, 64, 32, 32, requires_grad=True)
    gram = gram_matrix(feat)
    loss = gram.mean()
    loss.backward()
    assert feat.grad is not None, "Gram matrix should allow gradient flow"
    print(f"   ✓ Gram matrix: output shape {gram.shape}")
    
    # Test VGG Style Extractor (if torchvision available)
    print("\n4. Testing VGG Style Extractor...")
    try:
        vgg = VGGStyleExtractor(device=device)
        x = torch.rand(1, 3, 256, 256).to(device)
        features = vgg(x)
        print(f"   ✓ VGG extractor: extracted {len(features)} layers")
        
        # Test Gram loss
        grams = vgg.compute_gram_matrices(x)
        x2 = torch.rand(1, 3, 256, 256).to(device)
        gram_loss = vgg.compute_gram_loss(x2, grams)
        print(f"   ✓ Gram loss: {gram_loss.item():.4f}")
    except ImportError as e:
        print(f"   ⚠ Skipped (torchvision not available: {e})")
    
    # Test MetricsComputer
    print("\n5. Testing MetricsComputer...")
    metrics = MetricsComputer(device=device)
    pred = torch.rand(1, 3, 256, 256)
    target_edge = torch.rand(1, 1, 256, 256)
    
    edge_loss = metrics.compute_edge_loss(pred, target_edge)
    print(f"   ✓ Edge loss: {edge_loss:.4f}")
    
    gram_loss = metrics.compute_gram_loss(pred)
    print(f"   ✓ Gram loss: {gram_loss:.4f}")
    
    # Test LPIPS if available
    lpips_dist = metrics.compute_lpips(pred, pred)
    if lpips_dist == 0.0 and metrics._lpips_model is None:
        print(f"   ⚠ LPIPS not available (install with: pip install lpips)")
    else:
        print(f"   ✓ LPIPS (same image): {lpips_dist:.4f}")
        pred2 = torch.rand(1, 3, 256, 256)
        lpips_dist2 = metrics.compute_lpips(pred, pred2)
        print(f"   ✓ LPIPS (different images): {lpips_dist2:.4f}")
    
    # Test config
    print("\n6. Testing LossGuidanceConfig...")
    config = LossGuidanceConfig(
        ttlg_edge_scale=1.0,
        ttlg_gram_scale=0.5,
        ttlg_lr=0.05
    )
    assert config.is_enabled(), "Config should be enabled with positive scales"
    print(f"   ✓ Config enabled: {config.is_enabled()}")
    
    config2 = LossGuidanceConfig()
    assert not config2.is_enabled(), "Config should be disabled with zero scales"
    print(f"   ✓ Config disabled when scales are 0")
    
    print("\n✅ All self-tests passed!")
    return True


if __name__ == "__main__":
    self_test()
