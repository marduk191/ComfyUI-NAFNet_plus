"""
ComfyUI-NAFNet_plus: Comprehensive Image Restoration Nodes

Supported Models:
- NAFNet: Denoising (SIDD), Deblurring (GoPro/REDS), Stereo SR
- SCUNet: Blind real image denoising
- Restormer: Transformer-based denoising
- SwinIR: Swin Transformer denoising and super-resolution
- Real-ESRGAN: Practical image restoration and enhancement
- BSRGAN: Blind super-resolution with degradation handling
- HINet: Half Instance Normalization denoising
- MIRNet-v2: Multi-scale residual learning
- DnCNN: Classic CNN denoising
- FFDNet: Fast flexible denoising with noise level control
"""

import os
import torch
import numpy as np

try:
    import folder_paths
except ImportError:
    folder_paths = None

# Import architectures
try:
    from .nafnet_arch import NAFNet, NAFNetSR
    from .scunet_arch import SCUNet
    from .restormer_arch import Restormer
    from .swinir_arch import SwinIR
    from .rrdbnet_arch import RRDBNet
    from .hinet_arch import HINet
    from .mirnetv2_arch import MIRNet_v2
    from .dncnn_arch import DnCNN
    from .ffdnet_arch import FFDNet
except ImportError:
    from nafnet_arch import NAFNet, NAFNetSR
    from scunet_arch import SCUNet
    from restormer_arch import Restormer
    from swinir_arch import SwinIR
    from rrdbnet_arch import RRDBNet
    from hinet_arch import HINet
    from mirnetv2_arch import MIRNet_v2
    from dncnn_arch import DnCNN
    from ffdnet_arch import FFDNet

# Register model folder
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Auto-tile threshold (images larger than this will be tiled automatically)
AUTO_TILE_THRESHOLD = 1024 * 1024  # 1 megapixel
DEFAULT_TILE_SIZE = 512
DEFAULT_TILE_OVERLAP = 64


def get_device():
    """Get the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_state_dict_flexible(checkpoint):
    """Extract state dict from various checkpoint formats."""
    if isinstance(checkpoint, dict):
        for key in ['params', 'params_ema', 'state_dict', 'model', 'model_state_dict']:
            if key in checkpoint:
                return checkpoint[key]
        return checkpoint
    return checkpoint


def tiled_inference(model, img, tile_size, overlap, scale=1):
    """Process image in tiles for memory efficiency."""
    B, C, H, W = img.shape
    device = img.device

    tile_size = max(tile_size, 64)

    if H <= tile_size and W <= tile_size:
        return model(img)

    # Output dimensions
    out_H, out_W = H * scale, W * scale
    out = torch.zeros(B, C, out_H, out_W, device=device)
    count = torch.zeros(B, C, out_H, out_W, device=device)

    stride = tile_size - overlap
    h_tiles = max(1, (H - overlap + stride - 1) // stride)
    w_tiles = max(1, (W - overlap + stride - 1) // stride)

    for h_idx in range(h_tiles):
        for w_idx in range(w_tiles):
            h_start = min(h_idx * stride, max(0, H - tile_size))
            w_start = min(w_idx * stride, max(0, W - tile_size))
            h_end = min(h_start + tile_size, H)
            w_end = min(w_start + tile_size, W)

            tile = img[:, :, h_start:h_end, w_start:w_end]

            with torch.no_grad():
                tile_out = model(tile)

            out_h_start, out_w_start = h_start * scale, w_start * scale
            out_h_end, out_w_end = out_h_start + tile_out.shape[2], out_w_start + tile_out.shape[3]

            out[:, :, out_h_start:out_h_end, out_w_start:out_w_end] += tile_out
            count[:, :, out_h_start:out_h_end, out_w_start:out_w_end] += 1

    return out / count


def process_image(model, image, tile_size=0, tile_overlap=DEFAULT_TILE_OVERLAP, scale=1):
    """Process an image through a model with optional tiling."""
    device = next(model.parameters()).device

    # ComfyUI format: [B, H, W, C] -> Model format: [B, C, H, W]
    img_tensor = image.permute(0, 3, 1, 2).to(device)

    B, C, H, W = img_tensor.shape
    num_pixels = H * W

    if tile_size == 0 and num_pixels > AUTO_TILE_THRESHOLD:
        tile_size = DEFAULT_TILE_SIZE
        print(f"[Restoration] Auto-tiling enabled for {W}x{H} image (tile_size={tile_size})")

    with torch.no_grad():
        if tile_size > 0:
            output = tiled_inference(model, img_tensor, tile_size, tile_overlap, scale)
        else:
            output = model(img_tensor)

    # Convert back to ComfyUI format [B, H, W, C]
    output = output.permute(0, 2, 3, 1).cpu()
    output = torch.clamp(output, 0, 1)

    return output


# =============================================================================
# Model Loading Functions
# =============================================================================

def load_nafnet_model(model_name, device):
    """Load NAFNet/NAFSSR model."""
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nRun download_models.py")

    configs = {
        "NAFNet-SIDD-width32.pth": {"width": 32, "enc_blks": [2, 2, 4, 8], "middle_blk_num": 12, "dec_blks": [2, 2, 2, 2]},
        "NAFNet-SIDD-width64.pth": {"width": 64, "enc_blks": [2, 2, 4, 8], "middle_blk_num": 12, "dec_blks": [2, 2, 2, 2]},
        "NAFNet-GoPro-width32.pth": {"width": 32, "enc_blks": [1, 1, 1, 28], "middle_blk_num": 1, "dec_blks": [1, 1, 1, 1]},
        "NAFNet-GoPro-width64.pth": {"width": 64, "enc_blks": [1, 1, 1, 28], "middle_blk_num": 1, "dec_blks": [1, 1, 1, 1]},
        "NAFNet-REDS-width64.pth": {"width": 64, "enc_blks": [1, 1, 1, 28], "middle_blk_num": 1, "dec_blks": [1, 1, 1, 1]},
    }

    nafssr_configs = {
        "NAFSSR-L_2x.pth": {"up_scale": 2, "width": 64, "num_blks": 64, "fusion_from": 0, "fusion_to": 62},
        "NAFSSR-L_4x.pth": {"up_scale": 4, "width": 64, "num_blks": 64, "fusion_from": 0, "fusion_to": 62},
    }

    if model_name in nafssr_configs:
        cfg = nafssr_configs[model_name]
        model = NAFNetSR(up_scale=cfg["up_scale"], width=cfg["width"], num_blks=cfg["num_blks"],
                         img_channel=3, fusion_from=cfg["fusion_from"], fusion_to=cfg["fusion_to"], dual=True)
    else:
        cfg = configs.get(model_name, {"width": 64, "enc_blks": [2, 2, 4, 8], "middle_blk_num": 12, "dec_blks": [2, 2, 2, 2]})
        model = NAFNet(img_channel=3, width=cfg["width"], middle_blk_num=cfg["middle_blk_num"],
                       enc_blk_nums=cfg["enc_blks"], dec_blk_nums=cfg["dec_blks"])

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(load_state_dict_flexible(checkpoint), strict=True)
    return model.eval().to(device)


def load_scunet_model(model_name, device):
    """Load SCUNet model."""
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nRun download_models.py")

    model = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(load_state_dict_flexible(checkpoint), strict=True)
    return model.eval().to(device)


def load_restormer_model(model_name, device):
    """Load Restormer model."""
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nRun download_models.py")

    model = Restormer(inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8],
                      num_refinement_blocks=4, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                      bias=False, LayerNorm_type='BiasFree', dual_pixel_task=False)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(load_state_dict_flexible(checkpoint), strict=True)
    return model.eval().to(device)


def load_swinir_model(model_name, device, task='denoise'):
    """Load SwinIR model."""
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nRun download_models.py")

    if 'real_sr' in model_name or 'realSR' in model_name:
        # Real-world SR large model
        model = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                       img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                       num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8], mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
    else:
        # Color denoising model
        model = SwinIR(upscale=1, in_chans=3, img_size=128, window_size=8,
                       img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                       num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='', resi_connection='1conv')

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = load_state_dict_flexible(checkpoint)

    # Handle 'module.' prefix
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    return model.eval().to(device)


def load_realesrgan_model(model_name, device):
    """Load Real-ESRGAN model."""
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nRun download_models.py")

    # Determine scale from model name
    if 'x2' in model_name.lower():
        scale = 2
    else:
        scale = 4

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = load_state_dict_flexible(checkpoint)

    # Handle 'module.' prefix
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    return model.eval().to(device)


def load_bsrgan_model(model_name, device):
    """Load BSRGAN model."""
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nRun download_models.py")

    scale = 2 if 'x2' in model_name.lower() else 4
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(load_state_dict_flexible(checkpoint), strict=True)
    return model.eval().to(device)


def load_hinet_model(model_name, device):
    """Load HINet model."""
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nRun download_models.py")

    # Determine width from model name
    if '0.5x' in model_name:
        wf = 32
    else:
        wf = 64

    model = HINet(in_chn=3, wf=wf, depth=5, relu_slope=0.2, hin_position_left=0, hin_position_right=4)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(load_state_dict_flexible(checkpoint), strict=True)
    return model.eval().to(device)


def load_mirnetv2_model(model_name, device):
    """Load MIRNet-v2 model."""
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nRun download_models.py")

    model = MIRNet_v2(inp_channels=3, out_channels=3, n_feat=80, chan_factor=1.5, n_RRG=4, n_MRB=2, height=3, width=2, scale=1)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(load_state_dict_flexible(checkpoint), strict=True)
    return model.eval().to(device)


def load_dncnn_model(model_name, device):
    """Load DnCNN model."""
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nRun download_models.py")

    if 'color' in model_name:
        model = DnCNN(in_nc=3, out_nc=3, nc=64, nb=20, act_mode='R')
    else:
        model = DnCNN(in_nc=1, out_nc=1, nc=64, nb=17, act_mode='BR')

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(load_state_dict_flexible(checkpoint), strict=True)
    return model.eval().to(device)


def load_ffdnet_model(model_name, device):
    """Load FFDNet model."""
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nRun download_models.py")

    if 'color' in model_name:
        model = FFDNet(in_nc=3, out_nc=3, nc=96, nb=12, act_mode='R')
    else:
        model = FFDNet(in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R')

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(load_state_dict_flexible(checkpoint), strict=True)
    return model.eval().to(device)


# =============================================================================
# Node Classes - NAFNet
# =============================================================================

class NAFNetDenoise:
    """Denoise using NAFNet SIDD model - best for smartphone camera noise."""

    _model = None
    _model_name = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_variant": (["width64", "width32"], {"default": "width64"}),
            },
            "optional": {
                "tile_size": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = "image/restoration"

    def denoise(self, image, model_variant="width64", tile_size=0):
        model_name = f"NAFNet-SIDD-{model_variant}.pth"
        device = get_device()
        if NAFNetDenoise._model is None or NAFNetDenoise._model_name != model_name:
            NAFNetDenoise._model = load_nafnet_model(model_name, device)
            NAFNetDenoise._model_name = model_name
        return (process_image(NAFNetDenoise._model, image, tile_size),)


class NAFNetDeblur:
    """Deblur using NAFNet GoPro model - for GoPro-style motion blur only."""

    _model = None
    _model_name = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_variant": (["width64", "width32"], {"default": "width64"}),
            },
            "optional": {
                "tile_size": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "deblur"
    CATEGORY = "image/restoration"

    def deblur(self, image, model_variant="width64", tile_size=0):
        model_name = f"NAFNet-GoPro-{model_variant}.pth"
        device = get_device()
        if NAFNetDeblur._model is None or NAFNetDeblur._model_name != model_name:
            NAFNetDeblur._model = load_nafnet_model(model_name, device)
            NAFNetDeblur._model_name = model_name
        return (process_image(NAFNetDeblur._model, image, tile_size),)


# =============================================================================
# Node Classes - SCUNet
# =============================================================================

class SCUNetDenoise:
    """Blind denoising using SCUNet - excellent for astrophotography and unknown noise."""

    _model = None
    _model_name = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_variant": (["real_gan", "real_psnr"], {"default": "real_gan"}),
            },
            "optional": {
                "tile_size": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = "image/restoration"

    def denoise(self, image, model_variant="real_gan", tile_size=0):
        model_name = f"scunet_color_{model_variant}.pth"
        device = get_device()
        if SCUNetDenoise._model is None or SCUNetDenoise._model_name != model_name:
            SCUNetDenoise._model = load_scunet_model(model_name, device)
            SCUNetDenoise._model_name = model_name
        if tile_size > 0:
            tile_size = max((tile_size // 64) * 64, 64)
        return (process_image(SCUNetDenoise._model, image, tile_size),)


# =============================================================================
# Node Classes - Restormer
# =============================================================================

class RestormerDenoise:
    """Transformer-based denoising using Restormer - state-of-the-art quality."""

    _model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "tile_size": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = "image/restoration"

    def denoise(self, image, tile_size=0):
        device = get_device()
        if RestormerDenoise._model is None:
            RestormerDenoise._model = load_restormer_model("restormer_real_denoising.pth", device)
        return (process_image(RestormerDenoise._model, image, tile_size),)


# =============================================================================
# Node Classes - SwinIR
# =============================================================================

class SwinIRDenoise:
    """Swin Transformer denoising - handles various noise levels."""

    _model = None
    _model_name = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "noise_level": (["15", "25", "50"], {"default": "25"}),
            },
            "optional": {
                "tile_size": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = "image/restoration"

    def denoise(self, image, noise_level="25", tile_size=0):
        model_name = f"swinir_color_dn_noise{noise_level}.pth"
        device = get_device()
        if SwinIRDenoise._model is None or SwinIRDenoise._model_name != model_name:
            SwinIRDenoise._model = load_swinir_model(model_name, device)
            SwinIRDenoise._model_name = model_name
        if tile_size > 0:
            tile_size = max((tile_size // 8) * 8, 64)
        return (process_image(SwinIRDenoise._model, image, tile_size),)


class SwinIRSuperResolution:
    """Swin Transformer real-world super-resolution (4x)."""

    _model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "tile_size": ("INT", {"default": 256, "min": 0, "max": 1024, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/restoration"

    def upscale(self, image, tile_size=256):
        device = get_device()
        if SwinIRSuperResolution._model is None:
            SwinIRSuperResolution._model = load_swinir_model("swinir_real_sr_x4_large.pth", device)
        if tile_size > 0:
            tile_size = max((tile_size // 8) * 8, 64)
        return (process_image(SwinIRSuperResolution._model, image, tile_size, scale=4),)


# =============================================================================
# Node Classes - Real-ESRGAN
# =============================================================================

class RealESRGANUpscale:
    """Real-ESRGAN upscaling with denoising - practical image enhancement."""

    _model = None
    _model_name = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (["RealESRGAN_x4plus", "RealESRGAN_x2plus", "realesr-general-x4v3"], {"default": "RealESRGAN_x4plus"}),
            },
            "optional": {
                "tile_size": ("INT", {"default": 256, "min": 0, "max": 1024, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/restoration"

    def upscale(self, image, model="RealESRGAN_x4plus", tile_size=256):
        model_name = f"{model}.pth"
        device = get_device()
        if RealESRGANUpscale._model is None or RealESRGANUpscale._model_name != model_name:
            RealESRGANUpscale._model = load_realesrgan_model(model_name, device)
            RealESRGANUpscale._model_name = model_name
        scale = 2 if 'x2' in model.lower() else 4
        return (process_image(RealESRGANUpscale._model, image, tile_size, scale=scale),)


# =============================================================================
# Node Classes - BSRGAN
# =============================================================================

class BSRGANUpscale:
    """BSRGAN blind super-resolution - handles unknown degradation."""

    _model = None
    _model_name = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale": (["4x", "2x"], {"default": "4x"}),
            },
            "optional": {
                "tile_size": ("INT", {"default": 256, "min": 0, "max": 1024, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/restoration"

    def upscale(self, image, scale="4x", tile_size=256):
        model_name = "BSRGANx2.pth" if scale == "2x" else "BSRGAN.pth"
        device = get_device()
        if BSRGANUpscale._model is None or BSRGANUpscale._model_name != model_name:
            BSRGANUpscale._model = load_bsrgan_model(model_name, device)
            BSRGANUpscale._model_name = model_name
        scale_factor = 2 if scale == "2x" else 4
        return (process_image(BSRGANUpscale._model, image, tile_size, scale=scale_factor),)


# =============================================================================
# Node Classes - HINet
# =============================================================================

class HINetDenoise:
    """HINet denoising - Half Instance Normalization for image restoration."""

    _model = None
    _model_name = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_variant": (["1x", "0.5x"], {"default": "1x"}),
            },
            "optional": {
                "tile_size": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = "image/restoration"

    def denoise(self, image, model_variant="1x", tile_size=0):
        model_name = f"HINet-SIDD-{model_variant}.pth"
        device = get_device()
        if HINetDenoise._model is None or HINetDenoise._model_name != model_name:
            HINetDenoise._model = load_hinet_model(model_name, device)
            HINetDenoise._model_name = model_name
        return (process_image(HINetDenoise._model, image, tile_size),)


# =============================================================================
# Node Classes - MIRNet-v2
# =============================================================================

class MIRNetV2Denoise:
    """MIRNet-v2 denoising - multi-scale residual learning."""

    _model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "tile_size": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = "image/restoration"

    def denoise(self, image, tile_size=0):
        device = get_device()
        if MIRNetV2Denoise._model is None:
            MIRNetV2Denoise._model = load_mirnetv2_model("MIRNet_v2_real_denoising.pth", device)
        return (process_image(MIRNetV2Denoise._model, image, tile_size),)


# =============================================================================
# Node Classes - DnCNN
# =============================================================================

class DnCNNDenoise:
    """DnCNN denoising - classic CNN denoiser, fast and effective."""

    _model = None
    _model_name = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_variant": (["color_blind", "gray_blind", "15", "25", "50"], {"default": "color_blind"}),
            },
            "optional": {
                "tile_size": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = "image/restoration"

    def denoise(self, image, model_variant="color_blind", tile_size=0):
        model_name = f"dncnn_{model_variant}.pth"
        device = get_device()
        if DnCNNDenoise._model is None or DnCNNDenoise._model_name != model_name:
            DnCNNDenoise._model = load_dncnn_model(model_name, device)
            DnCNNDenoise._model_name = model_name

        # Handle grayscale models
        if 'gray' in model_variant or model_variant in ['15', '25', '50']:
            # Convert to grayscale
            img_tensor = image.permute(0, 3, 1, 2).to(device)
            gray = 0.299 * img_tensor[:, 0:1] + 0.587 * img_tensor[:, 1:2] + 0.114 * img_tensor[:, 2:3]

            with torch.no_grad():
                if tile_size > 0:
                    output = tiled_inference(DnCNNDenoise._model, gray, tile_size, DEFAULT_TILE_OVERLAP)
                else:
                    output = DnCNNDenoise._model(gray)

            # Convert back to RGB
            output = output.repeat(1, 3, 1, 1)
            output = output.permute(0, 2, 3, 1).cpu()
            return (torch.clamp(output, 0, 1),)
        else:
            return (process_image(DnCNNDenoise._model, image, tile_size),)


# =============================================================================
# Node Classes - FFDNet
# =============================================================================

class FFDNetDenoise:
    """FFDNet denoising - fast flexible denoiser with adjustable noise level."""

    _model = None
    _model_name = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "noise_level": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 75.0, "step": 1.0}),
                "color_mode": (["color", "gray"], {"default": "color"}),
            },
            "optional": {
                "tile_size": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = "image/restoration"

    def denoise(self, image, noise_level=25.0, color_mode="color", tile_size=0):
        model_name = f"ffdnet_{color_mode}.pth"
        device = get_device()
        if FFDNetDenoise._model is None or FFDNetDenoise._model_name != model_name:
            FFDNetDenoise._model = load_ffdnet_model(model_name, device)
            FFDNetDenoise._model_name = model_name

        img_tensor = image.permute(0, 3, 1, 2).to(device)

        if color_mode == "gray":
            img_tensor = 0.299 * img_tensor[:, 0:1] + 0.587 * img_tensor[:, 1:2] + 0.114 * img_tensor[:, 2:3]

        # Normalize noise level to [0, 1]
        sigma = torch.full((img_tensor.size(0), 1, 1, 1), noise_level / 255.0, device=device)

        with torch.no_grad():
            output = FFDNetDenoise._model(img_tensor, sigma)

        if color_mode == "gray":
            output = output.repeat(1, 3, 1, 1)

        output = output.permute(0, 2, 3, 1).cpu()
        return (torch.clamp(output, 0, 1),)


# =============================================================================
# Node Mappings
# =============================================================================

NODE_CLASS_MAPPINGS = {
    # NAFNet
    "NAFNetDenoise": NAFNetDenoise,
    "NAFNetDeblur": NAFNetDeblur,
    # SCUNet
    "SCUNetDenoise": SCUNetDenoise,
    # Restormer
    "RestormerDenoise": RestormerDenoise,
    # SwinIR
    "SwinIRDenoise": SwinIRDenoise,
    "SwinIRSuperResolution": SwinIRSuperResolution,
    # Real-ESRGAN
    "RealESRGANUpscale": RealESRGANUpscale,
    # BSRGAN
    "BSRGANUpscale": BSRGANUpscale,
    # HINet
    "HINetDenoise": HINetDenoise,
    # MIRNet-v2
    "MIRNetV2Denoise": MIRNetV2Denoise,
    # DnCNN
    "DnCNNDenoise": DnCNNDenoise,
    # FFDNet
    "FFDNetDenoise": FFDNetDenoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # NAFNet
    "NAFNetDenoise": "NAFNet Denoise (SIDD)",
    "NAFNetDeblur": "NAFNet Deblur (GoPro)",
    # SCUNet
    "SCUNetDenoise": "SCUNet Denoise (Blind/Real)",
    # Restormer
    "RestormerDenoise": "Restormer Denoise (Real)",
    # SwinIR
    "SwinIRDenoise": "SwinIR Denoise",
    "SwinIRSuperResolution": "SwinIR Super-Resolution (4x)",
    # Real-ESRGAN
    "RealESRGANUpscale": "Real-ESRGAN Upscale",
    # BSRGAN
    "BSRGANUpscale": "BSRGAN Upscale (Blind SR)",
    # HINet
    "HINetDenoise": "HINet Denoise (SIDD)",
    # MIRNet-v2
    "MIRNetV2Denoise": "MIRNet-v2 Denoise",
    # DnCNN
    "DnCNNDenoise": "DnCNN Denoise",
    # FFDNet
    "FFDNetDenoise": "FFDNet Denoise (Adjustable)",
}
