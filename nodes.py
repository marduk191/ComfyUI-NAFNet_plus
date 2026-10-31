"""
ComfyUI-NAFNet_plus: Comprehensive Image Restoration Nodes

Supported Models:
- NAFNet: Denoising (SIDD), Deblurring (GoPro)
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

import logging
import os

import torch
import torch.nn.functional as F

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    import comfy.model_management as model_management
    import comfy.utils as comfy_utils
except ImportError:
    model_management = None
    comfy_utils = None

try:
    from . import download_models
    from .dncnn_arch import DnCNN
    from .ffdnet_arch import FFDNet
    from .hinet_arch import HINet
    from .mirnetv2_arch import MIRNet_v2
    from .nafnet_arch import NAFNet
    from .restormer_arch import Restormer
    from .rrdbnet_arch import RealESRGANRRDBNet, RRDBNet
    from .scunet_arch import SCUNet
    from .srvgg_arch import SRVGGNetCompact
    from .swinir_arch import SwinIR
except ImportError:
    import download_models
    from dncnn_arch import DnCNN
    from ffdnet_arch import FFDNet
    from hinet_arch import HINet
    from mirnetv2_arch import MIRNet_v2
    from nafnet_arch import NAFNet
    from restormer_arch import Restormer
    from rrdbnet_arch import RealESRGANRRDBNet, RRDBNet
    from scunet_arch import SCUNet
    from srvgg_arch import SRVGGNetCompact
    from swinir_arch import SwinIR

log = logging.getLogger("NAFNet_plus")

# Model folders: ComfyUI/models/nafnet_plus (default) + this pack's models/ (legacy installs)
MODELS_FOLDER_NAME = "nafnet_plus"
MODELS_DIR = download_models.default_models_dir()
LEGACY_MODELS_DIR = download_models.LEGACY_MODELS_DIR

if folder_paths is not None:
    folder_paths.add_model_folder_path(MODELS_FOLDER_NAME, MODELS_DIR, is_default=True)
    if os.path.isdir(LEGACY_MODELS_DIR):
        folder_paths.add_model_folder_path(MODELS_FOLDER_NAME, LEGACY_MODELS_DIR)

# Auto-tile threshold (images larger than this will be tiled automatically)
AUTO_TILE_THRESHOLD = 1024 * 1024  # 1 megapixel
DEFAULT_TILE_SIZE = 512
DEFAULT_TILE_OVERLAP = 64

CATEGORY = "image/restoration"


# =============================================================================
# Device / memory helpers
# =============================================================================

def get_device():
    """Inference device chosen by ComfyUI (respects --cpu, --directml, MPS, XPU...)."""
    if model_management is not None:
        return model_management.get_torch_device()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_offload_device():
    if model_management is not None:
        return model_management.unet_offload_device()
    return torch.device("cpu")


def check_interrupt():
    if model_management is not None:
        model_management.throw_exception_if_processing_interrupted()


def soft_empty_cache():
    if model_management is not None:
        model_management.soft_empty_cache()


def free_vram_for(model, device):
    """Ask ComfyUI to unload other models so this one (plus activations) fits."""
    if model_management is None or device.type == "cpu":
        return
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    try:
        model_management.free_memory(model_bytes * 3 + 1024**3, device)
    except Exception as e:  # never fail inference because of a memory hint
        log.debug("free_memory failed: %s", e)


# =============================================================================
# Model files
# =============================================================================

def get_model_path(model_name):
    """Locate a model file, downloading it on first use if missing."""
    search_dirs = []
    if folder_paths is not None:
        search_dirs.extend(folder_paths.get_folder_paths(MODELS_FOLDER_NAME))
    search_dirs.extend([MODELS_DIR, LEGACY_MODELS_DIR])
    search_dirs = list(dict.fromkeys(search_dirs))

    if model_name in download_models.ALL_MODELS:
        existing = download_models.find_existing(model_name, search_dirs)
    else:
        existing = next((os.path.join(d, model_name) for d in search_dirs
                         if os.path.isfile(os.path.join(d, model_name))), None)
    if existing:
        return existing

    if model_name not in download_models.ALL_MODELS:
        raise FileNotFoundError(f"Model not found: {model_name} (searched {search_dirs})")

    log.info("[NAFNet_plus] %s not found, downloading to %s", model_name, MODELS_DIR)
    try:
        return download_models.download_model(model_name, MODELS_DIR)
    except Exception as e:
        raise FileNotFoundError(
            f"Model {model_name} is missing and the automatic download failed: {e}\n"
            f"Download it manually with: python download_models.py {model_name}\n"
            f"and place it in {MODELS_DIR}"
        ) from e


def load_checkpoint(model_path):
    """Load weights safely (weights_only) and unwrap common checkpoint containers."""
    if comfy_utils is not None:
        sd = comfy_utils.load_torch_file(model_path, safe_load=True)
    elif model_path.lower().endswith(".safetensors"):
        import safetensors.torch

        sd = safetensors.torch.load_file(model_path)
    else:
        sd = torch.load(model_path, map_location="cpu", weights_only=True)

    if isinstance(sd, dict):
        for key in ("params_ema", "params", "state_dict", "model", "model_state_dict"):
            if key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
    if sd and next(iter(sd)).startswith("module."):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    return sd


class RestorationModel:
    """A loaded network plus the metadata needed to run it."""

    def __init__(self, net, scale=1, pad_multiple=1, in_channels=3):
        self.net = net.eval().requires_grad_(False)
        self.scale = scale
        self.pad_multiple = pad_multiple
        self.in_channels = in_channels

    def __call__(self, x, *args):
        """Run the network, padding H/W to the required multiple and cropping back."""
        h, w = x.shape[-2:]
        m = self.pad_multiple
        pad_h, pad_w = (m - h % m) % m, (m - w % m) % m
        if pad_h or pad_w:
            mode = "reflect" if pad_h < h and pad_w < w else "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        out = self.net(x, *args)
        if isinstance(out, (list, tuple)):  # multi-stage nets (HINet) return every stage
            out = out[-1]
        return out[..., : h * self.scale, : w * self.scale]


_MODEL_CACHE = {}


def get_model(model_name, builder, **meta):
    """Load (or fetch from cache) a model. Cached models live on the offload device."""
    if model_name not in _MODEL_CACHE:
        path = get_model_path(model_name)
        net = builder(model_name)
        net.load_state_dict(load_checkpoint(path), strict=True)
        _MODEL_CACHE[model_name] = RestorationModel(net.to(get_offload_device()), **meta)
        log.info("[NAFNet_plus] Loaded %s", model_name)
    return _MODEL_CACHE[model_name]


# =============================================================================
# Inference
# =============================================================================

def tiled_inference(fn, img, tile_size, overlap, scale=1, out_channels=None, pbar=None):
    """Process a [B, C, H, W] tensor in overlapping tiles; accumulates on the CPU."""
    B, C, H, W = img.shape
    out_channels = out_channels or C
    tile_size = max(tile_size, 64)
    overlap = min(overlap, tile_size // 2)

    stride = tile_size - overlap
    h_starts = list(range(0, max(H - tile_size, 0) + 1, stride))
    w_starts = list(range(0, max(W - tile_size, 0) + 1, stride))
    if h_starts[-1] + tile_size < H:
        h_starts.append(H - tile_size)
    if w_starts[-1] + tile_size < W:
        w_starts.append(W - tile_size)

    out = torch.zeros(B, out_channels, H * scale, W * scale)
    count = torch.zeros(1, 1, H * scale, W * scale)

    for hs in h_starts:
        for ws in w_starts:
            check_interrupt()
            tile = img[:, :, hs : hs + tile_size, ws : ws + tile_size]
            tile_out = fn(tile).float().cpu()
            oh, ow = hs * scale, ws * scale
            th, tw = tile_out.shape[-2:]
            out[:, :, oh : oh + th, ow : ow + tw] += tile_out
            count[:, :, oh : oh + th, ow : ow + tw] += 1
            if pbar is not None:
                pbar.update(1)

    return out / count


def count_tiles(H, W, tile_size, overlap):
    tile_size = max(tile_size, 64)
    stride = tile_size - min(overlap, tile_size // 2)

    def n(size):
        if size <= tile_size:
            return 1
        k = (size - tile_size) // stride + 1
        return k + (1 if (k - 1) * stride + tile_size < size else 0)

    return n(H) * n(W)


def run_model(model, image, tile_size=0, tile_overlap=DEFAULT_TILE_OVERLAP, tile_multiple=8,
              forward=None, out_channels=3):
    """
    Run a RestorationModel on a ComfyUI IMAGE ([B, H, W, C], 0..1) and return an IMAGE.

    ``forward(model, x)`` may be supplied to customise the call (e.g. extra inputs).
    """
    device = get_device()
    forward = forward or (lambda m, x: m(x))

    img = image[..., :3].permute(0, 3, 1, 2).float()
    B, C, H, W = img.shape

    if tile_size == 0 and H * W > AUTO_TILE_THRESHOLD:
        tile_size = DEFAULT_TILE_SIZE
        log.info("[NAFNet_plus] Auto-tiling %dx%d image (tile_size=%d)", W, H, tile_size)
    if tile_size > 0:
        tile_size = max((tile_size // tile_multiple) * tile_multiple, 64)
    use_tiles = tile_size > 0 and (H > tile_size or W > tile_size)

    free_vram_for(model.net, device)
    model.net.to(device)
    try:
        with torch.inference_mode():
            outputs = []
            pbar = None
            if comfy_utils is not None:
                total = B * (count_tiles(H, W, tile_size, tile_overlap) if use_tiles else 1)
                pbar = comfy_utils.ProgressBar(total)
            for i in range(B):
                check_interrupt()
                x = img[i : i + 1].to(device)

                def fn(t):
                    return forward(model, t)

                if use_tiles:
                    y = tiled_inference(fn, x, tile_size, tile_overlap, model.scale, out_channels, pbar)
                else:
                    y = fn(x).float().cpu()
                    if pbar is not None:
                        pbar.update(1)
                outputs.append(y)
            output = torch.cat(outputs, dim=0)
    finally:
        model.net.to(get_offload_device())
        soft_empty_cache()

    if output.shape[1] == 1:
        output = output.repeat(1, 3, 1, 1)
    return torch.clamp(output.permute(0, 2, 3, 1), 0, 1).contiguous()


def to_gray(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


# =============================================================================
# Model builders
# =============================================================================

NAFNET_CONFIGS = {
    "NAFNet-SIDD-width32.pth": dict(width=32, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2]),
    "NAFNet-SIDD-width64.pth": dict(width=64, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2]),
    "NAFNet-GoPro-width32.pth": dict(width=32, enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1]),
    "NAFNet-GoPro-width64.pth": dict(width=64, enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1]),
}


def build_nafnet(name):
    return NAFNet(img_channel=3, **NAFNET_CONFIGS[name])


def build_scunet(name):
    return SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)


def build_restormer(name):
    return Restormer(inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8],
                     num_refinement_blocks=4, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                     bias=False, LayerNorm_type="BiasFree", dual_pixel_task=False)


def build_swinir(name):
    if "real_sr" in name:
        return SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.0,
                      depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240, num_heads=[8] * 9,
                      mlp_ratio=2, upsampler="nearest+conv", resi_connection="3conv")
    return SwinIR(upscale=1, in_chans=3, img_size=128, window_size=8, img_range=1.0,
                  depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6] * 6,
                  mlp_ratio=2, upsampler="", resi_connection="1conv")


def build_realesrgan(name):
    if name == "realesr-general-x4v3.pth":
        return SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu")
    scale = 2 if "x2" in name.lower() else 4
    return RealESRGANRRDBNet(num_in_ch=3, num_out_ch=3, scale=scale, num_feat=64, num_block=23, num_grow_ch=32)


def build_bsrgan(name):
    return RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=2 if "x2" in name.lower() else 4)


def build_hinet(name):
    return HINet(in_chn=3, wf=32 if "0.5x" in name else 64, depth=5, relu_slope=0.2,
                 hin_position_left=0, hin_position_right=4)


def build_mirnetv2(name):
    return MIRNet_v2(inp_channels=3, out_channels=3, n_feat=80, chan_factor=1.5, n_RRG=4, n_MRB=2,
                     height=3, width=2, scale=1)


def build_dncnn(name):
    if "color" in name:
        return DnCNN(in_nc=3, out_nc=3, nc=64, nb=20, act_mode="R")
    if "gray_blind" in name:
        return DnCNN(in_nc=1, out_nc=1, nc=64, nb=20, act_mode="R")
    # dncnn_15/25/50 are released with batch-norm merged into the conv weights
    return DnCNN(in_nc=1, out_nc=1, nc=64, nb=17, act_mode="R")


def build_ffdnet(name):
    if "color" in name:
        return FFDNet(in_nc=3, out_nc=3, nc=96, nb=12, act_mode="R")
    return FFDNet(in_nc=1, out_nc=1, nc=64, nb=15, act_mode="R")


def tile_input(default=0, max_value=2048):
    return ("INT", {
        "default": default, "min": 0, "max": max_value, "step": 64,
        "tooltip": "Tile size for low-VRAM processing. 0 = auto (tiles images over 1 megapixel).",
    })


# =============================================================================
# Node Classes - NAFNet
# =============================================================================

class NAFNetDenoise:
    """Denoise using NAFNet SIDD model - best for smartphone camera noise."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_variant": (["width64", "width32"], {"default": "width64"}),
            },
            "optional": {"tile_size": tile_input()},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = CATEGORY
    DESCRIPTION = "NAFNet trained on SIDD - removes real smartphone camera noise."

    def denoise(self, image, model_variant="width64", tile_size=0):
        model = get_model(f"NAFNet-SIDD-{model_variant}.pth", build_nafnet, pad_multiple=16)
        return (run_model(model, image, tile_size, tile_multiple=16),)


class NAFNetDeblur:
    """Deblur using NAFNet GoPro model - for GoPro-style motion blur only."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_variant": (["width64", "width32"], {"default": "width64"}),
            },
            "optional": {"tile_size": tile_input()},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "deblur"
    CATEGORY = CATEGORY
    DESCRIPTION = "NAFNet trained on GoPro - removes camera-shake motion blur."

    def deblur(self, image, model_variant="width64", tile_size=0):
        model = get_model(f"NAFNet-GoPro-{model_variant}.pth", build_nafnet, pad_multiple=16)
        return (run_model(model, image, tile_size, tile_multiple=16),)


# =============================================================================
# Node Classes - SCUNet
# =============================================================================

class SCUNetDenoise:
    """Blind denoising using SCUNet - excellent for astrophotography and unknown noise."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_variant": (["real_gan", "real_psnr"], {"default": "real_gan"}),
            },
            "optional": {"tile_size": tile_input()},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = CATEGORY
    DESCRIPTION = "SCUNet blind denoiser for real-world noise of unknown type (good for astrophotography)."

    def denoise(self, image, model_variant="real_gan", tile_size=0):
        model = get_model(f"scunet_color_{model_variant}.pth", build_scunet, pad_multiple=64)
        return (run_model(model, image, tile_size, tile_multiple=64),)


# =============================================================================
# Node Classes - Restormer
# =============================================================================

class RestormerDenoise:
    """Transformer-based denoising using Restormer - state-of-the-art quality."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {"tile_size": tile_input()},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = CATEGORY
    DESCRIPTION = "Restormer trained on SIDD real noise - high quality transformer denoiser."

    def denoise(self, image, tile_size=0):
        model = get_model("restormer_real_denoising.pth", build_restormer, pad_multiple=8)
        return (run_model(model, image, tile_size),)


# =============================================================================
# Node Classes - SwinIR
# =============================================================================

class SwinIRDenoise:
    """Swin Transformer denoising - handles various noise levels."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "noise_level": (["15", "25", "50"], {"default": "25"}),
            },
            "optional": {"tile_size": tile_input()},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = CATEGORY
    DESCRIPTION = "SwinIR color denoiser trained for Gaussian noise sigma 15/25/50."

    def denoise(self, image, noise_level="25", tile_size=0):
        model = get_model(f"swinir_color_dn_noise{noise_level}.pth", build_swinir, pad_multiple=8)
        return (run_model(model, image, tile_size),)


class SwinIRSuperResolution:
    """Swin Transformer real-world super-resolution (4x)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {"tile_size": tile_input(256, 1024)},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = CATEGORY
    DESCRIPTION = "SwinIR-L real-world 4x super-resolution (BSRGAN degradations, GAN)."

    def upscale(self, image, tile_size=256):
        model = get_model("swinir_real_sr_x4_large.pth", build_swinir, scale=4, pad_multiple=8)
        return (run_model(model, image, tile_size),)


# =============================================================================
# Node Classes - Real-ESRGAN
# =============================================================================

class RealESRGANUpscale:
    """Real-ESRGAN upscaling with denoising - practical image enhancement."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (["RealESRGAN_x4plus", "RealESRGAN_x2plus", "realesr-general-x4v3"],
                          {"default": "RealESRGAN_x4plus"}),
            },
            "optional": {"tile_size": tile_input(256, 1024)},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = CATEGORY
    DESCRIPTION = "Real-ESRGAN 2x/4x upscaling with artifact and noise removal."

    def upscale(self, image, model="RealESRGAN_x4plus", tile_size=256):
        scale = 2 if "x2" in model.lower() else 4
        net = get_model(f"{model}.pth", build_realesrgan, scale=scale, pad_multiple=2 if scale == 2 else 1)
        return (run_model(net, image, tile_size),)


# =============================================================================
# Node Classes - BSRGAN
# =============================================================================

class BSRGANUpscale:
    """BSRGAN blind super-resolution - handles unknown degradation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale": (["4x", "2x"], {"default": "4x"}),
            },
            "optional": {"tile_size": tile_input(256, 1024)},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = CATEGORY
    DESCRIPTION = "BSRGAN blind super-resolution for images with unknown degradation."

    def upscale(self, image, scale="4x", tile_size=256):
        model_name = "BSRGANx2.pth" if scale == "2x" else "BSRGAN.pth"
        model = get_model(model_name, build_bsrgan, scale=2 if scale == "2x" else 4)
        return (run_model(model, image, tile_size),)


# =============================================================================
# Node Classes - HINet
# =============================================================================

class HINetDenoise:
    """HINet denoising - Half Instance Normalization for image restoration."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_variant": (["1x", "0.5x"], {"default": "1x"}),
            },
            "optional": {"tile_size": tile_input()},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = CATEGORY
    DESCRIPTION = "HINet trained on SIDD real noise - fast denoiser."

    def denoise(self, image, model_variant="1x", tile_size=0):
        model = get_model(f"HINet-SIDD-{model_variant}.pth", build_hinet, pad_multiple=16)
        return (run_model(model, image, tile_size, tile_multiple=16),)


# =============================================================================
# Node Classes - MIRNet-v2
# =============================================================================

class MIRNetV2Denoise:
    """MIRNet-v2 denoising - multi-scale residual learning."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {"tile_size": tile_input()},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = CATEGORY
    DESCRIPTION = "MIRNet-v2 trained on SIDD real noise - multi-scale denoiser."

    def denoise(self, image, tile_size=0):
        model = get_model("MIRNet_v2_real_denoising.pth", build_mirnetv2, pad_multiple=8)
        return (run_model(model, image, tile_size),)


# =============================================================================
# Node Classes - DnCNN
# =============================================================================

class DnCNNDenoise:
    """DnCNN denoising - classic CNN denoiser, fast and effective."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_variant": (["color_blind", "gray_blind", "15", "25", "50"], {"default": "color_blind"}),
            },
            "optional": {"tile_size": tile_input()},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = CATEGORY
    DESCRIPTION = "DnCNN Gaussian denoiser. Non-color variants output grayscale."

    def denoise(self, image, model_variant="color_blind", tile_size=0):
        model = get_model(f"dncnn_{model_variant}.pth", build_dncnn,
                          in_channels=3 if "color" in model_variant else 1)
        if model.in_channels == 1:
            return (run_model(model, image, tile_size, forward=lambda m, x: m(to_gray(x)), out_channels=1),)
        return (run_model(model, image, tile_size),)


# =============================================================================
# Node Classes - FFDNet
# =============================================================================

class FFDNetDenoise:
    """FFDNet denoising - fast flexible denoiser with adjustable noise level."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "noise_level": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 75.0, "step": 1.0,
                                          "tooltip": "Noise sigma on a 0-255 scale."}),
                "color_mode": (["color", "gray"], {"default": "color"}),
            },
            "optional": {"tile_size": tile_input()},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = CATEGORY
    DESCRIPTION = "FFDNet denoiser with an adjustable noise level."

    def denoise(self, image, noise_level=25.0, color_mode="color", tile_size=0):
        model = get_model(f"ffdnet_{color_mode}.pth", build_ffdnet, pad_multiple=2,
                          in_channels=3 if color_mode == "color" else 1)

        def forward(m, x):
            if m.in_channels == 1:
                x = to_gray(x)
            sigma = torch.full((x.shape[0], 1, 1, 1), noise_level / 255.0, device=x.device, dtype=x.dtype)
            return m(x, sigma)

        return (run_model(model, image, tile_size, forward=forward, out_channels=model.in_channels),)


# =============================================================================
# Node Mappings
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "NAFNetDenoise": NAFNetDenoise,
    "NAFNetDeblur": NAFNetDeblur,
    "SCUNetDenoise": SCUNetDenoise,
    "RestormerDenoise": RestormerDenoise,
    "SwinIRDenoise": SwinIRDenoise,
    "SwinIRSuperResolution": SwinIRSuperResolution,
    "RealESRGANUpscale": RealESRGANUpscale,
    "BSRGANUpscale": BSRGANUpscale,
    "HINetDenoise": HINetDenoise,
    "MIRNetV2Denoise": MIRNetV2Denoise,
    "DnCNNDenoise": DnCNNDenoise,
    "FFDNetDenoise": FFDNetDenoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NAFNetDenoise": "NAFNet Denoise (SIDD)",
    "NAFNetDeblur": "NAFNet Deblur (GoPro)",
    "SCUNetDenoise": "SCUNet Denoise (Blind/Real)",
    "RestormerDenoise": "Restormer Denoise (Real)",
    "SwinIRDenoise": "SwinIR Denoise",
    "SwinIRSuperResolution": "SwinIR Super-Resolution (4x)",
    "RealESRGANUpscale": "Real-ESRGAN Upscale",
    "BSRGANUpscale": "BSRGAN Upscale (Blind SR)",
    "HINetDenoise": "HINet Denoise (SIDD)",
    "MIRNetV2Denoise": "MIRNet-v2 Denoise",
    "DnCNNDenoise": "DnCNN Denoise",
    "FFDNetDenoise": "FFDNet Denoise (Adjustable)",
}
