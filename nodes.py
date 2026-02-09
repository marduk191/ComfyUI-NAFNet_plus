import os
import torch
import folder_paths

try:
    from .nafnet_arch import NAFNet, NAFNetSR
except ImportError:
    from nafnet_arch import NAFNet, NAFNetSR

# Register model folder
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Model configurations matching the official pretrained weights
MODEL_CONFIGS = {
    # Denoising models (SIDD dataset)
    "NAFNet-SIDD-width32.pth": {"type": "nafnet", "width": 32, "enc_blks": [2, 2, 4, 8], "middle_blk_num": 12, "dec_blks": [2, 2, 2, 2]},
    "NAFNet-SIDD-width64.pth": {"type": "nafnet", "width": 64, "enc_blks": [2, 2, 4, 8], "middle_blk_num": 12, "dec_blks": [2, 2, 2, 2]},
    # Deblurring models (GoPro dataset)
    "NAFNet-GoPro-width32.pth": {"type": "nafnet", "width": 32, "enc_blks": [1, 1, 1, 28], "middle_blk_num": 1, "dec_blks": [1, 1, 1, 1]},
    "NAFNet-GoPro-width64.pth": {"type": "nafnet", "width": 64, "enc_blks": [1, 1, 1, 28], "middle_blk_num": 1, "dec_blks": [1, 1, 1, 1]},
    # REDS deblurring (video deblurring)
    "NAFNet-REDS-width64.pth": {"type": "nafnet", "width": 64, "enc_blks": [1, 1, 1, 28], "middle_blk_num": 1, "dec_blks": [1, 1, 1, 1]},
    # Stereo Super-Resolution (NAFSSR)
    "NAFSSR-L_2x.pth": {"type": "nafssr", "up_scale": 2, "width": 64, "num_blks": 64, "fusion_from": 0, "fusion_to": 62},
    "NAFSSR-L_4x.pth": {"type": "nafssr", "up_scale": 4, "width": 64, "num_blks": 64, "fusion_from": 0, "fusion_to": 62},
}


def get_available_models():
    """Return list of available model files in the models directory."""
    if not os.path.exists(MODELS_DIR):
        return []
    models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pth')]
    return models if models else ["No models found - download from NAFNet repo"]


def load_nafnet_model(model_name, device):
    """Load a NAFNet model from the models directory."""
    model_path = os.path.join(MODELS_DIR, model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nDownload from: https://github.com/megvii-research/NAFNet")

    # Get config for this model
    config = MODEL_CONFIGS.get(model_name)
    if config is None:
        # Default config for unknown models
        config = {"type": "nafnet", "width": 64, "enc_blks": [2, 2, 4, 8], "middle_blk_num": 12, "dec_blks": [2, 2, 2, 2]}

    # Create model based on type
    model_type = config.get("type", "nafnet")
    if model_type == "nafssr":
        model = NAFNetSR(
            up_scale=config["up_scale"],
            width=config["width"],
            num_blks=config["num_blks"],
            img_channel=3,
            fusion_from=config.get("fusion_from", -1),
            fusion_to=config.get("fusion_to", -1),
            dual=True
        )
    else:
        model = NAFNet(
            img_channel=3,
            width=config["width"],
            middle_blk_num=config["middle_blk_num"],
            enc_blk_nums=config["enc_blks"],
            dec_blk_nums=config["dec_blks"]
        )

    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'params' in checkpoint:
            state_dict = checkpoint['params']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)

    return model


class NAFNetLoader:
    """Load a NAFNet model for image restoration."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (get_available_models(),),
            }
        }

    RETURN_TYPES = ("NAFNET_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "image/restoration"

    def load_model(self, model_name):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_nafnet_model(model_name, device)
        return (model,)


class NAFNetRestore:
    """Apply NAFNet model to restore an image (denoise or deblur)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("NAFNET_MODEL",),
            },
            "optional": {
                "tile_size": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64,
                                       "tooltip": "Tile size for processing large images. 0 = no tiling"}),
                "tile_overlap": ("INT", {"default": 32, "min": 0, "max": 256, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "restore"
    CATEGORY = "image/restoration"

    def restore(self, image, model, tile_size=0, tile_overlap=32):
        device = next(model.parameters()).device

        # ComfyUI format: [B, H, W, C] float32 0-1
        # NAFNet expects: [B, C, H, W] float32 0-1
        img_tensor = image.permute(0, 3, 1, 2).to(device)

        with torch.no_grad():
            if tile_size > 0:
                output = self._tiled_inference(model, img_tensor, tile_size, tile_overlap)
            else:
                output = model(img_tensor)

        # Convert back to ComfyUI format [B, H, W, C]
        output = output.permute(0, 2, 3, 1).cpu()
        output = torch.clamp(output, 0, 1)

        return (output,)

    def _tiled_inference(self, model, img, tile_size, overlap):
        """Process image in tiles for memory efficiency."""
        B, C, H, W = img.shape

        # Calculate output size
        out = torch.zeros_like(img)
        count = torch.zeros_like(img)

        # Calculate number of tiles
        stride = tile_size - overlap
        h_tiles = max(1, (H - overlap) // stride + (1 if (H - overlap) % stride else 0))
        w_tiles = max(1, (W - overlap) // stride + (1 if (W - overlap) % stride else 0))

        for h_idx in range(h_tiles):
            for w_idx in range(w_tiles):
                h_start = min(h_idx * stride, H - tile_size)
                w_start = min(w_idx * stride, W - tile_size)
                h_start = max(0, h_start)
                w_start = max(0, w_start)
                h_end = min(h_start + tile_size, H)
                w_end = min(w_start + tile_size, W)

                tile = img[:, :, h_start:h_end, w_start:w_end]
                tile_out = model(tile)

                out[:, :, h_start:h_end, w_start:w_end] += tile_out
                count[:, :, h_start:h_end, w_start:w_end] += 1

        return out / count


class NAFNetDenoise:
    """Denoise an image using NAFNet SIDD model."""

    _model = None
    _model_name = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_variant": (["width64", "width32"], {"default": "width64"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = "image/restoration"

    def denoise(self, image, model_variant="width64"):
        model_name = f"NAFNet-SIDD-{model_variant}.pth"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cache model
        if NAFNetDenoise._model is None or NAFNetDenoise._model_name != model_name:
            NAFNetDenoise._model = load_nafnet_model(model_name, device)
            NAFNetDenoise._model_name = model_name

        model = NAFNetDenoise._model

        # ComfyUI format: [B, H, W, C] -> NAFNet: [B, C, H, W]
        img_tensor = image.permute(0, 3, 1, 2).to(device)

        with torch.no_grad():
            output = model(img_tensor)

        output = output.permute(0, 2, 3, 1).cpu()
        output = torch.clamp(output, 0, 1)

        return (output,)


class NAFNetDeblur:
    """Deblur an image using NAFNet GoPro model."""

    _model = None
    _model_name = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_variant": (["width64", "width32"], {"default": "width64"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "deblur"
    CATEGORY = "image/restoration"

    def deblur(self, image, model_variant="width64"):
        model_name = f"NAFNet-GoPro-{model_variant}.pth"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cache model
        if NAFNetDeblur._model is None or NAFNetDeblur._model_name != model_name:
            NAFNetDeblur._model = load_nafnet_model(model_name, device)
            NAFNetDeblur._model_name = model_name

        model = NAFNetDeblur._model

        # ComfyUI format: [B, H, W, C] -> NAFNet: [B, C, H, W]
        img_tensor = image.permute(0, 3, 1, 2).to(device)

        with torch.no_grad():
            output = model(img_tensor)

        output = output.permute(0, 2, 3, 1).cpu()
        output = torch.clamp(output, 0, 1)

        return (output,)


class NAFSSRStereoSR:
    """Stereo image super-resolution using NAFSSR."""

    _model = None
    _model_name = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "left_image": ("IMAGE",),
                "right_image": ("IMAGE",),
                "scale": (["2x", "4x"], {"default": "2x"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("left_sr", "right_sr",)
    FUNCTION = "stereo_sr"
    CATEGORY = "image/restoration"

    def stereo_sr(self, left_image, right_image, scale="2x"):
        model_name = f"NAFSSR-L_{scale}.pth"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cache model
        if NAFSSRStereoSR._model is None or NAFSSRStereoSR._model_name != model_name:
            NAFSSRStereoSR._model = load_nafnet_model(model_name, device)
            NAFSSRStereoSR._model_name = model_name

        model = NAFSSRStereoSR._model

        # ComfyUI format: [B, H, W, C] -> [B, C, H, W]
        left_tensor = left_image.permute(0, 3, 1, 2).to(device)
        right_tensor = right_image.permute(0, 3, 1, 2).to(device)

        # NAFSSR expects concatenated stereo pair: [B, 6, H, W]
        stereo_input = torch.cat([left_tensor, right_tensor], dim=1)

        with torch.no_grad():
            output = model(stereo_input)

        # Output is [B, 6, H*scale, W*scale], split into left and right
        left_sr = output[:, :3, :, :]
        right_sr = output[:, 3:, :, :]

        # Convert back to ComfyUI format [B, H, W, C]
        left_sr = left_sr.permute(0, 2, 3, 1).cpu()
        right_sr = right_sr.permute(0, 2, 3, 1).cpu()
        left_sr = torch.clamp(left_sr, 0, 1)
        right_sr = torch.clamp(right_sr, 0, 1)

        return (left_sr, right_sr)


NODE_CLASS_MAPPINGS = {
    "NAFNetLoader": NAFNetLoader,
    "NAFNetRestore": NAFNetRestore,
    "NAFNetDenoise": NAFNetDenoise,
    "NAFNetDeblur": NAFNetDeblur,
    "NAFSSRStereoSR": NAFSSRStereoSR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NAFNetLoader": "NAFNet Load Model",
    "NAFNetRestore": "NAFNet Restore",
    "NAFNetDenoise": "NAFNet Denoise (SIDD)",
    "NAFNetDeblur": "NAFNet Deblur (GoPro)",
    "NAFSSRStereoSR": "NAFSSR Stereo Super-Resolution",
}
