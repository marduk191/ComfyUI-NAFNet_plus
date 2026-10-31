"""
Download pretrained models for image restoration.

Run this script after installation to pre-download all model weights, or
pass model filenames to fetch only those. The nodes also download any missing
model automatically the first time it is used.

    python download_models.py                      # everything
    python download_models.py scunet_color_real_gan.pth RealESRGAN_x4plus.pth

Models are saved to ``ComfyUI/models/nafnet_plus`` when this pack is installed
inside ComfyUI's ``custom_nodes`` folder, otherwise to ``./models``. Files already
present in this pack's own ``models`` folder (older installs) are still used.

Sources:
- NAFNet: https://github.com/megvii-research/NAFNet
- SCUNet: https://github.com/cszn/SCUNet
- Restormer: https://github.com/swz30/Restormer
- SwinIR: https://github.com/JingyunLiang/SwinIR
- Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN
- BSRGAN: https://github.com/cszn/BSRGAN
- HINet: https://github.com/megvii-model/HINet
- MIRNet-v2: https://github.com/swz30/MIRNetv2
- DnCNN/FFDNet: https://github.com/cszn/KAIR
"""
import os
import subprocess
import sys

PACK_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_MODELS_DIR = os.path.join(PACK_DIR, "models")


def default_models_dir():
    """ComfyUI/models/nafnet_plus if installed in custom_nodes, else ./models."""
    try:
        import folder_paths  # available when running inside ComfyUI

        return os.path.join(folder_paths.models_dir, "nafnet_plus")
    except ImportError:
        pass
    comfy_root = os.path.dirname(os.path.dirname(PACK_DIR))
    if os.path.basename(os.path.dirname(PACK_DIR)) == "custom_nodes" and os.path.isdir(
        os.path.join(comfy_root, "models")
    ):
        return os.path.join(comfy_root, "models", "nafnet_plus")
    return LEGACY_MODELS_DIR


# Google Drive file IDs
# Format: filename -> (file_id, expected_size_bytes)
GDRIVE_MODELS = {
    # NAFNet models
    "NAFNet-SIDD-width32.pth": ("1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ", 116_861_841),
    "NAFNet-SIDD-width64.pth": ("14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR", 464_154_961),
    "NAFNet-GoPro-width32.pth": ("1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj", 68_671_121),
    "NAFNet-GoPro-width64.pth": ("1S0PVRbyTakYY9a82kujgZLbMihfNBLfC", 271_778_961),
    # HINet models
    "HINet-SIDD-1x.pth": ("1CU5z-M90Jc-TAcVpEaFjDCYA09fkubGi", 354_715_486),
    "HINet-SIDD-0.5x.pth": ("1Y5YJQVNL0weifE--5us344bLwzBNS_sU", 88_742_632),
}

# Direct URL downloads (GitHub releases)
# Format: filename -> (url, expected_size_bytes)
URL_MODELS = {
    # SCUNet models (KAIR)
    "scunet_color_real_gan.pth": ("https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_gan.pth", 71_982_835),
    "scunet_color_real_psnr.pth": ("https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth", 71_982_841),

    # Restormer - real image denoising (SIDD)
    "restormer_real_denoising.pth": ("https://github.com/swz30/Restormer/releases/download/v1.0/real_denoising.pth", 104_611_957),

    # MIRNet-v2 - real image denoising (SIDD)
    "MIRNet_v2_real_denoising.pth": ("https://github.com/swz30/MIRNetv2/releases/download/v1.0.0/real_denoising.pth", 23_560_589),

    # SwinIR models - Color Denoising
    "swinir_color_dn_noise15.pth": ("https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth", 122_905_743),
    "swinir_color_dn_noise25.pth": ("https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth", 122_905_743),
    "swinir_color_dn_noise50.pth": ("https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth", 122_905_743),

    # SwinIR models - Real-World SR
    "swinir_real_sr_x4_large.pth": ("https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth", 142_473_939),

    # Real-ESRGAN models
    "RealESRGAN_x4plus.pth": ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", 67_040_989),
    "RealESRGAN_x2plus.pth": ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth", 67_061_725),
    "realesr-general-x4v3.pth": ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth", 4_885_111),

    # BSRGAN models (KAIR)
    "BSRGAN.pth": ("https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth", 67_046_751),
    "BSRGANx2.pth": ("https://github.com/cszn/KAIR/releases/download/v1.0/BSRGANx2.pth", 66_894_171),

    # DnCNN models (KAIR) - Gaussian denoising
    "dncnn_color_blind.pth": ("https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth", 2_678_121),
    "dncnn_gray_blind.pth": ("https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_gray_blind.pth", 2_668_897),
    "dncnn_15.pth": ("https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_15.pth", 2_224_903),
    "dncnn_25.pth": ("https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_25.pth", 2_224_903),
    "dncnn_50.pth": ("https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_50.pth", 2_224_903),

    # FFDNet models (KAIR) - Fast flexible denoising
    "ffdnet_color.pth": ("https://github.com/cszn/KAIR/releases/download/v1.0/ffdnet_color.pth", 3_411_551),
    "ffdnet_gray.pth": ("https://github.com/cszn/KAIR/releases/download/v1.0/ffdnet_gray.pth", 1_945_077),
}

ALL_MODELS = {**{k: ("gdrive",) + v for k, v in GDRIVE_MODELS.items()},
              **{k: ("url",) + v for k, v in URL_MODELS.items()}}


def is_lfs_pointer(filepath):
    """Check if file is a Git LFS pointer (not actual content)."""
    if not os.path.exists(filepath) or os.path.getsize(filepath) >= 200:
        return False
    try:
        with open(filepath, "r") as f:
            return f.read(50).startswith("version https://git-lfs")
    except (OSError, UnicodeDecodeError):
        return False


def needs_download(filepath, expected_size):
    """Check if model is missing, an LFS pointer, or truncated."""
    if not os.path.exists(filepath) or is_lfs_pointer(filepath):
        return True
    return os.path.getsize(filepath) < expected_size * 0.8


def download_from_gdrive(file_id, dest_path, display_name=None):
    """Download file from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown

    print(f"Downloading {display_name or os.path.basename(dest_path)} from Google Drive...")
    result = gdown.download(f"https://drive.google.com/uc?id={file_id}", dest_path, quiet=False)
    if result is None or not os.path.exists(dest_path):
        raise RuntimeError("Google Drive download failed (file may be rate-limited; try again later)")


def download_from_url(url, dest_path, display_name=None):
    """Download file from a direct URL with a progress indicator."""
    import urllib.request

    print(f"Downloading {display_name or os.path.basename(dest_path)} from {url.split('/')[2]}...")

    def report_progress(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 // total_size)
            print(f"\r  {percent}% ({downloaded / 2**20:.1f}/{total_size / 2**20:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
    print()


def download_model(name, models_dir=None):
    """Download a single model by filename. Returns the final path."""
    if name not in ALL_MODELS:
        raise KeyError(f"Unknown model: {name}")
    source, ref, expected_size = ALL_MODELS[name]
    models_dir = models_dir or default_models_dir()
    os.makedirs(models_dir, exist_ok=True)
    dest = os.path.join(models_dir, name)
    tmp = dest + ".part"
    if os.path.exists(tmp):
        os.remove(tmp)
    try:
        if source == "gdrive":
            download_from_gdrive(ref, tmp, name)
        else:
            download_from_url(ref, tmp, name)
        if needs_download(tmp, expected_size):
            raise RuntimeError(f"Downloaded file is incomplete ({os.path.getsize(tmp)} bytes)")
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return dest


def find_existing(name, search_dirs):
    """Return the path of an already-downloaded, complete model, or None."""
    expected_size = ALL_MODELS[name][2] if name in ALL_MODELS else 0
    for d in search_dirs:
        path = os.path.join(d, name)
        if not needs_download(path, expected_size):
            return path
    return None


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    models_dir = default_models_dir()
    search_dirs = list(dict.fromkeys([models_dir, LEGACY_MODELS_DIR]))
    names = argv or list(ALL_MODELS)

    print("=" * 60)
    print("  Image Restoration Model Downloader (ComfyUI-NAFNet_plus)")
    print("=" * 60)
    print(f"\nModels directory: {models_dir}\n")

    failed = []
    for name in names:
        if name not in ALL_MODELS:
            print(f"  [SKIP] Unknown model: {name}")
            failed.append(name)
            continue
        existing = find_existing(name, search_dirs)
        if existing:
            print(f"  [✓] {name} (already present: {existing})")
            continue
        try:
            download_model(name, models_dir)
            print(f"  [OK] {name}")
        except Exception as e:  # keep going so one failure doesn't block the rest
            print(f"  [ERROR] Failed to download {name}: {e}")
            failed.append(name)

    print("\n" + "=" * 60)
    if failed:
        print(f"{len(failed)} model(s) failed: {', '.join(failed)}")
        return 1
    print("All requested models are available.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
