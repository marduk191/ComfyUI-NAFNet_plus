"""
Download pretrained models for image restoration.

This script downloads models from Google Drive and GitHub releases.
Run this script after installation to download all required model weights.

Supported Models:
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
import sys
import subprocess

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Google Drive file IDs
# Format: filename -> (file_id, expected_size_bytes)
GDRIVE_MODELS = {
    # NAFNet models
    "NAFNet-SIDD-width32.pth": ("1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ", 116_800_000),
    "NAFNet-SIDD-width64.pth": ("14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR", 464_400_000),
    "NAFNet-GoPro-width32.pth": ("1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj", 68_700_000),
    "NAFNet-GoPro-width64.pth": ("1S0PVRbyTakYY9a82kujgZLbMihfNBLfC", 271_800_000),
    "NAFNet-REDS-width64.pth": ("14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X", 271_800_000),
    "NAFSSR-L_2x.pth": ("1SZ6bQVYTVS_AXedBEr-_mBCC-qGYHLmf", 96_500_000),
    "NAFSSR-L_4x.pth": ("1TIdQhPtBrZb2wrBdAp9l8NHINLeExOwb", 97_500_000),
    # Restormer
    "restormer_real_denoising.pth": ("1FF_4NTboTWQ7sHCq4xhyLZsSl0U0JfjH", 110_000_000),
    # HINet models
    "HINet-SIDD-1x.pth": ("1CU5z-M90Jc-TAcVpEaFjDCYA09fkubGi", 88_000_000),
    "HINet-SIDD-0.5x.pth": ("1Y5YJQVNL0weifE--5us344bLwzBNS_sU", 22_000_000),
    "HINet-GoPro.pth": ("1dw8PKVkLfISzNtUu3gqGh83NBO83ZQ5n", 88_000_000),
    # MIRNet-v2 (real denoising folder)
    "MIRNet_v2_real_denoising.pth": ("1R2V80TFdTBLnwERFNBuq4eThKZGzv3bT", 22_000_000),
}

# Direct URL downloads (GitHub releases, etc.)
# Format: filename -> (url, expected_size_bytes)
URL_MODELS = {
    # SCUNet models (KAIR)
    "scunet_color_real_gan.pth": ("https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_gan.pth", 75_500_000),
    "scunet_color_real_psnr.pth": ("https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth", 75_500_000),

    # SwinIR models - Color Denoising
    "swinir_color_dn_noise15.pth": ("https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth", 12_000_000),
    "swinir_color_dn_noise25.pth": ("https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth", 12_000_000),
    "swinir_color_dn_noise50.pth": ("https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth", 12_000_000),

    # SwinIR models - Real-World SR (good for general enhancement)
    "swinir_real_sr_x4_large.pth": ("https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth", 137_000_000),

    # Real-ESRGAN models
    "RealESRGAN_x4plus.pth": ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", 67_000_000),
    "RealESRGAN_x2plus.pth": ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth", 67_000_000),
    "realesr-general-x4v3.pth": ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth", 67_000_000),

    # BSRGAN models (KAIR)
    "BSRGAN.pth": ("https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth", 67_000_000),
    "BSRGANx2.pth": ("https://github.com/cszn/KAIR/releases/download/v1.0/BSRGANx2.pth", 67_000_000),

    # DnCNN models (KAIR) - Gaussian denoising
    "dncnn_color_blind.pth": ("https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth", 2_500_000),
    "dncnn_gray_blind.pth": ("https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_gray_blind.pth", 700_000),
    "dncnn_15.pth": ("https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_15.pth", 700_000),
    "dncnn_25.pth": ("https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_25.pth", 700_000),
    "dncnn_50.pth": ("https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_50.pth", 700_000),

    # FFDNet models (KAIR) - Fast flexible denoising
    "ffdnet_color.pth": ("https://github.com/cszn/KAIR/releases/download/v1.0/ffdnet_color.pth", 3_500_000),
    "ffdnet_gray.pth": ("https://github.com/cszn/KAIR/releases/download/v1.0/ffdnet_gray.pth", 900_000),

    # DRUNet models (KAIR) - Deep residual U-Net denoising
    "drunet_color.pth": ("https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth", 130_000_000),
    "drunet_gray.pth": ("https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth", 33_000_000),
}


def is_lfs_pointer(filepath):
    """Check if file is a Git LFS pointer (not actual content)."""
    if not os.path.exists(filepath):
        return False
    size = os.path.getsize(filepath)
    # LFS pointers are small text files (~130 bytes)
    if size < 200:
        try:
            with open(filepath, 'r') as f:
                content = f.read(50)
                return content.startswith('version https://git-lfs')
        except:
            pass
    return False


def needs_download(filepath, expected_size):
    """Check if model needs to be downloaded."""
    if not os.path.exists(filepath):
        return True
    if is_lfs_pointer(filepath):
        return True
    # Check if file is roughly the expected size (within 20%)
    actual_size = os.path.getsize(filepath)
    if actual_size < expected_size * 0.8:
        return True
    return False


def download_from_gdrive(file_id, dest_path):
    """Download file from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {os.path.basename(dest_path)} from Google Drive...")
    gdown.download(url, dest_path, quiet=False)


def download_from_url(url, dest_path):
    """Download file from direct URL with progress."""
    import urllib.request

    filename = os.path.basename(dest_path)
    source = url.split('/')[2]
    print(f"Downloading {filename} from {source}...")

    try:
        # Simple progress indicator
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 // total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)

        urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
        print()  # New line after progress
    except Exception as e:
        print(f"\n[ERROR] Failed to download: {e}")
        raise


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 60)
    print("  Image Restoration Model Downloader")
    print("  NAFNet+ / SCUNet / Restormer / SwinIR / Real-ESRGAN")
    print("  BSRGAN / HINet / MIRNet-v2 / DnCNN / FFDNet")
    print("=" * 60)
    print(f"\nModels directory: {MODELS_DIR}\n")

    downloads_needed = 0
    total_models = len(GDRIVE_MODELS) + len(URL_MODELS)

    # Check Google Drive models
    for name, (file_id, expected_size) in GDRIVE_MODELS.items():
        dest = os.path.join(MODELS_DIR, name)
        if needs_download(dest, expected_size):
            downloads_needed += 1

    # Check URL models
    for name, (url, expected_size) in URL_MODELS.items():
        dest = os.path.join(MODELS_DIR, name)
        if needs_download(dest, expected_size):
            downloads_needed += 1

    if downloads_needed == 0:
        print("All models are already downloaded!")
    else:
        print(f"Need to download {downloads_needed} of {total_models} model(s)...\n")

        # Download Google Drive models
        for name, (file_id, expected_size) in GDRIVE_MODELS.items():
            dest = os.path.join(MODELS_DIR, name)
            if needs_download(dest, expected_size):
                if is_lfs_pointer(dest):
                    print(f"[LFS POINTER] {name} - downloading actual file...")
                    os.remove(dest)
                try:
                    download_from_gdrive(file_id, dest)
                    print(f"  [OK] {name}")
                except Exception as e:
                    print(f"  [ERROR] Failed to download {name}: {e}")

        # Download URL models
        for name, (url, expected_size) in URL_MODELS.items():
            dest = os.path.join(MODELS_DIR, name)
            if needs_download(dest, expected_size):
                if is_lfs_pointer(dest):
                    print(f"[LFS POINTER] {name} - downloading actual file...")
                    os.remove(dest)
                try:
                    download_from_url(url, dest)
                    print(f"  [OK] {name}")
                except Exception as e:
                    print(f"  [ERROR] Failed to download {name}: {e}")

    print("\n" + "=" * 60)
    print("Available models:")
    print("=" * 60)

    if os.path.exists(MODELS_DIR):
        for f in sorted(os.listdir(MODELS_DIR)):
            if f.endswith('.pth'):
                filepath = os.path.join(MODELS_DIR, f)
                if is_lfs_pointer(filepath):
                    print(f"  [ ] {f} (LFS pointer - needs download)")
                else:
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    print(f"  [✓] {f} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
