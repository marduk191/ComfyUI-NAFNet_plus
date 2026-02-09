"""
Download NAFNet pretrained models.
https://github.com/megvii-research/NAFNet

This script downloads models from Google Drive if Git LFS models are missing or corrupted.
Models should normally be included via Git LFS, but this provides a fallback.
"""
import os
import sys
import subprocess

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Google Drive file IDs from NAFNet README
# Format: filename -> (file_id, expected_size_bytes)
MODELS = {
    "NAFNet-SIDD-width32.pth": ("1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ", 116_800_000),
    "NAFNet-SIDD-width64.pth": ("14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR", 464_400_000),
    "NAFNet-GoPro-width32.pth": ("1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj", 68_700_000),
    "NAFNet-GoPro-width64.pth": ("1S0PVRbyTakYY9a82kujgZLbMihfNBLfC", 271_800_000),
    "NAFNet-REDS-width64.pth": ("14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X", 271_800_000),
    "NAFSSR-L_2x.pth": ("1SZ6bQVYTVS_AXedBEr-_mBCC-qGYHLmf", 96_500_000),
    "NAFSSR-L_4x.pth": ("1TIdQhPtBrZb2wrBdAp9l8NHINLeExOwb", 97_500_000),
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
    # Check if file is roughly the expected size (within 10%)
    actual_size = os.path.getsize(filepath)
    if actual_size < expected_size * 0.9:
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
    print(f"Downloading {os.path.basename(dest_path)}...")
    gdown.download(url, dest_path, quiet=False)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("NAFNet Model Downloader")
    print("=" * 50)
    print(f"Models directory: {MODELS_DIR}")
    print()

    downloads_needed = 0
    for name, (file_id, expected_size) in MODELS.items():
        dest = os.path.join(MODELS_DIR, name)
        if needs_download(dest, expected_size):
            downloads_needed += 1

    if downloads_needed == 0:
        print("All models are already downloaded!")
    else:
        print(f"Need to download {downloads_needed} model(s)...")
        print()

        for name, (file_id, expected_size) in MODELS.items():
            dest = os.path.join(MODELS_DIR, name)
            if needs_download(dest, expected_size):
                if is_lfs_pointer(dest):
                    print(f"[LFS POINTER] {name} - downloading actual file...")
                    os.remove(dest)
                try:
                    download_from_gdrive(file_id, dest)
                except Exception as e:
                    print(f"[ERROR] Failed to download {name}: {e}")
            else:
                print(f"[OK] {name}")

    print()
    print("Available models:")
    for f in sorted(os.listdir(MODELS_DIR)):
        if f.endswith('.pth'):
            filepath = os.path.join(MODELS_DIR, f)
            if is_lfs_pointer(filepath):
                print(f"  - {f} (LFS pointer - run this script to download)")
            else:
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  - {f} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
