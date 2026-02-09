"""
Download NAFNet pretrained models.
https://github.com/megvii-research/NAFNet

Models are hosted on Google Drive. This script uses gdown to download them.
"""
import os
import sys
import subprocess

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Google Drive file IDs from NAFNet README
# Format: filename -> Google Drive file ID
MODELS = {
    # Denoising (SIDD)
    "NAFNet-SIDD-width32.pth": "1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ",
    "NAFNet-SIDD-width64.pth": "14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR",
    # Deblurring (GoPro)
    "NAFNet-GoPro-width32.pth": "1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj",
    "NAFNet-GoPro-width64.pth": "1S0PVRbyTakYY9a82kujgZLbMihfNBLfC",
    # REDS (video deblurring with JPEG artifacts)
    "NAFNet-REDS-width64.pth": "14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X",
    # Stereo Image Super-Resolution (NAFSSR)
    "NAFSSR-L_2x.pth": "1SZ6bQVYTVS_AXedBEr-_mBCC-qGYHLmf",
    "NAFSSR-L_4x.pth": "1TIdQhPtBrZb2wrBdAp9l8NHINLeExOwb",
}


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
    print(f"Models will be saved to: {MODELS_DIR}")
    print()

    for name, file_id in MODELS.items():
        dest = os.path.join(MODELS_DIR, name)
        if os.path.exists(dest):
            print(f"[SKIP] {name} already exists")
        else:
            try:
                download_from_gdrive(file_id, dest)
            except Exception as e:
                print(f"[ERROR] Failed to download {name}: {e}")

    print()
    print("Download complete!")
    print()
    print("Available models:")
    for f in os.listdir(MODELS_DIR):
        if f.endswith('.pth'):
            size_mb = os.path.getsize(os.path.join(MODELS_DIR, f)) / (1024 * 1024)
            print(f"  - {f} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
