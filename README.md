# ComfyUI-NAFNet_plus

Comprehensive ComfyUI custom nodes for state-of-the-art image restoration - **especially good for astrophotography and telescope images**.

## Supported Models

| Model | Type | Best For |
|-------|------|----------|
| **SCUNet** | Blind Denoising | Astrophotography, unknown noise |
| **Restormer** | Transformer Denoising | High-quality restoration |
| **SwinIR** | Swin Transformer | Denoising & Super-Resolution |
| **Real-ESRGAN** | Upscaling + Enhancement | General image enhancement |
| **BSRGAN** | Blind Super-Resolution | Unknown degradation |
| **MIRNet-v2** | Multi-scale Denoising | Diverse noise types |
| **HINet** | Instance Norm Denoising | Fast denoising |
| **NAFNet** | Domain-Specific | Smartphone noise, GoPro blur |
| **DnCNN** | Classic CNN | Fast Gaussian denoising |
| **FFDNet** | Flexible Denoising | Adjustable noise level |

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/marduk191/ComfyUI-NAFNet_plus.git
cd ComfyUI-NAFNet_plus
pip install -r requirements.txt
python download_models.py
```

**Important:** Models are NOT included in the repo. Run `download_models.py` to download all pretrained weights (~2GB total).

## Nodes

### Recommended for Astrophotography
- **SCUNet Denoise (Blind/Real)** - Best for unknown noise types
- **Restormer Denoise (Real)** - State-of-the-art transformer
- **SwinIR Denoise** - Handles various noise levels (15/25/50)

### Super-Resolution / Upscaling
- **SwinIR Super-Resolution (4x)** - Real-world 4x upscaling
- **Real-ESRGAN Upscale** - Practical 2x/4x enhancement
- **BSRGAN Upscale (Blind SR)** - Handles unknown degradation

### Fast / Lightweight Denoising
- **DnCNN Denoise** - Classic fast denoiser
- **FFDNet Denoise (Adjustable)** - Flexible noise level control
- **HINet Denoise (SIDD)** - Half Instance Normalization

### Domain-Specific
- **NAFNet Denoise (SIDD)** - Smartphone camera noise
- **NAFNet Deblur (GoPro)** - GoPro motion blur only

## Model Downloads

All models are downloaded via `download_models.py` from:
- GitHub Releases (KAIR, SwinIR, Real-ESRGAN)
- Google Drive (NAFNet, Restormer, HINet, MIRNet-v2)

Total download size: ~2GB

## Usage Tips

### For Astrophotography / Telescope Images
1. Try **SCUNet (real_gan)** first - good balance of sharpness
2. Try **Restormer** for highest quality
3. Use **SwinIR Denoise (noise level 50)** for very noisy images
4. Chain with **Real-ESRGAN** or **SwinIR SR** for upscaling

### Tiling for Large Images
All nodes support tiling for memory efficiency:
- `tile_size=0` - Auto-tile images >1 megapixel
- `tile_size=512` - Manual tile size (adjust based on VRAM)

### FFDNet Noise Level
The FFDNet node allows adjusting noise level (0-75):
- Lower values (5-15): Light denoising, preserves detail
- Medium values (25): Balanced
- Higher values (50+): Strong denoising

## Credits

- [NAFNet](https://github.com/megvii-research/NAFNet) - MEGVII Research
- [SCUNet](https://github.com/cszn/SCUNet) - Kai Zhang
- [Restormer](https://github.com/swz30/Restormer) - Syed Waqas Zamir
- [SwinIR](https://github.com/JingyunLiang/SwinIR) - Jingyun Liang
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - Xintao Wang
- [BSRGAN](https://github.com/cszn/BSRGAN) - Kai Zhang
- [HINet](https://github.com/megvii-model/HINet) - MEGVII
- [MIRNet-v2](https://github.com/swz30/MIRNetv2) - Syed Waqas Zamir
- [KAIR](https://github.com/cszn/KAIR) - Kai Zhang (DnCNN, FFDNet)

## License

MIT License. Model weights are subject to their original licenses.
