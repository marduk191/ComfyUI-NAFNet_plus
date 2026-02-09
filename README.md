# ComfyUI-NAFNet

ComfyUI custom nodes for state-of-the-art image restoration:

- [NAFNet](https://github.com/megvii-research/NAFNet) - Simple Baselines for Image Restoration (MEGVII Research)
- [SCUNet](https://github.com/cszn/SCUNet) - Practical Blind Real Image Denoising
- [Restormer](https://github.com/swz30/Restormer) - Efficient Transformer for High-Resolution Image Restoration

This node pack provides image denoising, deblurring, and stereo super-resolution capabilities. **SCUNet and Restormer are particularly good for astrophotography and telescope images** as they handle unknown/real-world noise types.

## Features

- **Blind Image Denoising** - SCUNet and Restormer handle unknown noise types (great for astrophotography!)
- **Domain-Specific Denoising** - NAFNet SIDD models for smartphone camera noise
- **Motion Deblurring** - NAFNet models for GoPro/video motion blur
- **Stereo Super-Resolution** - Upscale stereo image pairs (2x/4x) using NAFSSR
- **Auto-Tiled Processing** - Large images (>1MP) are automatically tiled for VRAM efficiency
- **Model Caching** - Efficient memory usage with automatic model caching

## Nodes

| Node | Description | Best For |
|------|-------------|----------|
| **SCUNet Denoise (Blind/Real)** | Blind denoising for unknown noise types | Astrophotography, telescope images, real-world photos |
| **Restormer Denoise (Real)** | State-of-the-art transformer denoising | Diverse noise types, high-quality restoration |
| **NAFNet Denoise (SIDD)** | Domain-specific smartphone denoising | Smartphone camera photos with sensor noise |
| **NAFNet Deblur (GoPro)** | Domain-specific motion deblurring | GoPro camera motion blur only |
| **NAFNet Load Model** | Load any NAFNet/NAFSSR model | Advanced users |
| **NAFNet Restore** | Generic restoration with tiled processing | Use with NAFNet Load Model |
| **NAFSSR Stereo Super-Resolution** | Upscale stereo image pairs | Stereo camera pairs, VR content |

## Installation

Manual Installation:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/marduk191/ComfyUI-NAFNet.git
cd ComfyUI-NAFNet
pip install -r requirements.txt
```

**Note:** Models (~1.5 GB total) are included via Git LFS. If you don't have Git LFS installed, run the fallback downloader:
```bash
python download_models.py
```

This will download all NAFNet, SCUNet, and Restormer models.

## Models

### Recommended for Astrophotography / Telescope Images

| Model | Task | Size | Description |
|-------|------|------|-------------|
| scunet_color_real_gan.pth | Blind Denoising | 72 MB | Perceptually better results, sharper |
| scunet_color_real_psnr.pth | Blind Denoising | 72 MB | Higher PSNR, smoother |
| restormer_real_denoising.pth | Real Denoising | 105 MB | State-of-the-art transformer |

### NAFNet Models (Domain-Specific)

| Model | Task | Size | Best For |
|-------|------|------|----------|
| NAFNet-SIDD-width32.pth | Denoising | 111 MB | Smartphone photos, general sensor noise |
| NAFNet-SIDD-width64.pth | Denoising | 443 MB | Smartphone photos, general sensor noise |
| NAFNet-GoPro-width32.pth | Deblurring | 66 MB | **GoPro motion blur only** |
| NAFNet-GoPro-width64.pth | Deblurring | 259 MB | **GoPro motion blur only** |
| NAFNet-REDS-width64.pth | Video Deblurring | 259 MB | **Video frames with compression artifacts** |
| NAFSSR-L_2x.pth | Stereo SR 2x | 92 MB | Stereo image pairs |
| NAFSSR-L_4x.pth | Stereo SR 4x | 92 MB | Stereo image pairs |

### Important: Model Domain Specificity

**NAFNet models are domain-specific** - they only work well on images similar to their training data:

- **SIDD (Denoising)**: Trained on smartphone camera noise. Works well on most photos with sensor noise.
- **GoPro (Deblurring)**: Trained specifically on GoPro camera motion blur. Will produce artifacts on other blur types.
- **REDS (Video Deblurring)**: Trained on video frames with specific blur/compression patterns. Not for photos.

If you apply the wrong model to your image, you'll get colorful noise/artifacts instead of restoration.

## Usage Examples

### Denoise Astrophotography / Telescope Images

1. Add **Load Image** node
2. Add **SCUNet Denoise (Blind/Real)** node
3. Choose `real_gan` for sharper results or `real_psnr` for smoother
4. Connect and run

Alternative: Use **Restormer Denoise (Real)** for transformer-based denoising.

### Denoise Smartphone Photos

1. Add **Load Image** node
2. Add **NAFNet Denoise (SIDD)** node
3. Choose `width64` for best quality or `width32` for speed

### Deblur GoPro Video Frames

1. Add **Load Image** node
2. Add **NAFNet Deblur (GoPro)** node
3. Connect and run

**Note:** GoPro models only work on GoPro-style motion blur. For other blur types, results may be poor.

### Process Large Images (Tiled)

All nodes automatically tile images larger than 1 megapixel. You can also manually set:
- `tile_size`: Size of each tile (default: 512, 0 = auto)
- `tile_overlap`: Overlap between tiles (default: 64)

### Stereo Super-Resolution

1. Add two **Load Image** nodes (left and right views)
2. Add **NAFSSR Stereo Super-Resolution** node
3. Connect left/right images to corresponding inputs
4. Choose 2x or 4x upscaling

## Sample Workflows

Sample workflow JSON files are included in the `workflows/` folder:

- `scunet_denoise.json` - SCUNet blind denoising (best for astrophotography)
- `restormer_denoise.json` - Restormer transformer denoising
- `nafnet_denoise.json` - NAFNet SIDD denoising (smartphone photos)
- `nafnet_deblur.json` - NAFNet GoPro deblurring
- `nafnet_loader_restore.json` - Using model loader
- `nafnet_tiled_restore.json` - Tiled processing for large images
- `nafssr_stereo_sr.json` - Stereo super-resolution

## Performance Tips

- Use `width32` models for faster processing with slightly lower quality
- Use `width64` models for best quality
- Enable tiled processing for images larger than 1024x1024
- NAFSSR works best with properly aligned stereo pairs

## Requirements

- ComfyUI
- PyTorch (CUDA recommended)
- gdown (only needed if Git LFS models fail to download)

## Credits

- [NAFNet](https://github.com/megvii-research/NAFNet) by MEGVII Research
- [NAFSSR](https://github.com/megvii-research/NAFNet/tree/main/docs/NAFSSR.md) - Stereo Image Super-Resolution Using NAFNet
- [SCUNet](https://github.com/cszn/SCUNet) by Kai Zhang (cszn)
- [Restormer](https://github.com/swz30/Restormer) by Syed Waqas Zamir

## Citation

```bibtex
@inproceedings{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}

@inproceedings{chu2022nafssr,
  title={NAFSSR: Stereo Image Super-Resolution Using NAFNet},
  author={Chu, Xiaojie and Chen, Liangyu and Yu, Wenqing},
  booktitle={CVPR Workshop},
  year={2022}
}

@inproceedings{zhang2023scunet,
  title={Practical Blind Image Denoising via Swin-Conv-UNet and Data Synthesis},
  author={Zhang, Kai and Li, Yawei and Liang, Jingyun and Cao, Jiezhang and Zhang, Yulun and Tang, Hao and Timofte, Radu and Van Gool, Luc},
  booktitle={Machine Intelligence Research},
  year={2023}
}

@inproceedings{zamir2022restormer,
  title={Restormer: Efficient Transformer for High-Resolution Image Restoration},
  author={Zamir, Syed Waqas and Arora, Aditya and Khan, Salman and Hayat, Munawar and Khan, Fahad Shahbaz and Yang, Ming-Hsuan},
  booktitle={CVPR},
  year={2022}
}
```

## License

This project is released under the MIT License. The NAFNet model weights are subject to their original license from MEGVII Research.
