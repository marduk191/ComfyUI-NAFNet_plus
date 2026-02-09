# ComfyUI-NAFNet

ComfyUI custom nodes for [NAFNet](https://github.com/megvii-research/NAFNet) - Simple Baselines for Image Restoration.

This node pack provides state-of-the-art image denoising, deblurring, and stereo super-resolution capabilities using NAFNet and NAFSSR models from MEGVII Research.

## Features

- **Image Denoising** - Remove noise from images using SIDD-trained models
- **Image Deblurring** - Remove motion blur using GoPro/REDS-trained models
- **Stereo Super-Resolution** - Upscale stereo image pairs (2x/4x) using NAFSSR
- **Tiled Processing** - Handle large images with limited VRAM
- **Model Caching** - Efficient memory usage with automatic model caching

## Nodes

| Node | Description |
|------|-------------|
| **NAFNet Load Model** | Load any NAFNet/NAFSSR model for use with NAFNet Restore |
| **NAFNet Restore** | Generic restoration with optional tiled processing |
| **NAFNet Denoise (SIDD)** | Quick denoise using SIDD-trained models |
| **NAFNet Deblur (GoPro)** | Quick deblur using GoPro-trained models |
| **NAFSSR Stereo Super-Resolution** | Upscale stereo image pairs with cross-attention |

## Installation

Manual Installation:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/marduk191/ComfyUI-NAFNet.git
cd ComfyUI-NAFNet
pip install -r requirements.txt
```

**Note:** Models (1.3 GB) are included via Git LFS. If you don't have Git LFS installed, run the fallback downloader:
```bash
python download_models.py
```

## Models

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

### Denoise an Image

1. Add **Load Image** node
2. Add **NAFNet Denoise (SIDD)** node
3. Connect image output to NAFNet input
4. Choose `width64` for best quality or `width32` for speed

### Deblur an Image

1. Add **Load Image** node
2. Add **NAFNet Deblur (GoPro)** node
3. Connect and run

### Process Large Images (Tiled)

1. Add **Load Image** node
2. Add **NAFNet Load Model** node (select any model)
3. Add **NAFNet Restore** node
4. Set `tile_size` to 512 and `tile_overlap` to 64
5. Connect and run

### Stereo Super-Resolution

1. Add two **Load Image** nodes (left and right views)
2. Add **NAFSSR Stereo Super-Resolution** node
3. Connect left/right images to corresponding inputs
4. Choose 2x or 4x upscaling
5. Both outputs are upscaled with cross-view attention

## Sample Workflows

Sample workflow JSON files are included in the `workflows/` folder:

- `nafnet_denoise.json` - Basic denoising
- `nafnet_deblur.json` - Basic deblurring
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
```

## License

This project is released under the MIT License. The NAFNet model weights are subject to their original license from MEGVII Research.
