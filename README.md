# ComfyUI-NAFNet

ComfyUI custom nodes for [NAFNet](https://github.com/megvii-research/NAFNet) - Simple Baselines for Image Restoration.

This node pack provides state-of-the-art image denoising, deblurring, and stereo super-resolution capabilities using NAFNet and NAFSSR models from MEGVII Research.

![NAFNet Banner](https://raw.githubusercontent.com/megvii-research/NAFNet/main/figures/NAFNet-title.png)

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

### Option 1: ComfyUI Manager (Recommended)

Search for "NAFNet" in ComfyUI Manager and install.

### Option 2: Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/ComfyUI-NAFNet.git
cd ComfyUI-NAFNet
pip install -r requirements.txt
python download_models.py
```

### Option 3: Download Models Manually

Download models from the [NAFNet releases](https://github.com/megvii-research/NAFNet#results-and-pre-trained-models) and place them in the `models/` folder.

## Models

| Model | Task | Size | Dataset |
|-------|------|------|---------|
| NAFNet-SIDD-width32.pth | Denoising | 111 MB | SIDD |
| NAFNet-SIDD-width64.pth | Denoising | 443 MB | SIDD |
| NAFNet-GoPro-width32.pth | Deblurring | 66 MB | GoPro |
| NAFNet-GoPro-width64.pth | Deblurring | 259 MB | GoPro |
| NAFNet-REDS-width64.pth | Video Deblurring | 259 MB | REDS |
| NAFSSR-L_2x.pth | Stereo SR 2x | 92 MB | Flickr1024 |
| NAFSSR-L_4x.pth | Stereo SR 4x | 92 MB | Flickr1024 |

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
- gdown (for model downloads)

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
