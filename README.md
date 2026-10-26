# DRN-Segmentation

A detail-domain wavelet residual neck for remote sensing semantic segmentation with MambaVision and MMSegmentation.

## Setup

Install PyTorch and compatible MMCV, Mamba and causal-conv1d packages, then install the Python dependencies:

```bash
pip install -r requirements.txt
```

Download the [MambaVision-T pretrained checkpoint](https://huggingface.co/nvidia/MambaVision-T-1K) to `ckpts/mambavision_tiny_1k.pth.tar`.

## Data

The example configuration uses [LoveDA](https://github.com/Junjue-Wang/LoveDA). Place its files under:

```text
data/loveda/{train,val}/{Urban,Rural}/{images_png,masks_png}/
```

Update `configs/drn_loveda.py` for your dataset and training settings.

## Usage

```bash
python tools/train.py configs/drn_loveda.py --work-dir work_dirs/drn
python tools/test.py configs/drn_loveda.py path/to/checkpoint.pth
python tools/predict.py configs/drn_loveda.py path/to/checkpoint.pth --image path/to/image.png --output outputs/prediction.png
```

Predictions are saved as class-ID masks. The DRN implementation is in `tools/wsse.py`, with `streams='high_only'` selecting the detail branch.

## License

This project builds on [MambaVision](https://github.com/NVlabs/MambaVision) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). MambaVision-derived code is subject to the NVIDIA Source Code License-NC, which restricts use to non-commercial research or evaluation. See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
