# Degradation-aware Infrared Image Super-Resolution

**Jinyuan Liu, Bin Zhang, Liming Zhang, Xinwei Xue, Zhiying Jiang, and Xingyuan Li**

Official implementation of **DIRSR**, accepted by *IEEE Transactions on Image Processing*.

DIRSR is a degradation-aware framework for infrared image super-resolution. It combines an infrared degradation estimator (IDE), degradation-aware semantic guidance, and a physics-informed region-aware training objective to restore structural and thermal details.

<p align="center">
  <img src="assets/overview.png" alt="Overview of the DIRSR framework and infrared degradation estimator" width="100%">
</p>

## Quick Start

### 1. Installation

The code is configured for Linux with an NVIDIA GPU and CUDA. Create the environment from the supplied configuration:

```bash
git clone https://github.com/binbin1021/DIRSR.git
cd DIRSR
conda env create -f environment.yml
conda activate dirsr
```

The environment specifies Python 3.10, PyTorch 2.1, and the project dependencies. Use a compatible NVIDIA driver and CUDA setup. See `requirements.txt` for the pinned Python packages.

### 2. Prepare model files

The repository does not include model weights. Place the files in the following locations, or update their paths in `configs/inference.json`:

```text
DIRSR/
├── bert/                              # BERT tokenizer files, included in this repository
└── weights/
    ├── sr/
    │   └── model_38501.pkl            # DIRSR SR checkpoint
    ├── ide/
    │   └── best_ema.pth               # learned-offset IDE checkpoint
    ├── dape/
    │   └── DAPE.pth                   # SeeSR DAPE adapter weights
    └── base/
        ├── ram_swin_large_14m.pth     # RAM base weights
        └── sd-turbo/                  # complete SD-Turbo model directory
```

| Component | Source |
| --- | --- |
| SD-Turbo backbone | [Stability AI SD-Turbo on Hugging Face](https://huggingface.co/stabilityai/sd-turbo) |
| RAM base weights | [Recognize Anything Model on Hugging Face](https://huggingface.co/xinyu1205/recognize_anything_model/blob/main/ram_swin_large_14m.pth) |
| DAPE adapter weights | [SeeSR DAPE on Hugging Face](https://huggingface.co/CSWRY/SeeSR/blob/main/DAPE.pth) |
| DIRSR SR and IDE weights | Not included in this repository; supply the matching `model_38501.pkl` and `best_ema.pth` files. |

Download the **complete** SD-Turbo model, including the tokenizer, scheduler, text encoder, U-Net, and VAE subdirectories. For example:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='stabilityai/sd-turbo', local_dir='weights/base/sd-turbo', local_dir_use_symlinks=False)"
```

DAPE is loaded on top of the RAM base model; the adapter weights alone are not sufficient. The default DIRSR and IDE checkpoints are a matching pair.

### 3. Inference

Put low-resolution infrared images in `data/test/LR/`, or specify another input directory with `--lr-dir`. The inference script reads files with lowercase `.png`, `.jpg`, or `.jpeg` extensions in a single directory and saves ×4 results as PNG files.

```bash
# Check paths and missing files before running
python scripts/infer.py --config configs/inference.json --dry-run

# Run inference on GPU 0
python scripts/infer.py --config configs/inference.json --gpu 0
```

Results are saved to `outputs/SR/` by default. Use `--lr-dir` and `--output-dir` to change the input and output locations. To prevent accidental overwrites, use an output directory without existing files that have the same output names. The default positive and negative prompts are specified in `configs/inference.json`; DAPE tags are generated from each input image and appended to the positive prompt.

## Data Preparation

For continued SR training, prepare paired HR images, LR images, and foreground masks for both the training and validation splits:

```text
data/
├── train/
│   ├── HR/
│   ├── LR/
│   └── mask/
├── val/
│   ├── HR/
│   ├── LR/
│   └── mask/
└── test/
    ├── LR/
    └── HR/                            # paired ground truth, for evaluation
```

Paired HR, LR, and mask files should have matching relative filenames, including extensions. For ×4 training, HR and mask dimensions should be four times the LR dimensions. Set the dataset paths in `configs/train.json` if using a different layout. The mask-generation code is not included in this repository.

To synthesize LR images with the included infrared degradation pipeline:

```bash
python data_tools/make_lr.py --input-dir data/train/HR --output-dir data/train/LR --seed 42
```

The degradation parameters are configured in `data_tools/degradation_default.json`. The included pipeline applies blur, downsampling, and noise in that order. For HM-TIR dataset information, see the official [HM-TIR repository](https://github.com/Zihang-Chen/HM-TIR).

## Training

### Continue DIRSR training

`scripts/train.py` continues training from an existing DIRSR SR checkpoint, with IDE and DAPE frozen. It is not a training-from-scratch entry point. The default configuration starts from `model_38501.pkl` and runs another 10,000 steps.

```bash
python scripts/train.py --config configs/train.json --dry-run
python scripts/train.py --config configs/train.json --gpu 0
```

Configure the checkpoint and data paths in `configs/train.json`, or override them using the script's command-line options. Run `python scripts/train.py --help` for the available options. Choose a new output directory for a new run; use `--resume` with a complete saved training state to resume an existing run.

### Train the infrared degradation estimator

The included learned-offset IDE training code follows the degradation-degree learning approach of [MM-RealSR](https://github.com/TencentARC/MM-RealSR), using its degradation synthesis and ranking/anchor objectives. This training entry starts from a **compatible legacy IDE checkpoint**, configured as `warm_start` in `configs/ide.json`; the original MM-RealSR checkpoint is not a drop-in replacement for this custom architecture.

Prepare `outputs/ide_training/data_manifest/train.txt` and `val.txt`, with one absolute HR image path per line, and place the compatible warm-start checkpoint at `weights/ide/warm_start.pth` or update the configuration. Then run:

```bash
python training/ide/train_ide.py --config configs/ide.json --device cuda:0
```

`training/ide/prepare_manifest.py` provides a split and overlap-checking utility for the naming conventions used in this project; see its `--help` output for the required dataset paths and counts.

### DAPE training

DAPE is based on [SeeSR](https://github.com/cswry/SeeSR). For degradation-aware prompt extractor training, see SeeSR's [DAPE training instructions](https://github.com/cswry/SeeSR#step3-training-for-dape) and [training configuration](https://github.com/cswry/SeeSR/blob/main/basicsr/options/dape.yaml). The standalone DAPE adaptation script is not included in this repository.

## Evaluation

Evaluate restored images against paired HR images with matching filenames, including extensions, and dimensions. Since inference saves PNG files, use matching PNG ground-truth filenames when evaluating results generated from JPG or JPEG inputs.

```bash
python scripts/evaluate.py --sr outputs/SR --hr data/test/HR --device cuda:0
```

The evaluator reports PSNR and SSIM on the Y channel, and LPIPS and DISTS on RGB images. See `python scripts/evaluate.py --help` for metric and output options.

## Citation

If you find this work useful, please cite: Jinyuan Liu, Bin Zhang, Liming Zhang, Xinwei Xue, Zhiying Jiang, and Xingyuan Li, “Degradation-aware Infrared Image Super-Resolution,” *IEEE Transactions on Image Processing*, accepted.

## Acknowledgements

This project builds on [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo), [S3Diff](https://github.com/ArcticHare105/S3Diff), [SeeSR](https://github.com/cswry/SeeSR), [MM-RealSR](https://github.com/TencentARC/MM-RealSR), [RAM](https://github.com/xinyu1205/recognize-anything), and [BasicSR](https://github.com/XPixelGroup/BasicSR). We thank the authors for making their work available.
