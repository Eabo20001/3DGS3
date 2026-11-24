# 3DGS³: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering

**Yibo Zhao\*, Fan Gao\*, Youcheng Cai†, Ligang Liu**<br>
University of Science and Technology of China<br>
\* Equal contribution. † Corresponding author.

Official implementation of **3DGS³**, a post-rendering framework that jointly performs super sampling and frame interpolation to accelerate high-resolution 3D Gaussian Splatting rendering.

[Code](https://github.com/Eabo20001/3DGS3) · [Installation](#1-installation) · [Data Preparation](#2-data-preparation) · [Training](#3-training) · [Rendering](#4-rendering-and-data-export) · [Results](#5-paper-results) · [Citation](#9-citation)

## Overview

3DGS³ reconstructs high-resolution, high-frame-rate sequences from low-resolution 3DGS outputs using two cooperating modules:

- **Gradient-Aware Super Sampling (GASS)** combines analytical image-space gradients with a geometry-temporal recurrent refinement network to recover spatial detail.
- **Lightweight Temporal Frame Interpolation (LTFI)** fuses temporal information and gradient features through a compact U-Net-like network to synthesize intermediate frames.

The framework uses depth, camera motion, and recurrent features to maintain temporal consistency. It complements acceleration methods that optimize Gaussian representations or the splatting pipeline.

The implementation includes a modified Gaussian rasterizer that produces image gradients, custom CUDA kernels for motion vectors and warping, and a joint PyTorch training pipeline.

## 1. Installation

### Clone the repository

```bash
git clone --recursive https://github.com/Eabo20001/3DGS3.git
cd 3DGS3
git submodule update --init --recursive
```

Run the following commands from the repository root. Building the extensions requires an NVIDIA GPU, a CUDA toolkit with `nvcc`, and a compatible C++ compiler. The example below targets Linux with CUDA 12.1; CPU-only and macOS environments cannot run these CUDA kernels.

### Create an environment and install dependencies

```bash
conda create -n 3dgs3 python=3.10 -y
conda activate 3dgs3

python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install setuptools wheel ninja
python -m pip install "numpy<2" -r requirements.txt \
    scipy pillow matplotlib pyyaml toml tensorboard
```

The PyTorch/CUDA combination above is an installation example, rather than a validated reproduction environment. Other combinations should use matching PyTorch and torchvision builds from the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/). Match the local CUDA toolkit to the CUDA version used by PyTorch before compiling extensions.

### Install the CUDA extensions

The motion-vector extension expects GLM headers under `mv_tools/mv_cuda/glm/glm/`. Populate that location using the GLM copy initialized with the rasterizer:

```bash
mkdir -p mv_tools/mv_cuda/glm
cp -R submodules/diff-gaussian-rasterization/third_party/glm/glm mv_tools/mv_cuda/glm/

python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation ./submodules/simple-knn
python -m pip install --no-build-isolation ./mv_tools/mv_cuda
python -m pip install --no-build-isolation ./mv_tools/fill_with_order_cuda
```

Use the rasterizer included in this repository: the joint networks require its additional depth and image-gradient outputs. The tetrahedral triangulation module is optional and is not required for this training pipeline.

Check CUDA availability and extension imports:

```bash
python -c "import torch; import diff_gaussian_rasterization, simple_knn._C, mv_cuda, fill_with_order; print('CUDA available:', torch.cuda.is_available())"
```

## 2. Data Preparation

The default joint training uses the following fixed split:

| Role | Scenes | Scene configuration files |
| --- | --- | --- |
| Training | bicycle, kitchen, stump, truck | `config/exp-configs/{bicycle,kitchen,stump,truck}.toml` |
| Validation | garden | `config/exp-configs/garden.toml` |

Bicycle, kitchen, stump, and garden belong to Mip-NeRF 360; truck belongs to Tanks and Temples. Prepare undistorted images and COLMAP reconstructions in the same format used by Gaussian Splatting. The COLMAP loader supports `PINHOLE` and `SIMPLE_PINHOLE` cameras.

```text
datasets/
├── bicycle/
│   ├── images/
│   │   └── ...
│   └── sparse/
│       └── 0/
│           ├── cameras.bin
│           ├── images.bin
│           └── points3D.bin
├── kitchen/
├── stump/
├── truck/
└── garden/
```

The remaining scenes follow the same structure. COLMAP text files (`cameras.txt`, `images.txt`, and `points3D.txt`) are also supported. The loader creates `points3D.ply` from the sparse points if necessary.

Each scene TOML specifies its `dataset.source_path`, `dataset.model_path`, and rendering settings. The supplied paths use `./datasets/<scene>` and `./output/<scene>`. Adjust them if your data or models are stored elsewhere.

Joint training loads a pretrained Gaussian scene and `cameras.json` for **all five scenes**, including Garden. Prepare these models using the first training stage below before starting joint training.

## 3. Training

### Stage 1: Train the Gaussian scenes

Train each scene with this repository's rasterizer and Gaussian model implementation:

```bash
for scene in bicycle kitchen stump truck garden; do
    python train.py \
        -s "./datasets/${scene}" \
        -m "./output/${scene}" \
        -r 1024 \
        --iterations 30000
done
```

Here `-r 1024` sets the training image width to 1024 pixels while preserving the aspect ratio. If you use an image directory other than `images`, pass `-i <image_directory>` when training the Gaussian scene.

The output for each scene includes:

```text
output/<scene>/
├── cameras.json
├── cfg_args
├── input.ply
└── point_cloud/
    └── iteration_30000/
        └── point_cloud.ply
```

The joint loader selects the latest saved point-cloud iteration for each scene. Using checkpoints produced by this repository also preserves the Gaussian properties expected by its modified renderer.

### Stage 2: Train GASS and LTFI jointly

```bash
python UnionTrain.py --config config/union_model.yaml
```

The default training settings are:

| Setting | Default |
| --- | --- |
| Training scenes, in traversal order | bicycle → kitchen → stump → truck |
| Validation scene | garden |
| Optimizer | Adam, learning rate `1e-3` |
| Training duration | 150 epochs |
| Scheduler | CosineAnnealingWarmRestarts |
| Super sampling | 4× per spatial dimension |
| Checkpoint interval | Every 3 epochs, starting at epoch 0, plus the final checkpoint |
| Checkpoint root | `./checkpoint/union_model/` |

Each epoch visits every camera in each of the four training scenes, generating short camera trajectories for training. The scenes share one joint network and optimizer; the scheduler advances once after all four scenes have been processed. Gaussian scene parameters remain frozen while the two post-rendering networks are trained. Training renders high-resolution supervision on the fly, so its runtime does not represent the accelerated inference pipeline.

Validation runs when a checkpoint is saved. The current implementation uses a short generated trajectory from the first Garden camera to save SR/FI comparison images and PSNR values; it does not perform a full benchmark over all Garden views. Garden is excluded from joint-network optimization. The first run also downloads torchvision's pretrained VGG16 weights for the perceptual loss unless they are already cached.

Configuration is split between:

- **`config/union_model.yaml`**: optimizer, scheduler, losses, epoch count, and checkpoint settings.
- **`config/exp-configs/*.toml`**: scene model paths and rendering parameters consumed by the joint data loader. The legacy `dataset` and `pipeline` sections in the YAML do not select the joint-training scenes.
- **`DataLoader` in `UnionTrain.py`**: fixed training and validation scene lists. Adding another TOML does not automatically add a training scene; update the list explicitly.

The current joint network and training sampler use 4× super sampling, with 512 × 512 low-resolution samples and 2048 × 2048 high-resolution targets. Changing a scene's `sr.scale` alone does not configure a different network scale. The current camera generator does not use `sr.predown` to determine these training resolutions.

Saved results are organized as:

```text
checkpoint/union_model/
└── epoch_000150/
    ├── checkpoint.pth
    ├── FI_psnr<value>.png
    └── SR_psnr<value>.png
```

Each checkpoint contains the epoch, joint model state, and optimizer state. The FI image compares the interpolated result, fusion prediction, blending weight, and rendered target; the SR image compares the super-sampled result and rendered target. The training CLI currently starts a fresh run and does not expose a resume flag.

**Paper setting:** the paper describes training the post-rendering model on Garden. This release defaults to training on bicycle, kitchen, stump, and truck while holding out Garden for validation. The results below refer to the paper's experimental setting.

## 4. Rendering and Data Export

`render.py` exports rendered sequences from Gaussian scene models. It accepts either a single scene model directory or a parent directory containing multiple scene model directories:

```bash
# Export one scene using its latest saved Gaussian model.
python render.py -m ./output/garden --iteration -1 --cam_cut 20

# Export all scene models under ./output.
python render.py -m ./output --iteration -1 --cam_cut 20
```

Each scene must contain `cameras.json` and a saved point cloud. `--cam_cut` limits the number of base cameras; each base camera generates a five-frame translation sequence. The default `--iteration` is 30000; use `-1` to select the latest saved iteration.

The script currently uses fixed LR/HR resolutions of **1280 × 720** and **2560 × 1440**. These are export settings in `process_single_scene`, separate from the joint network's 4× training setup.

```text
output/<scene>/MobFGSR_Inputs/<sequence_id>/
├── HR_Inputs/
│   ├── view/
│   ├── depth/
│   ├── motion_vectors_x/
│   └── motion_vectors_y/
└── LR_Inputs/
    ├── view/
    ├── depth/
    ├── motion_vectors_x/
    └── motion_vectors_y/
```

RGB images are saved as PNGs. Depth and motion-vector components are encoded into RGBA PNGs by the export script. Even-numbered frames carry motion relative to the previous even-numbered frame; the first frame and odd-numbered frames receive zero motion.

This exporter renders Gaussian outputs and does not load the joint GASS/LTFI checkpoint. The repository currently does not include a standalone joint-network inference CLI, TensorRT export/deployment entry point, or a full paper benchmark runner.

## 5. Paper Results

The paper evaluates 4× super sampling with frame interpolation on Mip-NeRF 360, Deep Blending, and Tanks and Temples. The following results use an **NVIDIA RTX 3090**, **TensorRT inference**, and a parallel super-sampling/frame-interpolation pipeline, averaged over 300-frame sequences.

| Dataset | Output resolution | PSNR ↑ | SSIM ↑ | 3DGS³ FPS ↑ | HR 3DGS FPS ↑ |
| --- | --- | --- | --- | --- | --- |
| Mip-NeRF 360 | 4096 × 2160 | 49.22 | 0.995 | 96.47 | 17.35 |
| Deep Blending | 2560 × 1440 | 40.85 | 0.983 | 226.50 | 41.57 |
| Tanks and Temples | 2560 × 1440 | 37.64 | 0.978 | 246.35 | 50.40 |

For this table, PSNR and SSIM are measured against **high-resolution 3DGS renderings**, rather than captured photographs. They measure post-rendering reconstruction quality. The paper also reports separate comparisons using real-image ground truth and large-scale scenes containing over 10 million Gaussians.

These are paper-reported results. The TensorRT deployment path is not included in this release, and the Python training and data-export scripts do not reproduce these FPS measurements directly.

## 6. Viewer

The repository includes an SIBR Gaussian viewer for inspecting Gaussian scene models. See the [bundled SIBR documentation](SIBR_viewers/README.md) for build information. This viewer displays Gaussian scenes; it does not run the joint GASS/LTFI networks.

## 7. Acknowledgements

This project builds on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [RaDe-GS](https://github.com/BaowenZ/RaDe-GS), using RaDe-GS for depth and normal rendering.

The code also incorporates components from [Mip-Splatting](https://github.com/autonomousvision/mip-splatting), [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting), and [Gaussian Opacity Fields](https://github.com/autonomousvision/gaussian-opacity-fields), including filtering, geometry regularization, appearance modeling, and visualization utilities. We thank their authors for making these implementations available.

## 8. License

See [LICENSE.md](LICENSE.md) for the Gaussian-Splatting license distributed with this repository, including its non-commercial research and evaluation terms. Third-party components retain their respective licenses and notices.

## 9. Citation

If you use this work in your research, please cite:

```bibtex
@InProceedings{10.1007/978-3-032-37314-4_30,
author="Zhao, Yibo
and Gao, Fan
and Cai, Youcheng
and Liu, Ligang",
editor="Favaro, Paolo
and Kukelova, Zuzana
and Maki, Atsuto
and Rohrbach, Anna
and Schindler, Konrad
and Tombari, Federico",
title="3DGS{\$}{\$}^3{\$}{\$}: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering",
booktitle="Computer Vision -- ECCV 2026",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="543--561",
isbn="978-3-032-37314-4"
}


```
