# ACT-R: Adaptive Camera Trajectories for Single View 3D Reconstruction

A research implementation for generating adaptive camera trajectories to improve single-view 3D reconstruction quality through optimized viewpoint selection.

## Overview

ACT-R addresses the challenge of selecting optimal camera viewpoints for 3D reconstruction by computing semantic difference maps and planning adaptive trajectories that maximize object visibility. The system integrates with modern 3D reconstruction methods including NeUS and InstantMesh.

## Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended), this repository has been tested under NVIDIA RTX4090 with CUDA 12.1
- Conda package manager

## Installation

1. Clone the repository:
```bash
git clone --recursive https://github.com/mingrui-zhao/ACT-R.git
cd ACT-R
```

2. Create and activate the conda environment:
```bash
conda create -n actr python=3.10
conda activate actr
pip install -r requirements.txt
```

## Dataset Setup
### Data Preprocessing 

(1) Camera pose estimation: Refer to [the code here](https://1drv.ms/f/c/a0004126ab48d040/EkDQSKsmQQAggKD9rAEAAAABf7wWKkFKu7LBgMubV4ALgQ?e=zVruSV). Currently the checkpoint is missing and we are working on retrain the model.

(2) Generate slice imagess: Go to [Slice3D - Testing on Single Image](https://github.com/yizhiwang96/Slice3D?tab=readme-ov-file#testing-on-single-image), using regression-based slicing.

### Download Preprocessed GSO Dataset

Create the data directory and download our preprocessed Google Scanned Objects (GSO) dataset:

```bash
mkdir 00_data
cd 00_data
```

Download the preprocessed dataset from [One Drive](https://1sfu-my.sharepoint.com/:u:/g/personal/mza143_sfu_ca/EV4TFxmcy3FJnXEsMuWfKzgB-q8KNEOhnbnL9OaycoSgiQ?e=IHAn0s) and save it to the data folder

Extract the dataset:
```bash
unzip gso_preprocessed.zip
rm gso_preprocessed.zip
cd ..
```

The dataset includes:
- Estimated input view camera poses
- Cropped GSO images  
- Predicted GSO image slices

## Usage

### Step 1: Compute Semantic Difference Maps

Generate semantic difference maps between input views and image slices:

```bash
python 01_compute_semantic_diff/compute_diff_gso.py
```

**Options:**
- `--model dino`: Use DINO feature extractor instead of default VGG

**Output:** Results saved to `00_data/diff_blocks_{feature_extractor}_normalized/`

### Step 2: Trajectory Planning

Construct semantic difference blocks and compute adaptive camera trajectories:

```bash
python 02_compute_trajectory/cal_vis_area.py
python 02_compute_trajectory/cal_cam_trajectories.py
```

**Process:**
1. Calculate segment-wise optimal elevation increments
2. Convert increments to absolute azimuth and elevation pairs

**Output:** Trajectories saved to `06_results/predicted_path/`

### Step 3: Video Generation

Generate orbital videos using SV3D:

1. Download the [SV3D checkpoint](https://huggingface.co/stabilityai/sv3d/blob/main/sv3d_p.safetensors) to `./05_externals/sv3d/checkpoints/`

2. Generate videos:
```bash
bash 03_generate_videos/gen_gso_videos.sh
```

**Output:** Videos saved to `06_results/generated_videos/`

### Step 4: 3D Reconstruction

Extract frames and prepare data for reconstruction:

```bash
python 04_reconstruction/cal_camera_params.py
python 04_reconstruction/process_video_data.py
```

This processes video data to extract 21 image frames and separate foreground from background.

#### Option A: NeUS Reconstruction

**Setup:**
Create a separate environment following the [NeUS installation guide](https://github.com/Totoro97/NeuS).

**Run reconstruction:**
```bash
conda activate neus
bash 04_reconstruction/recon_neus_gso.sh
```

Uses all 21 views for high-quality reconstruction.

#### Option B: InstantMesh Reconstruction

**Setup:**
Create a separate environment following the [InstantMesh installation guide](https://github.com/TencentARC/InstantMesh).

**Run reconstruction:**
```bash
conda activate instantmesh
python 04_reconstruction/organise_im_data.py
bash 04_reconstruction/recon_im_gso.sh
```

Uses 6 key frames for faster reconstruction.

## Repository Structure

```
ACT-R/
├── 00_data/                    # Dataset and computed features
├── 01_compute_semantic_diff/   # Semantic difference computation
├── 02_compute_trajectory/      # Camera trajectory planning
├── 03_generat e_videos/         # Video generation scripts
├── 04_reconstruction/          # 3D reconstruction pipelines
├── 05_externals/              # External model checkpoints
├── 06_results/                # Output results
└── requirements.txt           # Python dependencies
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{wang2025act,
  title={ACT-R: Adaptive Camera Trajectories for 3D Reconstruction from Single Image},
  author={Wang, Yizhi and Zhao, Mingrui and Mahdavi-Amiri, Ali and Zhang, Hao},
  journal={arXiv preprint arXiv:2505.08239},
  year={2025}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For questions or issues, please open an issue on the GitHub repository.