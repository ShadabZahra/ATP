# ATP – UCF101 & HMDB51 Zero-Shot Evaluation

This repository provides a simplified pipeline to evaluate **ATP** on the **UCF101** and **HMDB51** datasets using zero-shot evaluation.
The workflow includes environment setup, dataset preparation, frame extraction, pretrained model loading, configuration updates, and evaluation scripts.

---

## 1. Environment Setup

We provide an `environment.yaml` file to create the required Conda environment.

```bash
conda env create -f environment.yaml
conda activate ATP
```

This installs all necessary dependencies for running ATP, including PyTorch and required utilities.

---

## 2. Dataset Preparation

Download the datasets from the official sources:

* **UCF101:** [link](http://crcv.ucf.edu/data/UCF101.php)
* **HMDB51:** [link](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)

After downloading, you should have the raw `.avi` or `.mp4` video files.

---

## 3. Convert Videos to Frames

We use **`video2frame.py`** to convert each video into a folder of frames.

### Usage

```bash
python video2frame.py --input_dir <path_to_videos> --output_dir <where_to_save_frames>
```

Example:

```bash
python video2frame.py --input_dir ./datasets/UCF101/videos \
                      --output_dir ./datasets/UCF101/frames
```

Required arguments:

* `--input_dir`  : directory containing video files
* `--output_dir` : directory where extracted frames will be saved

---

## 4. Download Pretrained Kinetics Model

You must download the pretrained Kinetics-400 ActionCLIP model:

**Pretrained Kinetics Weights:**
[link](https://drive.google.com/drive/folders/1osuph2BJVPUsI_fr92cjRzvKOz8W402p)

Place the downloaded checkpoint (e.g., `.pt` or `.pth`) in your preferred directory.

---

## 5. Update Configuration Files

Before running evaluation, update the configuration YAML file:

### You must provide:

1. **Path to pretrained Kinetics model**

```yaml
pretrain: "/absolute/path/to/pretrained_kinetics_ckpt.pt"
```

2. **Path to dataset frames**

For example in `configs/ucf101/ucf_zero_shot.yaml`:

```yaml
data:
  root: "/absolute/path/to/UCF101/frames"
```

And similarly for HMDB51:

```yaml
data:
  root: "/absolute/path/to/HMDB51/frames"
```

---

## 6. Zero-Shot Evaluation

### Run Zero-Shot on **UCF101**

```bash
bash scripts/run_test.sh ./configs/ucf101/ucf_zero_shot.yaml
```

### Run Zero-Shot on **HMDB51**

```bash
bash scripts/run_test.sh ./configs/hmdb51/hmdb_zero_shot.yaml
```

All results, logs, and predictions will be automatically stored in the output directory defined in the YAML file.

---

## Contributors

* **Shadab Zahra**
* **Taejoon Kim**

---

## Acknowledgments

This project is based on the official **ActionCLIP** implementation:
[link](https://github.com/sallymmx/ActionCLIP/blob/master/README.md)

We thank the original authors for releasing their excellent work.

---
