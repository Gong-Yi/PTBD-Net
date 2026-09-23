# PTBD-Net

**Progressive Target-Background Discrimination Network** for infrared small-target segmentation.

This repository contains the PTBD-Net model and a PyTorch training script. The model combines a DHiF encoder, BGHEC feature fusion, and RAHMoE decoder components, with deep supervision during training.

## Repository contents

```text
PTBD-Net/
├── model/
│   ├── bghec.py
│   ├── config.py
│   ├── dhif.py
│   ├── ptbd_net.py
│   └── rahmoe.py
└── train.py
```

## Requirements

- Python 3
- PyTorch (install a build compatible with your operating system and CUDA setup, if applicable)
- NumPy
- Pillow
- SciPy
- TensorBoard
- `ml-collections`

Install the Python packages with:

```bash
pip install numpy pillow scipy tensorboard ml-collections
```

Install PyTorch separately using the instructions for your machine at [pytorch.org](https://pytorch.org/get-started/locally/).

## Dataset layout

Set `--dataset_dir` to the directory containing one subdirectory per dataset. Each dataset directory should contain `images`, `masks`, and `img_idx`:

```text
data/
└── NUAA-SIRST/
    ├── images/
    │   ├── image_1.png
    │   └── ...
    ├── masks/
    │   ├── image_1.png
    │   └── ...
    └── img_idx/
        ├── train_NUAA-SIRST.txt
        └── test_NUAA-SIRST.txt
```

The train and test text files contain one image stem per line, without the image extension. The corresponding image and mask files must use matching stems. The loader supports `.png` and `.bmp` image/mask pairs.

The script includes normalization statistics for `NUAA-SIRST`, `NUDT-SIRST`, and `IRSTD-1K`. For another dataset name, it calculates statistics from the image files listed in that dataset's train and test index files.

## Train

From the repository root, run:

```bash
python train.py --dataset_names NUAA-SIRST --dataset_dir ./data
```

To train the listed datasets one after another:

```bash
python train.py --dataset_names IRSTD-1K NUDT-SIRST NUAA-SIRST --dataset_dir ./data
```

Useful options include:

| Option | Default | Description |
| --- | --- | --- |
| `--dataset_names` | `NUAA-SIRST` | One or more dataset directory names |
| `--dataset_dir` | `./data` | Root directory containing dataset folders |
| `--batchSize` | `16` | Training batch size |
| `--patchSize` | `256` | Random training crop size |
| `--epochs` | `1000` | Number of training epochs |
| `--begin_test` | `500` | Epoch at which periodic evaluation starts |
| `--every_test` | `5` | Evaluate every N epochs after `--begin_test` |
| `--save` | `./checkpoints` | Checkpoint output directory |
| `--log_dir` | `./logs/training` | TensorBoard log directory |

The script uses CUDA for training. It writes checkpoints under `--save/<dataset-name>/` and TensorBoard logs under `--log_dir`. Evaluation during training runs on the same dataset currently being trained.

## TensorBoard

Start TensorBoard with:

```bash
tensorboard --logdir ./logs/training
```

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.

## 中文说明

PTBD-Net（Progressive Target-Background Discrimination Network）用于红外小目标分割。本仓库提供 PyTorch 模型实现和训练脚本，模型由 DHiF 编码器、BGHEC 特征融合模块和 RAHMoE 解码器组成。

数据集根目录由 `--dataset_dir` 指定，每个数据集子目录需要包含 `images/`、`masks/` 和 `img_idx/`；索引文件每行填写一个不带扩展名的图像文件名。训练命令示例：

```bash
python train.py --dataset_names NUAA-SIRST --dataset_dir ./data
```

训练依赖包括 PyTorch、NumPy、Pillow、SciPy、TensorBoard 和 `ml-collections`。使用与本机 CUDA 环境匹配的 PyTorch 安装包。许可证为 MIT，具体条款见 [LICENSE](LICENSE)。

