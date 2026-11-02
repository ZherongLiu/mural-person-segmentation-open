# Installation Guide

## 1. Clone with Git LFS

The two model weights and 138 test images are managed by Git LFS. A normal ZIP download may contain only pointer
files, so use:

```powershell
git lfs install
git clone <YOUR_REPOSITORY_URL>
cd mural-person-segmentation-open
git lfs pull
```

## 2. Create the Python environment

Python 3.12 and a CUDA-capable NVIDIA GPU are recommended.

```powershell
py -3.12 -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

## 3. Install official SAM 2

This repository includes the project-tuned checkpoint but does not redistribute the third-party SAM 2 source code.

```powershell
git clone https://github.com/facebookresearch/sam2.git external/sam2
pip install -e external/sam2
```

The full text-prompt workflow also uses `IDEA-Research/grounding-dino-base`. Grounding DINO is optional and is
downloaded by Transformers the first time that workflow is run.

## 4. Verify file integrity

```powershell
python scripts/verify_release.py
```

The verifier checks model hashes, 138 test images, 422 person annotations, 25 grayscale samples, and possible private
paths or secrets.

## 5. Run person segmentation

```powershell
python scripts/infer_person.py --input path/to/images --output-dir outputs/inference
```

Defaults: Mask R-CNN score threshold 0.5, SAM 2 box-plus-foreground-point prompts, 2% box expansion, and mask
threshold 0.0.

---

# 安装说明

## 1. 使用 Git LFS 克隆

两个模型权重和138张测试图片由 Git LFS 管理。普通下载 ZIP 可能只得到指针文件，建议使用：

```powershell
git lfs install
git clone <你的仓库地址>
cd mural-person-segmentation-open
git lfs pull
```

## 2. 创建 Python 环境

建议使用 Python 3.12 和支持 CUDA 的 NVIDIA GPU。

```powershell
py -3.12 -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

## 3. 安装官方 SAM 2

本仓库提供项目微调权重，但不重复分发第三方 SAM 2 源代码。

```powershell
git clone https://github.com/facebookresearch/sam2.git external/sam2
pip install -e external/sam2
```

完整文本提示工作流还会使用 `IDEA-Research/grounding-dino-base`。Grounding DINO 是可选依赖，
首次运行文本提示工作流时由 Transformers 下载。

## 4. 检查文件完整性

```powershell
python scripts/verify_release.py
```

检查内容包括模型哈希、138张测试图、422个人物标注、25张灰度样例和敏感路径扫描。

## 5. 运行人物分割

```powershell
python scripts/infer_person.py --input path/to/images --output-dir outputs/inference
```

默认参数为 Mask R-CNN 阈值0.5、SAM 2框加前景点提示、候选框外扩2%、掩膜阈值0.0。
