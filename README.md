# Digital Heritage Mural Person Segmentation

[English](#english) | [中文版](#中文版)

## English

An open project for locating and segmenting human figures in cultural-heritage murals. The repository includes
a trained Mask R-CNN model, a domain-tuned SAM 2 model, reproducible scripts, a fixed test set, and privacy-reviewed
grayscale examples from 25 temples.

### Workflow

```text
image
  -> Mask R-CNN person candidate boxes
  -> box + foreground point prompts
  -> domain-tuned SAM 2 masks
  -> PNG masks, overlays and structured JSON
```

The optional full text-prompt experiment also supports:

```text
Grounding DINO text candidates + frozen Mask R-CNN candidates
  -> NMS
  -> domain-tuned SAM 2
```

### Included release

- Mask R-CNN checkpoint: `169.9 MiB`
- Tuned SAM 2 checkpoint: `308.7 MiB`
- Fixed public test split: 138 images and 422 person instances
- Public 25-temple sample subset: 25 grayscale derivatives
- Training, evaluation and inference scripts
- Machine-readable metrics and reports

### Quick start

1. Install Git LFS before cloning.
2. Follow `docs/INSTALL.md` to install PyTorch and the official SAM 2 package.
3. Verify the download:

```powershell
python scripts/verify_release.py
```

4. Run inference:

```powershell
python scripts/infer_person.py --input path/to/images --output-dir outputs/inference
```

### Openness statement

This project follows an “open by default, confidential by exception” policy. Project-owned code, model weights,
de-identified annotations, approved test data, and non-sensitive metadata are released unless they contain heritage
security information, precise locations, personal data, unauthorized color field photographs, restricted third-party
materials, or information explicitly limited by an agreement. See `OPEN_SOURCE_STATEMENT.md` and `DATA_LICENSE.md`.

### Important limitations

Outputs are machine pre-annotations. Damaged, low-contrast, overlapping or very small figures may be missed,
split into multiple instances or only partially segmented. Expert review remains required.

---

## 中文版

面向文物壁画人物的开放人物定位与实例分割项目。仓库提供训练后的 Mask R-CNN、领域微调
SAM 2、可复现实验脚本、固定测试集以及25座庙宇的脱敏灰度样例。

### 工作流程

```text
image
  -> Mask R-CNN person candidate boxes
  -> box + foreground point prompts
  -> domain-tuned SAM 2 masks
  -> PNG masks, overlays and structured JSON
```

完整文本提示实验还支持：

```text
Grounding DINO 文本候选框 + 冻结的 Mask R-CNN 候选框
  -> NMS 去重合并
  -> 领域微调 SAM 2
```

### 发布内容

- Mask R-CNN 权重：`169.9 MiB`
- 微调后的 SAM 2 权重：`308.7 MiB`
- 固定公开测试集：138 张图片、422 个人物实例
- 25 庙公开样例：25 张灰度派生图
- 训练、评估和推理脚本
- 机器可读的指标与报告

### 快速开始

1. 克隆前安装 Git LFS。
2. 按照 `docs/INSTALL.md` 安装 PyTorch 和官方 SAM 2。
3. 检查下载完整性：

```powershell
python scripts/verify_release.py
```

4. 执行推理：

```powershell
python scripts/infer_person.py --input path/to/images --output-dir outputs/inference
```

### 开放声明

本项目采用“默认开放、保密例外”原则。除文物安全敏感信息、精确位置、个人信息、未获授权的
彩色实拍原图、第三方受限材料及合作协议明确限制的内容外，项目自有代码、模型权重、脱敏标注、
测试数据和非敏感元数据予以公开。具体边界见 `OPEN_SOURCE_STATEMENT.md` 和 `DATA_LICENSE.md`。

### 重要限制

输出属于机器预标注。残损、低对比、重叠或尺寸很小的人物可能被漏检、拆分为多个实例，
或只完成局部分割。所有结果仍需专业人员复核。
