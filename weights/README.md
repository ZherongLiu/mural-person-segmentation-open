# Model Weights

- `maskrcnn_person_latest.pth`: the project-trained Mask R-CNN ResNet50-FPN person model.
- `sam2_mural_best.pt`: a mural-domain-tuned checkpoint based on SAM 2.1 Hiera Base+.

Both files exceed GitHub's ordinary 100 MB file limit and must be managed through Git LFS. After cloning, run
`python scripts/verify_release.py` to verify their SHA-256 checksums.

---

# 模型权重

- `maskrcnn_person_latest.pth`：项目训练的Mask R-CNN ResNet50-FPN人物模型。
- `sam2_mural_best.pt`：以SAM 2.1 Hiera Base+为基础进行壁画领域微调的权重。

两个文件均超过GitHub普通文件100 MB限制，必须使用Git LFS。克隆后运行
`python scripts/verify_release.py`检查权重SHA-256。
