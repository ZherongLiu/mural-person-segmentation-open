# Public GitHub Publishing Checklist

Before the first public push, the project owner must confirm every item:

- [ ] Written authorization has been obtained to distribute all 138 color test images publicly.
- [ ] The 25-temple sample directory contains only grayscale derivatives with a maximum dimension of 1024 pixels.
- [ ] No original color field photographs are included outside the approved test split.
- [ ] The repository contains no precise coordinates, personal data, security information, or confidential
      conservation records.
- [ ] Third-party SAM 2 and Grounding DINO materials are linked to their sources and have not been copied without
      applicable licenses.
- [ ] Code, model, and data licenses have been approved by the project owner or responsible institution.
- [ ] `python scripts/verify_release.py` passes.
- [ ] `git lfs ls-files` lists 138 test images and two model weights, for a total of 140 files.

The local repository has been initialized and staged, but no Git author identity was set on behalf of the project
owner. Before upload, run:

```powershell
git config user.name "YOUR NAME OR INSTITUTION"
git config user.email "YOUR GITHUB EMAIL"
git commit -m "Initial open release"
git remote add origin https://github.com/<OWNER>/<REPOSITORY>.git
git push -u origin main
```

GitHub's web uploader rejects ordinary files larger than 100 MB. Use Git LFS from the command line, or attach the
weights to a GitHub Release and add download links to the README.

---

# GitHub 公开发布检查清单

第一次推送到公开仓库前，项目负责人必须逐项确认：

- [ ] 已书面确认有权公开传播全部138张彩色测试图片。
- [ ] 25庙展示目录中只有灰度、最长边不超过1024像素的派生样例。
- [ ] 除经批准的测试集外，仓库没有彩色实拍原图。
- [ ] 仓库没有精确坐标、个人信息、安防信息和保密修缮记录。
- [ ] 第三方 SAM 2 与 Grounding DINO 只提供来源链接，没有无许可复制其代码或权重。
- [ ] 代码许可、模型许可和数据许可已经得到项目负责人或所属机构批准。
- [ ] `python scripts/verify_release.py` 检查通过。
- [ ] `git lfs ls-files` 能列出138张测试图和2个模型权重，共140个文件。

本地仓库已经初始化并完成暂存，但没有代替项目负责人设置Git作者身份。上传前执行：

```powershell
git config user.name "你的姓名或机构名称"
git config user.email "你的GitHub邮箱"
git commit -m "Initial open release"
git remote add origin https://github.com/<OWNER>/<REPOSITORY>.git
git push -u origin main
```

如果使用GitHub网页直接上传，模型权重会超过100 MB限制。必须使用Git LFS命令行推送，
或者将权重改为GitHub Release附件并在README中提供下载链接。
