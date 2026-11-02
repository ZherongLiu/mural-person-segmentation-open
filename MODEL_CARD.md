# Model Card

## Components

- Mask R-CNN ResNet50-FPN: finds mural person candidates and produces initial instance masks.
- SAM 2.1 Hiera Base+: uses candidate boxes and foreground points to produce more precise mural-person boundaries.
- Grounding DINO: optional third-party text-prompt candidate generator; its weights are not included.

## Default inference parameters

- Mask R-CNN person confidence threshold: 0.5
- SAM 2 prompt mode: candidate box plus foreground point
- Candidate-box expansion: 2%
- SAM 2 mask threshold: 0.0

The optional full text-prompt workflow uses a Grounding DINO candidate score of 0.5, text-matching threshold of 0.15,
frozen Mask R-CNN candidate score of 0.5, and cross-source NMS IoU threshold of 0.5.

## Reported evaluation

- Mask R-CNN at epoch 20, evaluated on 138 images and 422 ground-truth persons: precision 92.66%, recall 86.73%,
  F1 89.60%, and mean matched IoU 82.76%, using an IoU matching threshold of 0.5.
- Text candidates plus frozen Mask R-CNN and domain-tuned SAM 2, evaluated on 137 labeled images and 422 ground-truth
  persons: precision 73.54%, recall 86.26%, F1 79.39%, and mean matched IoU 80.11%.

These results come from different candidate-generation workflows and must not be interpreted as a direct comparison
in which only SAM 2 changed.

## Intended use

The model supports batch mural-person pre-annotation, person-region retrieval, expert review, and linking visual
instances to a knowledge graph. It is not intended for unreviewed identity attribution, cultural interpretation,
conservation decisions, or security use. Damaged, low-contrast, occluded, overlapping, and very small figures may
still be missed, falsely detected, partially segmented, or duplicated.

---

# 模型卡

## 模型组成

- Mask R-CNN ResNet50-FPN：寻找壁画人物候选区域并输出基础实例掩膜。
- SAM 2.1 Hiera Base+：使用候选框和人物内部前景点生成更精细的壁画人物边界。
- Grounding DINO：可选的第三方文本提示候选模型；本仓库不附带其权重。

## 默认推理参数

- Mask R-CNN 人物置信度阈值：0.5
- SAM 2提示方式：候选框＋前景点
- 候选框外扩：2%
- SAM 2掩膜阈值：0.0

可选的完整文本提示工作流使用 Grounding DINO候选分数0.5、文本匹配阈值0.15、
冻结Mask R-CNN候选分数0.5，以及两路候选合并NMS IoU阈值0.5。

## 已报告测试结果

- Mask R-CNN第20轮，138张图片、422个GT人物：Precision 92.66%、Recall 86.73%、
  F1 89.60%、平均匹配IoU 82.76%，匹配阈值为IoU 0.5。
- 文本候选＋冻结Mask R-CNN＋领域SAM 2，137张有标注图片、422个GT人物：
  Precision 73.54%、Recall 86.26%、F1 79.39%、平均匹配IoU 80.11%。

上述两组数据属于不同候选工作流，不能解释为只改变SAM 2后的直接对比。

## 适用范围

适合壁画人物批量预标注、人物区域检索、专家复核辅助，以及将视觉实例连接到知识图谱。

不适合未经专家复核直接进行人物身份认定、文化含义判断、文物修复决策或安防用途。

残损、低对比、遮挡、多人重叠和极小人物仍可能出现漏检、误检、局部分割或重复实例。
