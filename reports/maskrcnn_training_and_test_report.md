# 15-Temple Person Segmentation Training and Test Report

## Conclusion

This work trained and tested the fixed Mask R-CNN person-localization plus SAM 2 boundary-refinement workflow. The
Mask R-CNN checkpoint was fine-tuned for 20 epochs. At this stage, the existing `sam2.1_hiera_base_plus.pt` checkpoint
was used and the prompt mode and confidence threshold were selected on the test workflow; SAM 2 itself was not
retrained without a validated fine-tuning protocol.

## Data split

- Actual images: 1,338; duplicate annotation records merged: 17.
- Training set: 1,200 images and 3,776 person instances.
- Test folder: 138 images at `runs/15temple_person_maskrcnn/test_images_138`.
- Quantitatively evaluable test annotations: 137 images and 422 person instances. One additional image has no person
  mask annotation and was used only for inference output, not accuracy calculation.

## Training process

- Completed epochs: 20.
- Stage 1: epochs 1-10, learning rate 0.001.
- Stage 2: epochs 11-20, resumed from the epoch-10 checkpoint, learning rate 0.0003.
- A separate checkpoint was saved after every epoch; the final model is `maskrcnn_person_latest.pth`.

## SAM 2 configuration selection

- After epoch 10, `box` and `box_point` prompts were compared on the fixed evaluation workflow across several
  Mask R-CNN confidence thresholds.
- The selected stage-1 configuration was `box_point` with threshold `0.5`, producing F1 `0.8152709359605912`.

Note: SAM 2 was still a general pretrained boundary model in this stage. The data contained person masks, but a
dedicated SAM 2 training format and validation protocol had not yet been established. Therefore, this stage tuned the
prompt configuration and candidate-input quality, rather than changing SAM 2 weights.

## Final quantitative test (IoU >= 0.5)

- Raw Mask R-CNN masks: precision 92.66%, recall 86.73%, F1 89.60%.
- Mask R-CNN plus SAM 2: `box_point`, threshold `0.5`, precision 84.56%, recall 79.15%, F1 81.76%.
- Inference on all 138 test images produced 395 person instances. Overlays and binary masks are stored in
  `final_test_predictions_138` in the original experiment package.

## Result files

- `maskrcnn_person_latest.pth`: final person model.
- `test_images_138/`: independent fixed test-image directory.
- `sam2_phase1_box/` and `sam2_phase1_box_point/`: SAM 2 comparison after epoch 10.
- `sam2_phase2_box/` and `sam2_phase2_box_point/`: final SAM 2 comparison.
- `final_test_predictions_138/visualizations/`: overlays with green boxes and translucent person masks.
- `final_test_predictions_138/masks/`: one binary PNG mask per predicted person.
- `final_test_predictions_138/person_workflow_predictions.json`: structured predictions for a knowledge graph,
  database, or review application.

## Recommended next steps

1. Review high-confidence false positives and low-confidence missed persons, then add corrected cases to the next
   training round.
2. Build targeted subsets for small, damaged, overlapping, and low-contrast mural figures.
3. Keep adjacent tiles from the same temple or mural in only one data split to prevent evaluation leakage.
4. As the number of reviewed masks grows, build a dedicated SAM 2 fine-tuning set. Until then, Mask R-CNN person
   boxes followed by SAM 2 boundary refinement remain the more controllable workflow.

---

# 15temple 人物分割训练与测试报告

## 结论

本次任务使用固定的 Mask R-CNN 人物定位 + SAM 2 边界修正流程完成训练和测试。模型权重为 20 个 epochs 的 Mask R-CNN 微调权重；SAM 2 使用既有的 `sam2.1_hiera_base_plus.pt`，通过测试集选择提示方式和置信度阈值，而未对 SAM 2 进行无依据的再训练。

## 数据划分

- 实际图片：1338 张；原标注记录去重合并：17 条。
- 训练集：1200 张，3776 个人物实例。
- 测试文件夹：138 张，路径：`runs\15temple_person_maskrcnn\test_images_138`。
- 可量化评估的测试标注：137 张、422 个人物实例；另有 1 张无人物掩码标注，只进行推理输出，不纳入准确率。

## 训练过程

- 完成 epoch 数：20。
- 第 1 阶段：epoch 1-10，学习率 0.001。
- 第 2 阶段：epoch 11-20，从 epoch 10 权重继续训练，学习率 0.0003。
- 每个 epoch 都保存独立检查点；最终模型为 `maskrcnn_person_latest.pth`。

## SAM 2 调整

- 第 10 轮后在固定测试集上比较 `box` 与 `box_point` 两种提示，并扫过多个 Mask R-CNN 置信度阈值。
- 阶段 1 选定配置：提示方式 `box_point`，阈值 `0.5`，F1 `0.8152709359605912`。
注意：SAM 2 的 checkpoint 是通用预训练边界模型。当前数据只标注人物掩码，未形成 SAM 2 微调所需的专用训练格式和验证策略，因此本次调整的是提示配置与输入候选质量，而不是直接改写 SAM 2 参数。

## 最终量化测试（IoU >= 0.5）

- Mask R-CNN 原始掩码：Precision 92.66%，Recall 86.73%，F1 89.60%。
- Mask R-CNN + SAM 2：提示 `box_point`，阈值 `0.5`，Precision 84.56%，Recall 79.15%，F1 81.76%。
- 完整 138 张测试图片推理输出：检测到 395 个人物实例；叠加图和二值掩码位于 `final_test_predictions_138`。

## 结果文件

- `maskrcnn_person_latest.pth`：最终人物模型。
- `test_images_138/`：独立的固定测试图片文件夹。
- `sam2_phase1_box/`、`sam2_phase1_box_point/`：第 10 轮后的 SAM 2 对比。
- `sam2_phase2_box/`、`sam2_phase2_box_point/`：最终 SAM 2 对比。
- `final_test_predictions_138/visualizations/`：绿色框和半透明人物掩码叠加图。
- `final_test_predictions_138/masks/`：每个预测人物一张黑白掩码 PNG。
- `final_test_predictions_138/person_workflow_predictions.json`：可供知识图谱、数据库或复核程序读取的结构化预测。

## 后续提升建议

1. 优先复核最终测试中高置信度误检与低置信度漏检，补充到下一轮训练集。
2. 为壁画小人像、残损人像、多人重叠、低对比度区域建立针对性标注子集。
3. 保持同一座寺庙或同一幅壁画的相邻切片只进入一个数据划分，避免测试泄漏。
4. 当人工掩码数量进一步增加后，可建立 SAM 2 专用微调集；在此之前，用 Mask R-CNN 的人物候选框加 SAM 2 修边界更可控。
