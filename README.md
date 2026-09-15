# TGCA — MCTformer+ 弱监督语义分割实验仓库

本仓库基于 MCTformer/MCTformer+，用于研究图像级标签监督下的多类别 token、patch token、attention 与类别激活图（CAM）之间的关系，支持基线复现、匹配消融、分类验证和 CAM 定位评估。

当前主要工作是比较 MCTformer+ patch branch 的 GWRP 与实际 class-to-patch attention（C2P）加权 pooling，包括最后三层/全部层、平均/连乘，以及 patch-to-patch affinity 传播。近期实验在 VOC、COCO 上沿用匹配训练配置，评估到原生 raw CAM，不包含后续 CRF 或分割网络训练。

仓库也保留 Token-Group Calibrated Attention（TGCA）的实现和机制诊断代码；代码存在不代表其有效性已经完成验证，结论应以对应实验报告为准。

## 目录与入口

- `models/`：MCTformer+、attention 与 pooling 的模型实现。
- `train_model_v2.py`、`engine.py`：训练入口和训练/验证逻辑。
- `make_cam.py`：原生 CAM 生成入口。
- `experiments/`：基线、消融及诊断的实验运行脚本。
- `analysis/`、`tools/`：特征与 attention 分析、分类及 CAM 评估工具。
- `tests/`：数值一致性、梯度、混合精度及评估流程测试。
- `docs/`：实验设计、分析说明与结果总结。
- `results/`：本地实验指标、报告、命令、配置、日志和 checkpoints。

## 实验与结果

优先阅读 [C2P Product / Affinity 实验设计与结果对比](docs/MCTformerPlus_C2P_Product_Affinity_Design_and_Results.md)，其中汇总了 GWRP、两组 mean、两组 product 和两组 affinity 的已完成 VOC 对照，以及精确结果路径。

相关实验计划：

- [C2P pooling](docs/MCTformerPlus_C2P_Pooling.md)
- [全部层 C2P pooling](docs/MCTformerPlus_C2P_All_Layers.md)
- [Product pooling](docs/MCTformerPlus_C2P_Product.md)
- [Product + P2P affinity](docs/MCTformerPlus_C2P_Product_P2P_Outline.md)
- [COCO all-product](docs/MCTformerPlus_C2P_All_Product_COCO.md)

主要指标包括 class-token / patch-head macro mAP、分类验证 loss、固定阈值 raw CAM mIoU、共同阈值曲线及前景 precision/recall。分类与 CAM 的数据 split、阈值和后处理必须按各报告区分；单 seed 结果不代表稳定收益。

## 使用说明

当前主实验使用 Conda 环境 `tgca-repro`。数据集和预训练权重需单独准备，路径、环境清单及实际命令以对应运行目录的记录为准，不假定克隆仓库后已包含这些文件。

复现实验前先阅读对应计划，再核对代码版本、初始化、seed、训练配置和评估协议。已有运行的 `manifest.json`、`commands.sh`、配置及 checkpoint SHA256 是复核依据；新运行使用独立输出目录，不覆盖已有结果，也不要重复启动活动队列。长时间实验通过 tmux 执行。

部分逐图 CAM 和 smoke 中间文件已按授权清理；最终数值与非 smoke checkpoints 保留在本地结果目录，具体文件状态以报告和清理记录为准。
