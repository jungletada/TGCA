# MCTformer+ patch-first 顺序消融

唯一改动：进入所有 12 个原始 ViT block 的序列，从 `[class, patch]`
改为 `[patch, class]`。Class / patch 各自的位置编码、组内顺序不变。
不交换两类 token 的位置编码，不拆分 attention，不改变信息交互方向。

实现开关：`--patch-first`，默认关闭。关闭时原 baseline 数值不变。
开启后每层 CCT 收集末尾的原始 class tokens。12 层执行结束后，仅将读出
恢复为现有接口的 class / patch 切片。CAM 将记录的 attention 两个轴按语义
索引还原，继续原始 last-three C2P、3x3 head、sqrt、all-layer P2P 公式。
Transformer 内部从不还原为 class-first。`forward_features` 的 attention
记录按实际序列顺序返回；CAM 公共 attention 输出采用原语义顺序。

原始 class mean、CCT、3x3 Conv、GWRP、损失权重、优化器、学习率调度不变。
此消融仅支持 vanilla、joint、BCSS E0、PSL baseline、无其他实验变体。
Checkpoint 的 `model_spec.patch_first` 保存开关，CAM 要求显式匹配；
旧 checkpoint 缺省为 false。参数数量及 state_dict 键和值初始化不变。

## 预期与解释边界

无因果 mask / 相对序列偏置时，self-attention 对完整 token 排列等变。
各自位置编码跟随 token 移动，所以这是数学等价的顺序控制实验，不能把
仅由数值精度或训练轨迹差异导致的结果差异解释为新语义机制。
检查相同权重下的 logits、raw class tokens、梯度和 native CAM 一致性。

## 本次执行

先做 VOC（未排入新的 COCO 训练）。对照为已完成的
`results/default_mctformerplus/20260912-voc-coco-s0-r2/VOC12`。
保持 seed 0、448、batch 32、45 epochs、DeiT-S 官方初始化、nominal LR
5e-4、min LR 1e-5 及所有其余训练默认值不变。
复用 baseline 的实验函数，唯一额外训练/CAM 参数为 `--patch-first`。
先跑 64 张训练 / 4 张验证 / 2 张 CAM 的 smoke，再全量训练。
之后原三尺度 raw CAM 评估 1464 张 train，固定 .45，完整阈值网格
0:.01:.59。无 CRF / 下游 refinement / segmentation。

入口：`python -m experiments.ablations.run_patch_first --output <new-directory>`。
结果记录 exact commands、配置、环境、测试日志、代码 SHA、checkpoint SHA256，
最终 `comparison.csv` 和 `PATCH_FIRST_REPORT.md`。
