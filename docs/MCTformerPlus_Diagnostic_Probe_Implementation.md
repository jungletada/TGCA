# MCTformer+ 训练动力学探针：实现与使用说明

本说明对应 [诊断计划](MCTformerPlus_Diagnostic_Probe_Plan.md)。本次完成 S0–S2 的代码基础设施、测试及后续运行入口；没有启动 S3 的新训练，没有修改正在运行的 COCO all-product，也没有得出 H0–H3 的实验结论。原计划文件保持不变。

## 实现范围

- `analysis/diagnostics/metrics.py`：D1 共现正类 token cosine、D2 有效支撑比、D3 Gini、D5 GWRP/C2P rank 与 JS 背离度。Spearman 使用 average ranks 处理 ties；常数输入返回缺失值。
- `spectral.py`：D4 空间 FFT、径向谱、高频比例、二维自相关及固定范围 ξ 拟合。不改变模型特征或训练计算。
- `probe_set.py`：固定 256 张图像级标签 probe，seed=0、每类至少 8 张；另选不重叠的 500 张带 mask 的 mini-CAM 图像。所有缓存写到 `results/`，不写 `data/probe/`。
- `probe.py`、`state.py`：独立 eval/no-grad/FP32 前向；保存并恢复所有子模块 train/eval 状态、Python/NumPy/Torch CPU/CUDA RNG 和 TF32 开关。DDP 只由 rank 0 前向和写盘，初始化及运行错误向其他 ranks 同步。
- `writer.py`：独立目录、逐 epoch 完成标记、长/宽 CSV、谱数组、配置、命令、Git SHA、集合/标签哈希和内存模型状态指纹。禁止覆盖已有运行。
- `tools/evaluate_mini_cam.py`：调用现有 `MCTformerPlusCam.get_cam`，复用原有 CAM normalization、threshold confusion 和指标实现；不复制第二套 mIoU 算法。
- `plot_trajectory.py`：GWRP/C2P 轨迹 PDF、逐层 κ/Gini 热图 PDF、逐 seed 相关性与峰值偏差表、跨 seed 均值/标准差及判定条件记录。
- `train_model_v2.py`：仅增加默认关闭的参数及 epoch 边界调用。`engine.py`、模型、loss、优化器、调度和 best/final checkpoint 保存条件未改变。

## 已明确的统计与几何口径

1. **448 是当前匹配训练分辨率。** 语义 probe 跟随 `--input-size`，谱分辨率默认 448，网格两边必须至少 28。两者相同时复用同一次前向，不无谓重复计算。确定性变换为 resize 短边至 `int(res×512/448)`、center crop、ImageNet normalize。
2. **前向后的 class tokens 是输入相关的。** `rho_cc` 是每图共现正类对的均值；单标签图无可用类别对时记缺失，不填零。`rho_cc_all_image` 是每图所有类别对；`rho_cc_all` 是先跨 probe 图像求每类输出 token 均值，再算类别间 cosine；`rho_cc_parameter` 单独衡量输入无关的 `cls_token` 参数，不把参数与输出混称。
3. **D2/D3/D5 默认保存全部 12 层与实际聚合结果。** 配置层号为 0-based，CSV 层号为 1-based；`aggregate` 通过模型自身 C2P helper 计算。逐层 D5 均对比同一次前向的最终 3×3 classifier 原始 logits，不把中间层特征再送进分类头。若启用 affinity，aggregate 包含实际 affinity 传播，逐层量不包含。
4. **GWRP 权重严格匹配原实现。** 使用原 `[B,N,C]` sort 和 FP32 `torch.logspace`，再散射回空间顺序。并列 logits 的顺序沿用原排序，不另定 GWRP 规则。JS 单位为 nats。
5. **统计先按图像归约，再等权汇总。** 保存 mean、population std、有效/总样本数及 per-class 结果；std 不是置信区间。COCO per-class 输出按固定 probe 标签频次选 top-20 类，整体指标仍使用全部正类。
6. **D4 先 L2 归一化 token、再逐通道去均值及加 Hann 窗。** 常数场去均值后无有效能量，不能同时要求“全部能量仍在 DC”。零能量时高频比例/ξ 记缺失。白噪声没有可靠的正指数相关尾部，不强行令 ξ≈1。
7. **ξ 从完整二维功率谱的 inverse FFT 得到自相关后径向归约。** 不把径向谱的一维 irfft 当作严格二维自相关；固定拟合半径 1–5 个 patch，不调范围，要求正尾部、负 slope、R²≥0.8。主 ξ 在 checkpoint 级平均自相关上拟合；保存 R²/slope/valid，不对失败值强行截断出一个大数。
8. **E_hi 严格采用计划中的径向 bin 均值比值。** 它不是按环面积加权的全部二维 Fourier 能量比例。保存 bin centers/counts 以避免混用两种定义。D4 per-class E_hi 只是按图像标签分组，不是目标物体区域谱。
9. **mini-CAM 与无 mask 的诊断分开。** mini 使用相同 resize/crop 几何，mask 用 nearest；single-scale、无 flip、无 CRF，保留原生 last3 mean C2P、sqrt、all-layer P2P CAM。固定阈值 0.45，共同网格 0.00–0.59/0.01，输出 semantic foreground precision/recall。它是裁剪子集诊断，不是正式原图多尺度 raw CAM 指标。
10. **图中的分类曲线是固定 probe 上的两分支 macro-class AP**，复用现有分类指标函数及 sigmoid，明确标为 `probe_*_macro_ap`，不是完整 val 指标或训练日志的 mean-image AP。

## 默认关闭的训练接入

普通命令不加 `--probe`，不创建 probe、不前向、不写诊断结果。启用时在原训练命令后增加：

```bash
--probe \
--probe-set results/NEW_RUN/sets/probe_train_ids.txt \
--probe-mini-set results/NEW_RUN/sets/mini_cam_train_ids.txt \
--probe-mini-mask-dir data/VOCdevkit/VOC2012/SegmentationClass
```

默认每轮执行一次、batch=32、全 12 层、谱分辨率 448，输出到训练目录的 `diagnostics/`。不传 `--probe-mini-set` 时仅运行无像素标注诊断，不读取 masks。其他开关：`--probe-out-dir`、`--probe-every`、`--probe-batch-size`、`--probe-spectral-size`、`--probe-layers`；`--probe-dump-raw` 默认关闭。

第一版支持普通 class-first、raw final tokens 的 joint MCTformer+ GWRP/C2P；不隐式扩展到 FinalLN、patch-first、解耦 blocks、BCSS/PSL/CTI 或 LaST 变体。不会切换 class-token 初始化。

**不增加每个 epoch 的 checkpoint。** 原 best/final 保存逻辑保持不变；在线诊断以 `model_state_sha256` 标识当时的内存权重，`checkpoint_sha256` 无对应磁盘文件时为 null。指纹不是可恢复权重，不能据此恢复旧训练的中间 epoch。

## 后续队列入口（本次未执行训练）

先预览，默认不创建输出、不启动训练：

```bash
bash experiments/diag_trajectory_voc.sh --output results/diagnostic_trajectory/NEW_VOC_RUN
bash experiments/diag_trajectory_coco.sh --output results/diagnostic_trajectory/NEW_COCO_RUN
```

VOC 队列固定顺序为 GWRP seed0 → C2P seed0 → GWRP seed1 → C2P seed1 → GWRP seed2 → C2P seed2。C2P 采用当前 VOC 固定阈值结果最好的 **all-product、无 affinity**；GWRP 的反事实 C2P 诊断也使用 all-product，但训练仍走原 GWRP。

两组使用相同 DeiT-S 初始化、448、45 epochs、batch32、AdamW、nominal LR=5e-4（实际 batch 缩放 LR=3.125e-5）、原 loss/augmentation。队列保存环境和命令，复核原基线的 optimizer/pretraining 配置；seed 是预注册的重复变量。COCO 入口只包含 DIAG-D seed0，不会自动接在 VOC 后执行。

实际训练需在检查 GPU/现有队列、提交实现之后，在新 tmux 中显式传 `--execute`。runner 保留 tracked-clean 和独立输出目录检查，不自动终止已有任务。当前 COCO all-product 不会被加挂探针或重启。

独立绘图示例（参数为各训练目录下的 `diagnostics/`）：

```bash
python -m analysis.diagnostics.plot_trajectory \
  results/NEW_RUN/gwrp_s0/diagnostics \
  results/NEW_RUN/c2p_s0/diagnostics \
  --output results/NEW_RUN/trajectory_analysis
```

## 判定边界与暂未执行部分

相关性按 epoch 计算，按 seed 单列；同一图像内 patches/类别对不当作独立观测。分析器核对 probe/mini/标签哈希及分辨率匹配，只读取完成的 epoch。满足 3 个匹配 seeds、两组都有、同方向、每条 |ρ|≥0.7 和峰值偏差≤2 才记录描述性门槛通过；单 seed 不通过。

κ/Gini 不是单调质量分数，δ 也没有预注册最优方向，因此只给相关性，不从同一条 mask-backed 曲线事后选取最有利的 argmin/argmax。ξ 无效则保留缺失并继续报告 E_hi，不更换拟合范围。时间序列相关有自相关、多指标比较和事后解释风险，**达到描述性门槛不等于验证了无标注 checkpoint 选择方法**。

S3 多 seed 实验、H0–H3 判定及正式结果报告均未完成。DIAG-C 尚需固定单 CLS 的 classifier/loss/CAM 定义，不能简单把类别数改成 1 冒充匹配对照。DIAG-E 是条件阶段，不自动延长训练；“最后一轮最大”不等于全程单调，“没有观察到退化”也不单独证明 multi-class token 的架构优势。本次不实现 Gram loss 或其他后续方法。

## 测试与当前验证

新增测试覆盖合成指标、ties/缺失、谱质量筛选、真实 GWRP 逐值权重、原生 CAM 逐位一致、固定集合/不重叠、路径限制、rank0 分支、FP32/模式/RNG 恢复、训练后续步一致，以及 CSV→图表完整流程。

真实 VOC/COCO 图像级标签的只读采样核对均能选出 256 张；VOC 最少类别覆盖 12 张、COCO 最少 8 张。没有为该检查读取语义 mask 像素或启动模型训练。

本次使用 CPU 测试，避免占用活动 COCO 的 GPU；CUDA 专项测试尚未在本次执行，DDP rank 分支测试不等于完成多 GPU 训练验证。GPU 开销是否低于计划估算的 15 秒/epoch 也尚未实测，不作保证。

最终验证：新增 `tests/test_diagnostics.py` **20 passed**；包含原 C2P/product/affinity、模型变体、patch-first、raw CAM 和 width aggregation 的兼容测试组 **78 passed, 8 skipped**（CUDA 项）。原 `engine.py` 的 class/CCT/patch loss + AdamW 两轮小型 CPU 训练中，probe 开关两侧的训练指标、最终权重及优化器状态逐位一致。

命令和日志保存在 [验证目录](../results/diagnostic_probe_implementation/20260915-code-v1/)，主要文件为 `commands.sh`、`tests_final.log`、`diagnostics_final.log`、`VALIDATION.md`。本目录只记录实现验证，不是 S3 实验结果。
