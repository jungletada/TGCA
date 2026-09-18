# TGCA 归档清点 — Phase 0

日期：2026-09-18；主机：LHR；仓库：`/home/peng/code/TGCA`。
核查版本：`7edd58d`，分支 `main`。本文路径均相对仓库根目录，冒号后为当前文件行号。
依据：`docs/TGCA_Archive_Extraction_Spec.md:23`；该规格 SHA256：
`def034611ff570fdf28e1bfb22c756d61fc9277124440865871501b6c2345020`。

**状态：Phase 0 清点完成，等待用户确认；Phase 1–7 未开始。**
没有创建包、修改函数、安装依赖、复制 checkpoint、重新推理或重新计算实验指标。
启动时已有 `docs/CHAT_HANDOFF.md` 修改和未跟踪的归档规格，均原样保留。
`docs/design.md` 缺失；交接文件已完整阅读；无实验 tmux，仅用户会话。

## 1. 诊断函数（对应规格 0.1）

“实际存在”仅表示实现存在，不表示已完成独立包迁移或新 parity 验证。
同名函数缺失、等价能力存在时分别说明，避免把接口草图当成实现。

| 项目 | 状态 | 实际路径与行号 | 口径 / 耦合 / 现有测试 |
|---|---|---|---|
| `effective_support` / κ | 实际存在 | `analysis/diagnostics/metrics.py:12` | `exp(H)/N`；输入应已是条件分布，入口没有自动归一化/强制 FP32；`tests/test_diagnostics.py:71`。 |
| 排序 Gini | 实际存在 | `analysis/diagnostics/metrics.py:18` | 排序秩公式；也有接收质量和 head 统计版本，不能默认 epsilon/缺失口径一致。 |
| `weight_stats`：entropy/top1/DC | 实际存在，规格位置不准确 | `analysis/affinity_repair.py:61` | 返回归一化熵、top1、`N*min(w)`、正类对 Jaccard、fallback 比例；按图像内正类平均，依赖 labels。 |
| `analysis/weight_stats.py` | 实际存在，但没有同名 `weight_stats` 函数 | `analysis/weight_stats.py:6` | 此文件实现 `normalized_entropy`、`all_product_weights`、`positive_image_mean`（后两者分别在行10、16）。 |
| `class_token_affinity` / ρcc | 实际存在 | `analysis/diagnostics/metrics.py:25` | 正类非对角余弦、逐类/全类输出；没有正类对时为 NaN；`tests/test_diagnostics.py:97`。 |
| `patch_norm_stats` | 实际存在 | `analysis/artifact_probe.py:31` | patch 范数/图像中位数、`hi=3.0`、p99/p50；底层输入已是 patch tensor。 |
| `attractor_stats` | 实际存在 | `analysis/artifact_probe.py:40` | 对 P2P 的 query 轴求和得到 key 接收质量，返回 Gini/top-share/k；不要颠倒方向。 |
| `symptom_overlap` | 实际存在 | `analysis/artifact_probe.py:52` | 两组 top-k 支持集 Jaccard；`k=max(1,round(topk_frac*N))`。 |
| `lowinfo_score` 能力 | 实际存在，实际名称为复数 | `analysis/artifact_probe.py:61` | `lowinfo_scores`：RGB 灰度梯度、默认方形 grid=28、固定 top .01/ratio>3；并非通用语义背景判据。 |
| γ / class-agnosticism 能力 | 实际存在 | `analysis/head_heatmap.py:15` | `head_statistics` 中实现，gamma 在行38；先按行质量缩放再求 L2 余弦，单标签图缺失，不填零；`tests/test_head_alpha.py:35` 覆盖极小概率。 |
| 独立同名 `class_agnosticism` API | 只在计划文档中 | `docs/MCTformerPlus_HeadAxis_and_Alpha_Plan.md:120` | 参考代码与实装的缺失值/极小概率处理并不逐字相同；不能替代上一行的实际实现作为已验证来源。 |
| 径向功率谱 | 实际存在 | `analysis/diagnostics/spectral.py:20`；`analysis/diagnostics/spectral.py:48` | L2 token→空间去均值→Hann→二维 FFT→径向 bin 均值；网格两个方向均要求至少28。 |
| `high_freq_ratio` | 实际存在 | `analysis/diagnostics/spectral.py:53` | 径向 bin 均值的高频占比，不是按环带频点数加权的总能量占比；零谱为 NaN。 |
| `correlation_length` | 实际存在 | `analysis/diagnostics/spectral.py:59` | 来自真实二维逆功率变换；固定 fit_range/R² 条件，失败为 NaN；`tests/test_diagnostics.py:121`。 |
| 探针状态恢复 | 实际存在（补充） | `analysis/artifact_probe.py:77`；`analysis/diagnostics/state.py:11` | 前者恢复各模块 training；后者还恢复 RNG/TF32。不是同一保障范围，提取前需选定。 |

## 2. 数值与聚合（对应规格 0.2）

| 项目 | 状态 | 路径与行号 | 核查结果 |
|---|---|---|---|
| `aggregate_p2p` | 实际存在 | `models/mctformer_plus.py:36` | 层集 all/last3/last6；sum/mean/rownorm_mean；幂次和对称化；已有 patch_slice 参数；FP64 参考路径保留。 |
| `propagate_c2p_weights` | 实际存在 | `models/mctformer_plus.py:67` | `w @ P.T`，floor none/min/mean、beta、零质量 fallback；已有 FP32/FP64 分支。 |
| `c2p_spatial_weights` | 实际存在 | `models/mctformer_plus.py:1303` | mean/product 与 last3/all；依赖实例的布局/foreground slice/patch-first 配置。 |
| attention 精度转换点 | 实际存在，规格行号已过时 | `models/tgca.py:129`；`models/tgca.py:135`；`models/vit.py:190` | 行129是 logits 转 FP32；行135才是普通路径概率转回原 dtype；新增路径在转回前采集 FP32 C2P，行194仍保留原生 attention 转型。 |
| pooling 精度转换点 | 实际存在，规格行号已过时 | `models/mctformer_plus.py:1321`；`models/mctformer_plus.py:1354` | 普通路径收到 half/bfloat16 后转 float；可选 `c2p_pooling_fp32` 路径接收 pre-cast 记录。规格中的行1312不是当前转换点。 |
| 对数域跨层连乘 | 实际存在，不是待从零实现 | `models/mctformer_plus.py:1327`；`analysis/weight_stats.py:10` | 先概率域 head 均值，再 `sum(log)` 和 spatial softmax；epsilon 是 dtype tiny，不是规格草案的固定值。 |
| `log_domain_aggregate(mats,weights,tau,eps)` 通用 API | 只在归档规格中 | `docs/TGCA_Archive_Extraction_Spec.md:193` | 现存特化 log-product 无通用 weights/tau API；上述特化实现可作为特定参数下的参考，不能说所有聚合都未实现。 |
| `renorm_chain_product` / 逐层重归一化连乘 | 只在归档规格中 | `docs/TGCA_Archive_Extraction_Spec.md:190` | 在本仓库研究代码中未找到该 API 或对应逐层乘后重归一化循环；与现存一次性 log-product 区分。 |
| `underflow_telemetry(mats,layout)` 通用 API | 只在归档规格中 | `docs/TGCA_Archive_Extraction_Spec.md:198` | 没有统一返回质量/min/clamp/零行的函数；有散落的特化审计，见下一行。 |
| 下溢审计片段 | 实际存在（部分能力） | `analysis/head_alpha_preflight.py:98`；`experiments/ablations/validate_c2p_fp32.py:84` | 包含概率 half round-trip、零行、质量/TV 等；不是已经可安装的布局无关遥测模块。 |

## 3. 扫描与统计（对应规格 0.3）

| 项目 | 状态 | 路径与行号 | 核查结果 |
|---|---|---|---|
| 逐图共享前向、遍历配置 | 实际存在，核心不在 runner 目录 | `analysis/affinity_repair.py:265`；`analysis/affinity_repair.py:279` | 同一 view 的 features/P 在配置间复用；`experiments/ablations/run_affinity_repair.py` 负责调度。 |
| 新版共享前向扫描 | 实际存在 | `analysis/head_alpha.py:148`；`analysis/head_alpha.py:171` | α/H/控制/top-m 使用同一 scan；按8个配置分块处理，不是每配置跑 backbone。 |
| `ConfigSweep` 独立通用类 | 只在归档规格中 | `docs/TGCA_Archive_Extraction_Spec.md:222` | 现有是领域绑定扫描函数，未找到规格同名类。 |
| 多阈值混淆矩阵能力 | 实际存在 | `tools/evaluate_cam_threshold_grid.py:99`；`analysis/affinity_repair.py:259` | 阈值直方图→多阈值 confusion；配置 totals 与逐图 fixed 分开存储；现用60档，不是规格存储示例的50档。 |
| 在线累加器 | 实际存在 | `tools/evaluate_raw_cam_streaming.py:61` | `OnlineCamEvaluator` 依赖 mask目录/id列表/原始 CAM payload，尚不是通用 `ConfusionAccumulator`。 |
| 配对 bootstrap：图像均值 | 实际存在 | `analysis/c2p_pooling_attention.py:74` | 有效值交集、图像等权、multinomial；默认 repetitions=5000，默认 seed=20260914；artifact 调用时传20260916。 |
| 配对 bootstrap：全数据集 mIoU | 实际存在 | `analysis/affinity_repair.py:211` | 先重采样图像 confusion 再算 mIoU，不能与“每图 IoU 均值”互换；5000次，SEED来自 artifact模块。 |
| 配对 bootstrap / crossfit 新实现 | 实际存在 | `analysis/head_alpha_statistics.py:50`；`analysis/head_alpha_statistics.py:65` | seed=0，足够统计量/固定折内重采样与重选参数；并非所有版本相同随机流/估计量。 |
| manifest 字段集合 | 实际存在，多版本 | `results/artifact_probe/20260916-voc-s0/manifest.json:1`；`results/head_alpha/20260918-voc/full-r2/manifest.json:1` | 见下节完整字段清单；并不存在一个已经统一的 schema。 |
| probe-set hash | 实际存在（补充） | `analysis/diagnostics/probe.py:119`；`results/diagnostic_trajectory/20260916-voc-ab-s012/gwrp_s0/diagnostics/manifest.json:37` | 已有 `probe_set_sha256`，不能把它说成所有 TGCA manifest 都从未记录；各实验覆盖不同。 |

### 3.1 实际 manifest 字段（不创建或选择新 schema）

Artifact（2809 bytes）：

```text
git_sha, config, source_sha256_before, seed, bootstrap, unit, num_images,
multi_images, transform, precision, semantic_gt_loaded, gate_thresholds,
example_ids, M6, model_mode_restored, checkpoint_policy, source_sha256_after,
source_integrity_unchanged
```

A2（452120 bytes）：`results/affinity_repair/20260916-voc-s0/a2/manifest.json:1`。

```text
stage, git_sha, source_sha256_before, config, precision, seed, bootstrap,
checkpoint_policy, semantic_gt_loaded, mask_sha256_before, CAM_protocol,
repair_CAM, selection_bias, all_product_native_cam_max_abs_error,
all_product_historical_native_delta, gwrp_native_cam_max_abs_error,
gwrp_historical_native_delta, mask_sha256_after, source_sha256_after,
source_integrity_unchanged
```

Head/alpha（451694 bytes）：

```text
git_sha, precision, autocast, tf32, training, seed, bootstrap, scope, limit,
preflight, threshold_grid, epsilon, rownorm_control, multiplicity, gamma,
crossfit, fold_class_counts, source_sha256_before, source_sha256_after,
source_integrity_unchanged
```

前两个用 precision 字符串描述 autocast；第三个有独立 bool。三者都不含
probe_set_sha256，但上表另一个 diagnostics manifest 包含。没有把不同实验
的元数据假装成同一份完整记录；新 schema 的必需/可选字段留待后续确认。

## 4. 指定结果文件与 SHA256（对应规格 0.4）

以下仅核对文件并登记字节数/哈希，没有复制、解析后重存或修改数字。
存在的文件均低于 1 MB；缺失文件不伪造 SHA256。CSV 内容只读。

| ID | 状态 | 文件与行号 | bytes |
|---|---|---|---:|
| R1 | 实际存在 | `results/artifact_probe/20260916-voc-s0/decision.md:1` | 1302 |
| R2 | 实际存在 | `results/affinity_repair/20260916-voc-s0/ARTIFACT_AFFINITY_REPORT.md:1` | 598 |
| R3 | 实际存在 | `results/affinity_repair/20260916-voc-s0/a2/AFFINITY_SCREENING_REPORT.md:1` | 5440 |
| R4 | 实际存在 | `results/affinity_repair/20260916-voc-s0/stage_b_comparison.csv:1` | 552 |
| R5 | 不存在于指定 glob | `results/voc_epoch_checkpoints/20260916-gwrp-all-product-s0-s11-r2/**/epoch_metrics.csv`；期望来源 `docs/TGCA_Archive_Extraction_Spec.md:67` | N/A |
| R6 | 实际存在 | `docs/MCTformerPlus_C2P_Product_Affinity_Design_and_Results.md:1` | 18596 |
| A1 | 发现替代位置，尚未代替 R5 归档 | `results/voc_epoch_evaluation/20260916-gwrp-all-product-s0-s11/epoch_metrics.csv:1` | 89894 |
| A2 | 发现额外文件，不自动纳入范围 | `results/detach_channel/20260917-voc-s0-s11/epoch_metrics.csv:1` | 162774 |

```text
R1 8820309afdc508885cbc4696e1a691a7acfc2b79c30fb7411404d945fdb50393
R2 a7b0a3c19cf35e1061219decf393f032c31bce872f90f03f107de349997bc44d
R3 df6db4fd25c545ea9cb2b4e7e79ec14db781e10abc2cff82ef77f9442477bfa3
R4 dc87821abb2d451590b30c2982d04e625cc8ad162fe0f254cea8a0ae0eac9011
R5 MISSING — no matching file, no hash
R6 a3019a80164667ded91019f4fd5cab68b947ef4b4d0d54d66e52aa4f884b1417
A1 7da4622d6cb74be6a48ecdd874c2b2377f87c8b748c92d80508d7385967e92f8
A2 b384c3cf3223ede850a1c8dd48bf11b2026519fb17b09e4b0e4ddb408d9dd5f1
```

补充元数据 SHA256（仅登记）：

```text
results/artifact_probe/20260916-voc-s0/manifest.json
1a752664c259702b3bd39682c385f426c4b054ec8e4d38211a21edec6863bca9
results/affinity_repair/20260916-voc-s0/a2/manifest.json
a0e223ec715f180c85a883fcfd0e850096789a35ecf83ed54213ca464292d36b
results/head_alpha/20260918-voc/full-r2/manifest.json
f341c80524889bb5808efc3ad0b858bddfd5087ae427c2da14c22bc586c26e49
```

## 5. 发现的不符 / 未解决的分叉

下面是来源之间的差异登记，不修改任何来源，不重新计算历史指标，不先行
决定新包应采用哪一种语义。未得到确认前不写 Phase 1 产品代码。

| ID | 规格描述 | 实际证据 / 待确认问题 |
|---|---|---|
| D1 | 用 κ 标注传播前后数值 | `docs/MCTformerPlus_C2P_Product_Affinity_Design_and_Results.md:178` 明确为 `H(w)/log(784)`；行182原文 `0.6020` / `0.9776`，行185原文 `0.2938` / `0.9719`。`analysis/diagnostics/metrics.py:12` 的 κ 则是 `exp(H)/N`。不能把原归一化熵表直接当 κ 黄金值。 |
| D2 | 常数场全部能量留在 DC | `analysis/diagnostics/spectral.py:28` 去均值，行39–43对常数场返回零谱；`tests/test_diagnostics.py:121` 断言常数零谱、E_hi/xi 缺失。保留旧实现 parity 与直接满足该新 DC 验收不是同一个功能。 |
| D3 | 10.07 pp 差异根因已证明仅是 FP16，而非动力学 | `docs/C2P_FP32_Precision_Repair.md:35` 明确只有小规模 smoke，行46–47明确数值修复不能确认 seed 差异原因。规格 `docs/TGCA_Archive_Extraction_Spec.md:265` 的唯一根因表述与该证据边界不一致；本轮不裁定或改写。 |
| D4 | 两宿主符号复现替代多重比较校正 | `docs/Head_Alpha_Execution_Notes.md:55` 明确这不校正72次比较；两种方法论表述冲突，不能同时作为已验证统计保证。 |
| D5 | log-space“完全免疫下溢”，概率域 head 均值“绝对安全” | 现有 pre-cast 修复 `models/vit.py:190` / `models/mctformer_plus.py:1321` 正是为了避免概率先归零再转 float；log-space不能恢复已有零值，稳定 softmax也不能保证任意极端输入的所有输出非零。通用模块的保证范围需确认，不能照搬绝对性断言。 |
| D6 | 12层且每层最大值1e-3足以定义 FP32 零值压力测试 | 规格 `docs/TGCA_Archive_Extraction_Spec.md:213` 同时写出典型乘积 `10^-36` 与 FP32 最小正规数 `1.18×10^-38`；最大值约束本身不足以保证朴素乘积归零。须明确合成分布/零元素政策，不以改容差解决。没有在本轮新增数值实验。 |
| D7 | `renorm_chain_product(...eps=1e-8)` 无条件逐值等价且保持约1/N | 稀疏/零质量行、分母触及 clamp、集中分布时需额外定义；现存 propagate 有显式 fallback，不能未经批准将其复制成通用连乘的零行政策。 |
| D8 | bootstrap 应同 seed 同输入同 CI | 目前图像均值、全局 confusion mIoU、折内重选参数三种估计量并存，默认 seed 也不同。应由用户确认 parity 所针对的函数/估计量，不能任选一个后宣称与全部 TGCA 一致。 |
| D9 | epoch≥10 四条相关性为 `+0.65 ~ +0.95` | 已找到逐 epoch 原始 CSV，但 `results/voc_epoch_evaluation/20260916-gwrp-all-product-s0-s11/EPOCH_TRAJECTORY_REPORT.md:1` 无此相关性表；相关结果文件检索未找到对应四条预计算值。规格本身之外尚缺可引用原始汇总，不在本轮重新计算补齐。 |
| D10 | 所有可迁移函数都只受 token layout 耦合 | 还存在 input 归一化/NaN、FP64参考、RGB/grid默认、FFT去均值/window、固定21类 confusion、dataset读取和 pandas/import 等耦合；见上表。TokenLayout 有必要但不是全部工作。 |
| D11 | 新包仅 torch/numpy，scipy可选；`results()` 返回 pandas.DataFrame | 规格中存在依赖要求分叉；需确认返回普通记录还是另设 pandas 可选依赖。 |
| D12 | graph 可选，同时 MIGRATION 排除 affinity 修复 | 规格 Phase 1/2 允许提取 aggregate/propagate，Phase 7又要求不迁移 affinity 修复；需明确通用图算子与领域方法的归档边界。 |
| D13 | `TokenLayout.mctformer_plus()` 与代码中不得出现 mctformer | 工厂方法示例与最终全文搜索验收不一致；`n_cls/n_class/n_register` 同时存在时的具体排列也尚未定义。需决定验收例外/布局约定，不能靠搜索“通过”隐藏问题。 |
| D14 | 独立仓库或 TGCA 同级包 | 当前仓库指令要求开发在 TGCA 内；规格提供多个目标位置但没有选定。Phase 1 前需确认目录与是否另建 Git 仓库；本轮不创建任何包目录/新仓库。 |
| D15 | Artifact 门可概括为 NO-GO | `results/artifact_probe/20260916-voc-s0/decision.md:13` 对 gwrp 为 `[True, True, True]`；行14、15对另外两组为 `[True, True, False]`。不能省略 checkpoint 条件或把“省下两周”当成有日志证明的测量。 |
| D16 | A2“零结果”与最终状态门 | R2行5写 `Stage B gate passed: True`；R3行9原文 delta=`4.2947013945138224e-08`，CI=`-4.405026232667808e-09` 至 `1.0956406383544334e-07`。这些字段同时保留，不把布尔门改成统计显著性，不改精度或换算单位。 |
| D17 | 复制现有数学同时入口全部强制 `.float()` | `models/mctformer_plus.py:49`、行76明确保留 FP64，而多项 stats 按输入 dtype 运算。需要约定 parity 输入 dtype、零行/epsilon、atol与rtol；不在失败时擅自调整容差。 |

## 6. Phase 0 验收与下一步门

- 规格0.1–0.3每项已有明确状态、实际路径/行号；复合项已拆开，不再保留问号。
- 规格0.4的全部明确文件已核验；缺失 glob 与替代实际位置分别记录，未静默替换。
- 文件大小/源 SHA256 已登记；实验数字仅逐字引用，没有重新运行或舍入。
- 已追加本轮 Phase 0 日志到 `docs/ARCHIVE_LOG.md`。
- 本轮没有运行独立包 parity 或安装测试，因为包尚未创建；这些不是“通过”或“失败”，而是尚未执行。

**等待用户确认清点结果，以及后续归档位置、存在冲突的口径/验收边界。
在确认前不进入 Phase 1，也不创建 FINDINGS/METHODOLOGY 来替用户裁决上述差异。**
