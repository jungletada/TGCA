# ViT 诊断工具归档日志

## 2026-09-18 — Phase 0 完成，等待确认

### 做了什么

- 在 LHR `/home/peng/code/TGCA`，核对 cwd、Git status/log、tmux，完整阅读
  `docs/CHAT_HANDOFF.md` 和 `docs/TGCA_Archive_Extraction_Spec.md`。
- 基准 commit：`7edd58d`。仅用户 tmux 会话，无实验队列；`docs/design.md` 缺失。
- 使用 `rg --files` / `rg -n` 定位诊断、聚合、扫描与统计实现；使用 `nl -ba`
  核对定义和来源行号；使用 `stat` / `sha256sum` 核对指定结果及发现的替代位置。
- 生成 `docs/ARCHIVE_INVENTORY.md`：逐项存在性、路径/行号、精度和统计口径、
  manifest 字段、结果大小与 SHA256、不符事项及后续确认门。
- 完成文档只读校验：75处路径/行号引用有效，7个结果文件大小与 SHA256
  匹配；两个新增文档无行尾空白，`git diff --check` 通过。既有 handoff 与
  归档规格的 SHA256 和本轮启动时一致。

### 与规格不符

- κ/Gini/class-token affinity、谱与相关长度、γ、log-product 已有实装。
- `weight_stats` 在 affinity 分析文件；`lowinfo_scores` 是实际函数名；
  指定 epoch_metrics glob 没有匹配文件，评估目录存在对应 CSV。
- 多项拟归档数学/结论与来源不同：归一化熵不等于 κ、去均值谱的常数场为零、
  seed 唯一根因尚未建立、跨宿主符号不等于多重比较校正等。
- 详见清单 D1–D17；没有修改规格或源结果，也没有自行选择其中一方作为新包定义。

### 跳过 / 未执行

- 严格遵守用户确认门：Phase 1–7 全部尚未启动。
- 没有新建 vitprobe 包/仓库、提取实现、安装依赖、运行训练/推理或重新计算指标。
- 没有复制或读取 checkpoint 张量，没有复制任何大结果文件，没有 `git add -f`。
- 没有执行包 parity；不存在新包测试结果，不把原 TGCA 测试当成迁移已通过。
- 未选定可选 graph 模块、bootstrap canonical实现、谱策略、包位置或依赖分支。
- 未提交或 push；当前只新增两个 Phase 0 文档。既有 CHAT_HANDOFF 修改和未跟踪
  归档规格原样保留。其他模型、实验、数据加载代码及源结果未改动。

### 只读核查命令样例

```sh
pwd
git status --short --branch
git log -8 --oneline --decorate
tmux ls
rg -n '^def (effective_support|gini|class_token_affinity)' analysis/diagnostics/metrics.py
nl -ba analysis/diagnostics/spectral.py
rg --files -uuu results/voc_epoch_checkpoints/20260916-gwrp-all-product-s0-s11-r2
sha256sum results/artifact_probe/20260916-voc-s0/decision.md
sha256sum results/affinity_repair/20260916-voc-s0/stage_b_comparison.csv
stat -c '%s %n' results/voc_epoch_evaluation/20260916-gwrp-all-product-s0-s11/epoch_metrics.csv
git diff --check
```

### 下一步

等待用户确认 `ARCHIVE_INVENTORY.md`，再明确分叉后进入 Phase 1。
本日志后续按 Phase 追加，不覆盖已记录的清点结论或历史条目。

## 2026-09-18 — 用户确认后的第一批提取

### 已确认的选择

- 用户确认 Phase 0 后，又明确确认包位于 `packages/vitprobe/`、沿用当前
  Git；保留现有数学口径；原文数字与证据边界分开记录。
- 尚未回答本轮提出的接口/新数值边界问题，因此没有擅自实施那些分叉。

### Phase 1 — 部分完成

- 新增 src-layout、pyproject 和 README；运行时仅 torch/numpy。
- tgca-repro editable 安装成功；`--no-deps --no-build-isolation`，未升级依赖。
- TokenLayout 的混合 token 顺序、命名检查例外仍待本轮接口方案确认。
  未以空模块或占位函数冒充完成。

### Phase 2 — 无争议诊断子集完成，整体未完成

- 提取 stats/relations/artifacts/spectral/state：保留原数学，诊断入口升 FP32。
- 17项初始测试全部通过，其中11项调用原仓库函数做合成 FP32 对照，
  atol=1e-7、rtol=0、equal_nan=True；没有失败后调整容差。
- 常数谱采用用户确认的零谱口径；未添加不兼容的 DC 验收。
- probe 状态恢复沿用已有 isolated_eval，模块状态/RNG/TF32 都恢复。
- 尚未迁移图算子、布局适配、训练 C2P pooling；没有改动源函数。

### Phase 3 / Phase 4 — 暂缓

- 新数值模块的零行、clamp 与 FP64 策略待确认。
- bootstrap 的均值/全局 confusion 两种接口及扫描返回类型待确认。
- 本轮未新算 bootstrap、未重算任何历史指标，也未运行训练/模型推理。

### Phase 5 / Phase 6 — 文档与证据部分完成

- 新增 METHODOLOGY.md、FINDINGS.md，明确多重比较、κ/熵、因果和缺失
  相关性证据的限制，不照搬规格中超出原来源的结论。
- 按原字节归档8个小体积 md/CSV文件，总123434 bytes；逐文件 <=1MB。
  SHA256和原路径写入 evidence_manifest.json；epoch CSV 使用清点发现的
  评估目录位置，未伪造原 glob 下的文件。没有复制 checkpoint。
- 模板与整体迁移文档仍待接口完成；Phase 7 未完成。

### 原件完整性与执行记录

- 335项源代码/选定文档结果哈希前后相同（更新 handoff 前验证），
  包括既有 CHAT_HANDOFF 内容、用户规格以及本轮涉及的全部 tracked
  models/analysis/experiments/dataloaders/tools/tests 文件。
- 日志根目录：`results/vitprobe_archive/20260918-extraction/`。
- 安装：`/home/peng/anaconda3/envs/tgca-repro/bin/python -m pip install --no-deps --no-build-isolation -e packages/vitprobe`。
- 初始测试：`TGCA_PATH=/home/peng/code/TGCA OMP_NUM_THREADS=2 /home/peng/anaconda3/envs/tgca-repro/bin/python -m pytest packages/vitprobe/tests -x -q --junitxml=results/vitprobe_archive/20260918-extraction/diagnostics.xml`。
- 没有 commit/push；既有源文件与实验状态不变。

### 收尾验证（仍为部分提取）

- 加入来源快照完整性与运行时 import 独立性测试后，有源对照19 passed，
  无源环境8 passed / 11 explicitly skipped；无失败、无容差变更。
- 后两次完整命令及统计保存于 `packages/vitprobe/docs/VALIDATION.json`；
  原始日志为同一日志根目录的 with_source.log / standalone.log 及 XML。
- 向 CHAT_HANDOFF 追加本轮状态，保留所有既有段落。未完成项依然未完成，
  不将源码可用或合成测试通过当成全计划最终验收通过。

## 2026-09-18 — 第二批：已确认的接口提取完成

### 用户确认与范围

用户明确确认：保留布局工厂名称；额外 token 按 cls→class→register；
仅通用图算子；bootstrap 两种估计量分开；普通记录为默认、pandas 可选。
本轮没有收到新 numerics 的零行/eps/FP64 策略确认，Phase 3 继续暂缓。

### Phase 1 — 布局与已批准接口完成

- TokenLayout 支持 class-first/patch-first、矩形网格、cls/class/register
  分开切片；零长度组不会自动替换成其他查询组。
- mctformer_plus/dinov3 仅创建元数据，不加载模型；按批准记录名称检查例外。
- 新增 dataframe 可选依赖；普通运行仍仅 torch/numpy，不升级现有依赖。

### Phase 2 — 已批准诊断与通用图提取完成

- aggregate_p2p / propagate_weights 保留原方向、顺序累加、floor/damping、
  eps/fallback、原有 FP64 参考分支；只将 token 切片改为布局对象。
- lowinfo_scores 新增 layout 矩形网格入口；默认及方形布局与原实现对照。
- 不迁移模型 C2P pooling、GWRP、affinity 修复流程或训练/损失代码。
- 所有提取核心函数有逐值对照；新增包装接口使用独立行为测试。

### Phase 4 — 完成

- paired_image_bootstrap：图像内先归约、两模型有限值交集、默认5000次。
- paired_confusion_bootstrap：先配对重采样整图 confusion，再算全局 mIoU；
  不改成每图 IoU 均值。默认 seed20260916，均值估计量原 seed20260914
  仍可显式传入；两种估计量均与对应原函数同输入/seed做5000次 CI 对照。
- 通用 ConfusionAccumulator / ConfigSweep：int64累加，每样本一次forward，
  配置内独立readout。默认返回记录；pandas只在显式要求DataFrame时导入。
- 阈值直方图和混淆矩阵指标也对照原实现。新 callback 框架只有合成测试，
  未重新运行历史扫描；totals本身不足以重建图像bootstrap，文档明确说明。

### Phase 5 / Phase 6 / Phase 7

- 添加 gates.yaml 和新运行 manifest.schema.json；原始 manifest 不修改。
  模板 JSON语法/内部引用已检查，未安装外部 JSON Schema 验证器，不虚报
  完整 schema-validator 验证。
- 补齐 MIGRATION.md，列出三种布局、每个诊断的通用用途、两种统计量、
  不迁移项及新框架/历史证据的限制。既有 FINDINGS 数字不重算、不改动。
- 非语义布局 smoke 已跑通所有当前提取诊断、图算子、统计和扫描入口。
- Phase 3 未实现，故 Phase 7 的全包最终验收及完整数值模块结论仍待完成。

### 测试 / 完整性 / 交接

- 第一轮143 passed；加入隔离导入/名称例外/模板检查后146 passed，无失败。
- 无 TGCA_PATH：31 passed / 115 skipped（仅原实现对照不可用）。
- 所有 parity atol1e-7/rtol0，未放宽；原库和pandas被屏蔽时核心独立运行。
- 336项受保护文件哈希前后相同；8份紧凑快照还与原文件逐字节核对一致。
- 未训练、未读/复制checkpoint、未重算历史指标、未提交或push。
- 精确命令及状态：`packages/vitprobe/docs/VALIDATION_INTERFACES.json`；
  原实现/提取代码哈希：`packages/vitprobe/implementation_manifest.json`；
  日志：`results/vitprobe_archive/20260918-interfaces/`。
- CHAT_HANDOFF 新增本轮状态，保留先前所有段落和用户既有修改。
