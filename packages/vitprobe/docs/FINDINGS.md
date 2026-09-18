# 可迁移观察与证据边界

这是归档，不是新实验。下列数字逐字引用 evidence/ 中的原文件，没有重新
计算或调整舍入。原件中的运行配置、checkpoint 路径、命令和历史 Git SHA
保持原样；归档版本不能冒充历史训练版本。来源文件间的差异不作裁决。

## 空间分布与归一化

来源：`../evidence/MCTformerPlus_C2P_Product_Affinity_Design_and_Results.md`，§6。

| 模型 / 指标 | 传播前 | 传播后 | Δ（后−前）及 95% paired image bootstrap CI |
| --- | ---: | ---: | --- |
| last3-product-affinity / 归一化熵 | 0.6020 | 0.9776 | +0.3756 [0.3695, 0.3819] |
| all-product-affinity / 归一化熵 | 0.2938 | 0.9719 | +0.6781 [0.6724, 0.6838] |
| all-product-affinity / top-1 mass | 0.4651 | 0.0129 | −0.4522 [−0.4623, −0.4420] |

这里是 H/log(N)，不是 κ=exp(H)/N。来源明确说传播后并非严格 GAP。
可迁移的是同时检查归一化、空间集中度和输出指标的规程，不是“传播必然崩塌”
的普遍结论。和归一化保留的常数分量，与 min-max 消去常数平移的数学性质
不同；不能从某个实验的数值不变推出任何图传播都无效。

来源：`../evidence/AFFINITY_SCREENING_REPORT.md` 的 all_product/min 行：
delta=`4.2947013945138224e-08`，CI 下界=`-4.405026232667808e-09`，
上界=`1.0956406383544334e-07`。同一归档的
`ARTIFACT_AFFINITY_REPORT.md` 写有 `Stage B gate passed: True`。
两者照存。程序门的 True 不等于统计显著，更不等于方法成功。

## 数值精度边界

来源：`../evidence/C2P_FP32_Precision_Repair.md`。实际边界是 FP32 softmax
概率先转 FP16、后转 FP32 做聚合；后一次转换不能恢复已经变成零的值。
这与“直接在 FP16 中连乘”不是同一段代码。

可迁移的是审计最早丢失信息的位置，检查概率和梯度的 dtype，而不是只看
聚合输出 dtype。对数域计算减轻连乘下溢，不能恢复输入的零，也不保证任何
极端动态范围下的输出都非零。概率域 head 均值和 log 域均值不是同一个算子。

## 高范数与吸引质量不必重合

来源：`../evidence/decision.md`，原表如下。

| Checkpoint | M2 fraction | M4 top1% mass | M5 Jaccard | All gates |
|---|---:|---:|---:|---|
| gwrp | 0.013429 | 0.081556 | 0.323059 | [True, True, True]: register worth future consideration |
| all_product | 0.005396 | 0.063604 | 0.007630 | [True, True, False]: NO-GO register for this checkpoint |
| all_product_affinity | 0.016951 | 0.088546 | 0.066659 | [True, True, False]: NO-GO register for this checkpoint |

来源随机独立集合期望 Jaccard 为约 `0.00547`（k=8），不是 .01。
可迁移的是分别测量高范数、接收质量及其交集；不能把其中一个当另一个的替代。
三组 checkpoint 的观察不能证明训练机制因果，也不能直接外推到其他模型大小。
RGB 低梯度不等于语义背景。

## 不升级为已证实结论的两项

- 规格中的 epoch≥10、四条 Spearman `+0.65 ~ +0.95`：未找到预计算汇总。
  保留原 epoch CSV，但本归档不重新计算补齐，不能把规格文字当实测证据。
- 规格中的 seed 差异只由 FP16 引起、不是优化动力学：来源修复报告明确没有
  完成全程匹配重训来建立该因果关系。保留数值缺陷证据，不宣称其解释全部差异。

## 频谱口径

按用户确认保留原实现：token L2→空间去均值→Hann→FFT，常数场为零谱，
高频占比与相关长度缺失。高频占比基于等权径向 bin 均值，不是环带总能量。
这些定义可迁移，但阈值与网格限制不能不加说明地外推。
