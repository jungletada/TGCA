# vitprobe

从现有研究代码增量提取的 ViT 诊断包。运行时仅依赖 PyTorch 与 NumPy，
不加载模型、checkpoint 或数据集，不改变训练行为。

当前已实现：TokenLayout、空间权重统计、正 query 关系、高范数/吸引子
描述、去均值频谱、相关长度、探针状态隔离、通用图算子、两种图像配对
bootstrap 和共享前向扫描。提取函数有原实现逐值对照；新扫描框架有合成
行为测试。新的数值聚合模块尚待单独确认，不宣称全计划完成。

```sh
python -m pip install --no-deps --no-build-isolation -e .
python -m pytest tests -q
TGCA_PATH=/path/to/source/checkout python -m pytest tests/test_parity_tgca.py -x -q
```

未设置 TGCA_PATH 时原实现对照明确跳过，其余测试仍可独立运行。
完整参考环境是 Python 3.9 / torch 2.1.0 / numpy 1.26.3；没有升级这些依赖。

```python
import torch
from vitprobe.stats import conditional_attention, effective_support

p = conditional_attention(torch.rand(2, 4, 784))
kappa = effective_support(p)  # [images, queries]
```

输入已切出的 patch/query 张量，无固定类别数。FP32 对照与原数学一致；
lowinfo_scores 保留原 RGB 定义，并支持 layout 指定的矩形网格。频谱保留最小28网格约束，
去均值后的常数场为零，不把未定义量填为零。

- [布局、接口与迁移范围](docs/MIGRATION.md)
- [当前验证状态与精确命令](docs/VALIDATION_INTERFACES.json)；旧 `VALIDATION.json` 保留首批提取的历史记录。
- [来源与限制](docs/FINDINGS.md)
- [方法规程](docs/METHODOLOGY.md)
- [判定门模板](templates/gates.yaml)、[新运行 manifest 模板](templates/manifest.schema.json)
- `evidence/`：小体积原文快照；数字和文件字节不改动。文内相对链接属于
  原仓库上下文，来源映射与哈希单独保存，不为修复链接而改写原件。

此提取未授予新的第三方代码许可证；对外发布前应核查原仓库许可。
