# COCO → Diagnostic Probe 定时接续

配置日期：2026-09-15，主机 LHR，日本时间（UTC+09:00）。

首次检查为 **2026-09-16 00:00**。若 COCO 尚未完成，每隔 **30 分钟**自动检查一次并记录日志，不需要交互确认。不是在零点强制停止 COCO。

监控入口：`experiments/monitor_coco_to_diagnostics.py`（仅标准库，不建立 CUDA context）。

## 监控与接续目标

- 当前 COCO：`results/c2p_pooling/20260915-coco-all-product-s0-r2`。
- 监控 tmux：`mct-monitor-coco-to-diag-20260916`。
- 监控记录：`results/monitors/20260916-coco-to-diagnostics-v1/`，包括 `manifest.json`、`events.jsonl`、`status.json`、`commands.sh`。
- 监控标准输出：`results/monitors/20260916-coco-to-diagnostics-v1.monitor.log`。
- 下一实验 tmux：`mct-diag-voc-ab-s012-20260916`。
- 下一实验目录：`results/diagnostic_trajectory/20260916-voc-ab-s012`。
- 下一队列日志：监控记录目录中的 `diagnostic_queue.log`（启动后才出现）。

Diagnostic Probe 队列为 VOC 的 **GWRP / C2P-all-product × seeds 0、1、2**，共六组，按 seed 内先 GWRP 后 C2P 顺序执行。每组使用原匹配 45-epoch 训练配置，附加默认实现的逐 epoch probe 和 mini-CAM。不会自动启动未明确的单 CLS 对照、COCO diagnostic 或 3× 延长训练。

## 放行条件

COCO 必须有 `QUEUE_COMPLETE`，且训练、CAM、分类评估和结果报告齐全。监控只读复核 CAM 82,783 张、两组分类各 40,504 张、comparison 表、源文件与最终 checkpoint 哈希。仅看到 final checkpoint 或训练结束不会放行。

放行前还检查 main 分支、tracked-clean、空闲 GPU0（至少 30,000 MiB 可用）、至少 10 GiB 磁盘，以及下一 tmux/结果目录尚不存在。GPU 被占用或工作区尚未提交时继续等待；不抢占、不清理、不重启实验。

使用文件锁和独占启动标记防止重复监控/启动。允许后续仅文档变动的提交；若训练/分析 Python 或 shell 代码改变、COCO 失败或完整性校验失败，停止自动接续并将原因写入 `status.json` / `STOPPED`，不擅自切换代码或重试训练。脚本不向聊天界面发送定时消息，轮询结果保存在日志中。

接续成功后监控退出，训练在自己的 tmux 中继续；`LAUNCHED` 表示已经发出启动，不表示六组训练已完成。最终成功/失败以新实验目录的 `QUEUE_COMPLETE` / `QUEUE_FAILED` 为准。

## 启动命令

以下监控命令应只启动一次，且在独立 tmux 中运行：

```bash
/home/peng/anaconda3/envs/tgca-repro/bin/python -u -m experiments.monitor_coco_to_diagnostics \
  --watch-run results/c2p_pooling/20260915-coco-all-product-s0-r2 \
  --start-at 2026-09-16T00:00:00+09:00 \
  --interval-minutes 30 \
  --monitor-output results/monitors/20260916-coco-to-diagnostics-v1 \
  --next-output results/diagnostic_trajectory/20260916-voc-ab-s012 \
  --next-session mct-diag-voc-ab-s012-20260916
```

状态查看：

```bash
tmux ls
tail -n 10 results/monitors/20260916-coco-to-diagnostics-v1/events.jsonl
```

测试覆盖未完成/失败、完整结果审计、校验失败、重复启动、GPU/工作区等待，以及虚拟时钟下零点首次检查、00:30 再检查后只启动一次。现有 COCO 与 checkpoints 不作修改。
