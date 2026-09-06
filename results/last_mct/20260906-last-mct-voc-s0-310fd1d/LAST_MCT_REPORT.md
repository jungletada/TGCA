# Last-MCT matched VOC result

| Model | Class mAP (%) | Patch mAP (%) | Class val loss | Patch val loss | Fixed@0.45 CAM mIoU (%) | Best CAM mIoU (%) | Best threshold | FG precision@0.45 (%) | FG recall@0.45 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original MCTformer+ | 92.906316 | 93.257697 | 0.050633 | 0.049780 | 70.063058 | 70.339707 | 0.480000 | 80.735325 | 85.816898 |
| PatchFinalLN MCTformer+ | 92.916190 | 93.313498 | 0.049249 | 0.048791 | 69.190857 | 70.024359 | 0.500000 | 78.761943 | 87.035904 |
| Last-MCT | 88.877369 | 90.151890 | 0.063304 | 0.060058 | 58.878289 | 62.215372 | 0.560000 | 62.883366 | 84.410069 |

The best threshold is reported only from the same fixed `0.00:0.01:0.59` sensitivity sweep. Precision and recall use the prespecified fixed threshold `0.45`.

## Reproducibility

- Git commit: `310fd1d917948b56073eb9d781cb528a1e0fea0f`
- Last-MCT checkpoint SHA256: `3ed451f1062b5746b1cbf36a380c1643968d4a5938cad35fe0ed27ed0d69fcd3`
- Configuration: `config.json`
- Exact commands: `exact_commands.sh`
- Test status: `tests.txt` (`51 passed in 8.96s`)
- Original source summary SHA256: `0b845989429919ce7336f8e9525a7f41d69ada9536f5cde0178438b5310054e6`
- PatchFinalLN source summary SHA256: `38f514939ffcd6ab420c0f4b19270ecee59911f87f73ca3d6d68315522a993b5`
