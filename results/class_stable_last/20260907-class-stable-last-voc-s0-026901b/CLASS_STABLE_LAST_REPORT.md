# Class-Stable LaST Pooling matched VOC result

| Model | Class mAP (%) | Patch mAP (%) | Class val loss | Patch val loss | Fixed@0.45 CAM mIoU (%) | Best CAM mIoU (%) | Best threshold | FG precision@0.45 (%) | FG recall@0.45 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original MCTformer+ | 92.906316 | 93.257697 | 0.050633 | 0.049780 | 70.063058 | 70.339707 | 0.480000 | 80.735325 | 85.816898 |
| PatchFinalLN MCTformer+ | 92.916190 | 93.313498 | 0.049249 | 0.048791 | 69.190857 | 70.024359 | 0.500000 | 78.761943 | 87.035904 |
| Last-MCT (pool before classifier) | 88.877369 | 90.151890 | 0.063304 | 0.060058 | 58.878289 | 62.215372 | 0.560000 | 62.883366 | 84.410069 |
| Class-Stable LaST Pooling | 91.923881 | 92.200237 | 0.054099 | 0.053383 | 62.205797 | 64.967041 | 0.540000 | 69.369213 | 83.780445 |

The best threshold is reported from the same `0.00:0.01:0.59` sensitivity sweep. Precision and recall use the prespecified fixed threshold `0.45`.

## Reproducibility

- Git commit: `026901b64fe25425485080661b86c47128973354`
- Class-Stable LaST checkpoint SHA256: `f2c7fcbef0e5f215e0fb7d9c6793198acf6b920011c43a68495d6d2de3f7cca2`
- Configuration: `config.json`
- Exact commands: `exact_commands.sh`
- Test status: `training_logs/tests_complete.txt`
- Original summary SHA256: `0b845989429919ce7336f8e9525a7f41d69ada9536f5cde0178438b5310054e6`
- PatchFinalLN summary SHA256: `38f514939ffcd6ab420c0f4b19270ecee59911f87f73ca3d6d68315522a993b5`
- Previous Last-MCT summary SHA256: `683c13a493bc4ecb6e0c21750787ee5a1cc76f83859d8ed6ce22b8d8c313646b`
