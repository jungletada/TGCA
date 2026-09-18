# Artifact probe decision

Thresholds interpreted as proportions: M2 >= 0.005 (0.5%), M4 >= 0.05 (5%), M5 >= 0.2.
Gates use equal-image L12 means on the full validation set, not a selected subset or CI endpoints.
Each checkpoint is judged separately; no pooling of images/patches across checkpoints.
M6 uses RGB grayscale gradient, not semantic GT: low information does NOT imply background.
High-norm-specific M6 is missing if an image has no rho>3 patches; missing counts are reported.
k=round(0.01*784)=8. Exact random independent-set expected Jaccard is about 0.00547, not 0.01.
Image-clustered paired bootstrap 5000, seed20260916; no training-seed uncertainty.

| Checkpoint | M2 fraction | M4 top1% mass | M5 Jaccard | All gates |
|---|---:|---:|---:|---|
| gwrp | 0.013429 | 0.081556 | 0.323059 | [True, True, True]: register worth future consideration |
| all_product | 0.005396 | 0.063604 | 0.007630 | [True, True, False]: NO-GO register for this checkpoint |
| all_product_affinity | 0.016951 | 0.088546 | 0.066659 | [True, True, False]: NO-GO register for this checkpoint |

No register implementation is authorized by this probe. Continue the planned affinity analysis.
Between-checkpoint differences are observations under matched seed0 training, not proof that affinity causes artifacts.
