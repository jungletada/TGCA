# Diffusion and Flow Matching for Weakly Supervised Semantic Segmentation

**Survey date:** 2026-08-30

**Scope:** image-level weakly supervised semantic segmentation (WSSS), with
medical and other weak-supervision settings included only when they clarify the
role of diffusion or flow matching.

**Status:** literature survey and research-positioning document. It does not
claim that a new method has been implemented or validated in this repository.

## 1. Executive summary

The literature supports four high-level conclusions.

1. Generative diffusion models have already been used extensively in natural-
   image WSSS. They have served as frozen feature/locality teachers, sources of
   attention-derived pseudo-masks, controlled data generators, augmentation
   models, and complementary priors for CLIP.
2. Most natural-image methods do not learn a diffusion-based mask distribution
   directly from image-level labels. They import information from a pretrained
   text-to-image or denoising model and retain a conventional discriminative CAM
   or pseudo-label training pipeline.
3. Direct weakly supervised generative mask modeling is clearer in medical
   imaging. Conditional diffusion has been used with image-level presence tags,
   discrete diffusion has been trained from a single positive pixel, and a 3D
   rectified-flow prior has been guided by an image-level predictor.
4. As of the survey date, the search did not identify a published natural-image
   VOC/COCO method that uses image-level labels to make Flow Matching or
   Rectified Flow the core dense-mask learning objective. This is a plausible
   research gap, not proof that no unpublished or differently indexed work
   exists.

The weak novelty claim "diffusion improves WSSS" is no longer defensible.
A potentially defensible direction is narrower:

> Learn an uncertainty-conditioned transport over semantic mask distributions
> from image-level labels, and prove that the distributional transport adds
> value beyond the frozen teachers, a deterministic pseudo-label refiner, and
> ordinary soft-target distillation.

That direction has a fundamental identifiability problem: WSSS does not provide
samples from the true target mask distribution. If a flow model merely learns
to reproduce one hard pseudo-mask, it is a computationally expensive
pseudo-label student and does not establish a generative contribution.

## 2. Definitions and inclusion rules

### 2.1 Standard image-level WSSS

This document uses *standard image-level WSSS* for methods that train or adapt
using images and their image-level multi-label class vectors, without pixel
masks, boxes, points, or scribbles for the training images. PASCAL VOC 2012 and
MS COCO 2014 are the primary natural-image benchmarks.

Point-, scribble-, box-, and image-tag-supervised methods are all weakly
supervised in a broad sense, but they are not interchangeable. The supervision
type is therefore stated for every central paper.

### 2.2 Generative diffusion

The relevant diffusion family includes continuous or discrete denoising
generative models that learn or reuse a noising/denoising process. A paper may
use such a model in at least four distinct ways:

- as a frozen feature or spatial-relation extractor;
- as an attention-based pseudo-mask generator;
- as a conditional image/data generator;
- as the actual probabilistic mask predictor or weak localizer.

Only the last case directly makes diffusion the segmentation learning
mechanism. The first three are important prior art but should not be described
as training a diffusion segmentation model from image-level labels.

### 2.3 Flow Matching, Rectified Flow, and Normalizing Flow

These terms must not be collapsed into one "flow-based" category.

- **Flow Matching (FM)** learns a time-dependent vector field along a chosen
  probability path between source and target distributions.
- **Rectified Flow (RF)** is a flow-matching formulation that favors straight
  transport paths and efficient ODE sampling.
- **Normalizing Flow (NF)** is an invertible density model trained through
  change-of-variables likelihood or related objectives. It is prior art for
  distribution modeling, but it is not the same training objective as FM.
- **Graph diffusion**, random-walk propagation, affinity diffusion, and label
  spreading are not generative diffusion models and are excluded from the
  central survey.

### 2.4 Evidence and publication status

Primary conference pages, publisher pages, arXiv records, and official code
repositories were preferred. Author-reported metrics are marked as such and
must not be compared across rows as a matched leaderboard because backbones,
pretraining, thresholds, synthetic data, post-processing, and supervision
differ.

An arXiv paper marked "submitted to" a venue is treated as a preprint, not as
an accepted paper.

## 3. Taxonomy of existing work

| Family | Representative methods | Generative model trained for WSSS? | Main scientific role |
|---|---|---:|---|
| Frozen diffusion feature teacher | DiG, DiCLIP, ComCD | No | Add locality, spatial correlation, or semantics to CAM learning |
| Training-free attention/mask extraction | DiffSegmenter, iSeg | No | Convert pretrained diffusion attention into dense class maps |
| Diffusion-controlled augmentation | IACD | Usually no | Expand an image-level-labeled training set |
| Diffusion-synthetic segmentation data | DiffuMask, Attn2mask | Usually no downstream diffusion training | Generate images and noisy pixel labels for a conventional segmenter |
| Direct weak localizer/mask distribution | Conditional medical diffusion, single-pixel D3PM | Yes | Learn a conditional object-location or mask distribution from weak labels |
| Weakly guided Rectified Flow | 3D lung-nodule RF | Frozen RF; predictor adapted | Produce counterfactual localization through generative guidance |
| Fully supervised FM segmentation | FlowSDF, SymmFlow, FlowDIS | Yes | Learn image-to-mask transport from dense mask pairs; not WSSS |
| Normalizing-flow WSSS | BRNF | Yes, but NF rather than FM | Model class-wise pixel-feature distributions to reduce CAM bias |

## 4. Natural-image WSSS using diffusion

### 4.1 DiG: diffusion as a frozen locality teacher

[Diffusion-Guided Weakly Supervised Semantic Segmentation (DiG), ECCV
2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6482_ECCV_2024_paper.php)
is a direct image-level WSSS method on VOC and COCO. It uses a pretrained DDPM
without fine-tuning and aggregates features from early denoising timesteps.
Locality Fusion Cross Attention aligns these diffusion features with ViT
features to produce a Diffusion-CAM that supervises the ViT-CAM. A denoised
version of the input also supplies an augmentation for patch-affinity
consistency.

The important positioning point is that DiG does not learn a mask diffusion
process from image labels. Its contribution is importing the spatial and
semantic structure of a frozen image diffusion model into a conventional
ViT/CAM WSSS system.

The paper reports a protocol-specific CAM ablation from 65.7 for its baseline
to 69.3 for the full method on VOC train. Those values use the paper's own CAM
selection/evaluation protocol and are not directly comparable to this
repository's fixed-threshold MCTformer+ result.

### 4.2 IACD: controlled diffusion for labeled-data expansion

[Image Augmentation with Controlled Diffusion for Weakly-Supervised Semantic
Segmentation (IACD), ICASSP
2024](https://openreview.net/forum?id=yJsAzJdXjg) starts from images and
image-level labels and uses them as controls for diffusion-based generation.
A quality-selection stage rejects unreliable generated samples. The reported
benefit is strongest when only a limited fraction of the labeled dataset is
available.

This establishes that "controlled diffusion augmentation for image-label
WSSS" is occupied prior art. It does not establish diffusion as a dense mask
predictor, because the generated data augment an otherwise discriminative WSSS
pipeline.

### 4.3 DiffuMask and Attn2mask: diffusion-synthetic training

[DiffuMask, ICCV
2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_DiffuMask_Synthesizing_Images_with_Pixel-level_Annotations_for_Semantic_Segmentation_Using_ICCV_2023_paper.pdf)
uses a pretrained text-to-image diffusion model to synthesize images and derive
pixel annotations from internal attention. This reduces reliance on manually
annotated masks, but downstream segmentation is trained on the generated
image-mask pairs.

[Exploring Limits of Diffusion-Synthetic Training with Weakly Supervised
Semantic Segmentation, ACCV
2024](https://openaccess.thecvf.com/content/ACCV2024/html/Yoshihashi_Exploring_Limits_of_Diffusion-Synthetic_Training_with_Weakly_Supervised_Semantic_Segmentation_ACCV_2024_paper.html),
also known through the Attn2mask framing, explicitly treats inaccurate
diffusion attention masks as weak labels. It adds prompt augmentation,
reliability-aware robust co-training, and optional domain adaptation. The
method can train a segmenter without real downstream training images or manual
masks.

Its author-reported synthetic-only VOC mIoU increases with generated-data
scale, from 41.2 with 2,000 synthetic samples to 51.1 with 168,679 samples;
adding the paper's BECO setting gives 58.3. The result demonstrates feasibility
but also exposes a material domain/noisy-label gap relative to strong
real-image WSSS systems.

This literature makes a new "generate images and use diffusion attention as
masks" framework incremental unless it addresses a distinct failure mode and
outperforms confidence-matched synthetic-data controls.

### 4.4 DiffSegmenter and iSeg: training-free diffusion masks

[DiffSegmenter](https://vcg-team.github.io/DiffSegmenter-webpage/) treats a
frozen conditional latent diffusion model as a training-free open-vocabulary
segmenter. It derives category maps from text-image cross-attention and refines
them with self-attention. It is relevant to WSSS as a seed or pseudo-mask
teacher, but its central task is training-free/open-vocabulary segmentation,
not learning a WSSS classifier from image labels.

[iSeg](https://arxiv.org/abs/2409.03209) iteratively refines cross-attention
with entropy-reduced self-attention and class-enhanced cross-attention. Its
official implementation includes WSSS evaluation and reports author-provided
pseudo-mask scores, but the segmentation mechanism remains frozen and
training-free. It is therefore a strong teacher baseline for a new framework,
not evidence that a learned flow or diffusion student is necessary.

### 4.5 DiCLIP: diffusion-enhanced CLIP dense knowledge

[DiCLIP: Diffusion Model Enhances CLIP's Dense Knowledge for Weakly Supervised
Semantic Segmentation](https://doi.org/10.1109/TIP.2026.3692055) is especially
important for current positioning. Its Visual Correlation Enhancement extracts
diverse diffusion correlation maps and uses them to refine CLIP self-attention.
Its Text Semantic Augmentation uses diffusion-generated single-class images to
form a dynamic visual knowledge cache for text-semantic retrieval.

The official project reports VOC val/test and COCO results, but those numbers
combine CLIP and an external diffusion prior and should be separated from
ImageNet-only class-token methods in comparison tables. For a new diffusion
framework, DiCLIP is a required competitive baseline rather than a host that
can be ignored.

Official code: [zwyang6/DiCLIP](https://github.com/zwyang6/DiCLIP).

### 4.6 ComCD: entropy-based CLIP/diffusion fusion

[Unveiling the Complementary Synergy of CLIP and Diffusion Models for Weakly
Supervised Semantic Segmentation (ComCD), Expert Systems with Applications
2026](https://doi.org/10.1016/j.eswa.2026.131884) constructs separate CLIP-CAM
and diffusion-CAM branches. An entropy-difference rule produces per-pixel
fusion weights and a refined pseudo-mask. A trainable Feature Aligned Decoder
then aligns both feature streams and uses logit gating under pseudo-mask
supervision.

ComCD directly occupies the generic claim that CLIP and diffusion have
complementary uncertainty and can be fused according to per-pixel confidence.
A new method needs a stronger distinction than replacing entropy fusion with a
learned gate. In particular, it must isolate whether distributional transport
adds value beyond deterministic uncertainty-weighted fusion.

## 5. Direct weakly supervised diffusion in medical imaging

### 5.1 Conditional diffusion from image-level tags

[Conditional Diffusion Models for Weakly Supervised Medical Image
Segmentation](https://arxiv.org/abs/2306.03878) uses image-level category or
presence labels and extracts localization from the class-conditioned denoising
model. The target-object response is obtained from the sensitivity or
difference of conditional noise predictions. Experiments cover medical
datasets such as BraTS and CHAOS.

This is one of the clearest precedents where the conditional diffusion model
itself contains the weak localization mechanism. Its domain is typically
binary/small-class medical segmentation rather than multi-label natural-image
WSSS, so transferring the idea to VOC still requires explicit background and
multi-class conflict handling.

### 5.2 Discrete diffusion from a single positive pixel

[A Single Pixel is All You Need: Weakly Supervised Medical Image Segmentation
using Discrete Denoising Diffusion Models, CVPR Findings
2026](https://openaccess.thecvf.com/content/CVPR2026F/html/Demirel_A_Single_Pixel_is_All_You_Need_Weakly_Supervised_Medical_CVPRF_2026_paper.html)
recasts weak segmentation as conditional generation. A D3PM is trained from a
single positive point to generate a sparse object point cloud; repeated samples
are aggregated into a dense segmentation.

This paper is strong evidence that discrete generative label modeling can use
extremely sparse supervision. It is not an image-level WSSS precedent because
the positive point provides spatial information unavailable in VOC image
labels.

### 5.3 Controlled medical synthesis

[Enhancing Weakly Supervised Semantic Segmentation for Fibrosis via
Controllable Image Generation](https://arxiv.org/abs/2411.03551) uses
image-level fibrosis annotations and a diffusion-based generator to synthesize
controlled pathology together with location information. It reinforces the
data-generation branch of the literature, while remaining domain-specific and
dependent on a healthy-to-pathology synthesis assumption that does not
generalize directly to multi-object natural scenes.

## 6. Flow Matching and Rectified Flow

### 6.1 Direct weakly supervised Rectified Flow evidence

[Weakly-Supervised Lung Nodule Segmentation via Training-Free Guidance of 3D
Rectified Flow](https://arxiv.org/abs/2604.08313) combines a pretrained 3D
rectified-flow model with predictors fine-tuned using image-level labels. The
generative model is not retrained. Predictor guidance steers reconstruction and
localizes counterfactual differences associated with lung nodules.

The arXiv record states that the work was submitted to MICCAI 2026; it must not
be cited as accepted unless that status is independently updated. Scientifically
it is still important: it prevents an unrestricted claim of "first weakly
supervised segmentation with Rectified Flow." A narrower natural-image,
multi-class, image-level claim may remain available.

### 6.2 SymmFlow is not an image-level WSSS result

[Symmetrical Flow Matching, AAAI
2026](https://ojs.aaai.org/index.php/AAAI/article/view/37236) jointly models
image generation, semantic segmentation, and classification with bidirectional
flows. Its formulation accepts a semantic variable that may be either a dense
mask or a global class label.

The distinction is in the experiments: segmentation on CelebAMask-HQ and
COCO-Stuff uses paired dense semantic masks, while global labels are evaluated
in separate MNIST/CIFAR classification experiments. It does not demonstrate
learning a dense segmentation from image-level labels. Nevertheless, its
bidirectional image/semantic formulation and label dequantization are directly
relevant architectural prior art for a future WSSS flow model.

### 6.3 Fully supervised FM segmentation is adjacent, not WSSS

[FlowSDF, IJCV 2025](https://doi.org/10.1007/s11263-025-02373-y) learns an
image-conditioned flow over signed-distance representations of medical masks.
Its strength is probabilistic shape and boundary modeling, but it has dense
mask endpoints during training.

[FlowDIS](https://arxiv.org/abs/2605.05077) transports images to dichotomous
masks, optionally conditioned by language, and introduces a position-aware
pairing strategy. It also relies on image-mask training pairs rather than
image-level WSSS.

[RLFSeg](https://arxiv.org/abs/2605.04590) studies rectified-flow-based
text-conditioned/zero-shot segmentation and efficient image-to-mask transport.
It is relevant to architecture and initialization, but it does not supply a
matched VOC image-level WSSS protocol.

These papers mean that "Flow Matching for segmentation" is not novel. The
unoccupied part is the supervision problem: learning useful dense transport
without ground-truth mask endpoints.

## 7. Normalizing Flow WSSS must be treated as direct prior art

[Bias-Resilient Weakly Supervised Semantic Segmentation Using Normalizing
Flows (BRNF), ICCV
2025](https://openaccess.thecvf.com/content/ICCV2025/html/Qiu_Bias-Resilient_Weakly_Supervised_Semantic_Segmentation_Using_Normalizing_Flows_ICCV_2025_paper.html)
is a standard image-level WSSS method. It models dataset-wide pixel-feature
distributions with a normalizing flow, augments the conventional MLP classifier
with a Gaussian-mixture classifier, samples low-bias positive anchors for
contrastive learning, and uses distribution likelihood to suppress background
outliers.

BRNF is not Flow Matching, but it occupies several nearby claims:

- distribution modeling can reduce WSSS activation bias;
- a flow-based density model can exploit dataset-level pixel statistics;
- likelihood and sampling can provide lower-bias semantic anchors;
- background false positives can be treated as feature-distribution outliers.

Any future paper must use precise terminology such as *Flow Matching* or
*Rectified Flow*, cite BRNF, and explain why a time-dependent mask transport
solves a different problem from invertible pixel-feature density estimation.

## 8. What the literature does and does not establish

### 8.1 Established

- Pretrained diffusion U-Nets contain spatial and semantic signals useful for
  natural-image WSSS.
- Diffusion cross-attention can generate usable but noisy pseudo-masks.
- Controlled generation can expand image-level-labeled or synthetic training
  sets.
- CLIP and diffusion maps can be complementary, and simple confidence/entropy
  fusion is already a competitive baseline.
- Weak spatial supervision can train or guide generative label models in
  medical domains.
- Flow Matching can learn image-to-mask transport efficiently when dense masks
  are available.
- Normalizing flows have already been used directly in image-level WSSS.

### 8.2 Not established

- That a learned diffusion or flow refiner is better than a capacity-matched
  deterministic refiner when both receive exactly the same pseudo-labels.
- That stochastic samples improve the expected WSSS prediction rather than
  only producing visually plausible variation.
- That teacher disagreement is a calibrated approximation to the true mask
  distribution.
- That a flow model can correct errors absent from all of its pseudo-mask
  endpoints.
- That the extra training and sampling cost is justified by localization,
  calibration, or downstream segmentation gains.
- That image-level labels alone prevent all-background, all-foreground, or
  co-occurrence shortcuts in a categorical generative mask model.

## 9. Research gap and candidate contribution

The most defensible gap found by this survey is:

> Natural-image image-level WSSS lacks a controlled study of whether structured
> uncertainty over weak semantic masks can serve as the endpoint distribution
> for conditional Flow Matching, and whether this improves localization beyond
> deterministic hard/soft pseudo-label learning under matched teachers,
> capacity, data, and evaluation.

A candidate framework could use:

```text
image + image labels
    -> frozen, independent weak teachers
    -> multi-view categorical pseudo-mask samples
    -> calibrated endpoint distribution over 21 VOC labels
    -> image- and label-conditioned discrete Flow Matching
    -> expected mask plus uncertainty at inference
```

The foreground classes must be restricted by the image-level label vector, but
background must remain an explicit 21st state rather than being inferred as a
residual after independent foreground normalization. Cross-view equivariance,
boundary affinity, and teacher reliability may constrain the distribution, but
all such constraints must be shared with deterministic controls.

### 9.1 Necessary novelty controls

At minimum, compare against:

- the strongest individual frozen teacher;
- fixed and entropy-weighted teacher fusion;
- a capacity-matched deterministic hard-label refiner;
- a capacity-matched deterministic soft-label refiner;
- noise-augmented deterministic training;
- flow matching to one hard pseudo-mask endpoint;
- flow matching to the structured multi-teacher endpoint distribution;
- a shuffled or confidence-matched uncertainty control.

Without these controls, a gain could be explained by teacher quality, model
capacity, ordinary soft labels, augmentation noise, test-time ensembling, or
additional compute rather than Flow Matching.

### 9.2 Central technical risk

Flow Matching normally assumes access to endpoint samples from the target
distribution. In image-level WSSS, the true mask distribution is unobserved.
Teacher masks are biased observations, not ground truth. Therefore:

- transporting noise to a hard teacher mask only distills that teacher;
- transporting to independently sampled per-pixel labels can destroy spatial
  coherence;
- transporting to full teacher/view masks preserves coherence but may learn
  teacher identity instead of semantic uncertainty;
- selecting the best generated sample with validation ground truth is invalid;
- a stochastic model cannot be credited for uncertainty unless calibration and
  expected prediction improve under a fixed inference rule.

This identifiability issue is the scientific problem a new framework must solve,
not an implementation detail to defer until after full training.

## 10. Recommended positioning and claims

### 10.1 Potentially supportable after validation

- First controlled natural-image study of image-level WSSS using conditional
  discrete/rectified Flow Matching over weak semantic-mask distributions.
- Structured multi-teacher uncertainty provides a better endpoint than one
  hard pseudo-mask under matched training.
- Flow-based prediction improves expected mask quality and calibration, not
  merely best-of-many samples.
- The method improves raw seed quality and one fixed downstream segmentation
  pipeline under transparent external priors.

### 10.2 Claims to avoid

- First use of diffusion in WSSS.
- First use of flow in WSSS without distinguishing BRNF and the lung RF work.
- First generative segmentation model.
- Annotation-free if VOC image-level labels, teacher prompts, or pretrained
  models supply supervision.
- Superior efficiency without measured training and inference cost.
- State of the art across ImageNet-only, CLIP-, SAM-, language-, and
  diffusion-assisted systems without matched settings.

## 11. Implications for this repository

The current repository provides a trustworthy MCTformer+ baseline, fixed CAM
threshold behavior, machine-readable metrics, and established diagnostics for
precision, recall, background false positives, class conflict, and cross-scale
consistency. Those assets can be reused as an evaluation host.

The completed TGCA, BCSS, token-role, and persistent-semantic-latent screens
are negative explorations and should not be silently combined into a new
diffusion method. A diffusion/flow direction needs a separate experiment
namespace, environment manifest, hypothesis, and go/no-go record.

Before adopting a new framework, execute the three minimal validation
experiments predeclared in [validation_diffusion.md](validation_diffusion.md).
They test teacher signal, generative-objective necessity, and structured
uncertainty in that order.

## 12. Primary references

1. Sung-Hoon Yoon et al., [Diffusion-Guided Weakly Supervised Semantic
   Segmentation](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6482_ECCV_2024_paper.php),
   ECCV 2024.
2. Wangyu Wu et al., [Image Augmentation with Controlled Diffusion for
   Weakly-Supervised Semantic Segmentation](https://openreview.net/forum?id=yJsAzJdXjg),
   ICASSP 2024.
3. Ryota Yoshihashi et al., [Exploring Limits of Diffusion-Synthetic Training
   with Weakly Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/ACCV2024/html/Yoshihashi_Exploring_Limits_of_Diffusion-Synthetic_Training_with_Weakly_Supervised_Semantic_Segmentation_ACCV_2024_paper.html),
   ACCV 2024.
4. Weijia Wu et al., [DiffuMask: Synthesizing Images with Pixel-level
   Annotations for Semantic Segmentation Using Diffusion Models](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_DiffuMask_Synthesizing_Images_with_Pixel-level_Annotations_for_Semantic_Segmentation_Using_ICCV_2023_paper.html),
   ICCV 2023.
5. Lin Sun et al., [iSeg: An Iterative Refinement-based Framework for
   Training-free Segmentation](https://arxiv.org/abs/2409.03209), arXiv
   2409.03209; verify final bibliographic metadata before paper submission.
6. Jinglong Wang et al., [Diffusion Model is Secretly a Training-free Open
   Vocabulary Semantic Segmenter](https://vcg-team.github.io/DiffSegmenter-webpage/),
   IEEE T-IP 2025, DOI 10.1109/TIP.2025.3551648.
7. Zhiwei Yang et al., [DiCLIP: Diffusion Model Enhances CLIP's Dense Knowledge
   for Weakly Supervised Semantic Segmentation](https://doi.org/10.1109/TIP.2026.3692055),
   IEEE T-IP 2026.
8. Hang Yao et al., [Unveiling the Complementary Synergy of CLIP and Diffusion
   Models for Weakly Supervised Semantic Segmentation](https://doi.org/10.1016/j.eswa.2026.131884),
   Expert Systems with Applications, 2026.
9. [Conditional Diffusion Models for Weakly Supervised Medical Image
   Segmentation](https://arxiv.org/abs/2306.03878), arXiv 2306.03878.
10. Mehmet Demirel and Christos Kyrkou, [A Single Pixel is All You Need:
    Weakly Supervised Medical Image Segmentation using Discrete Denoising
    Diffusion Models](https://openaccess.thecvf.com/content/CVPR2026F/html/Demirel_A_Single_Pixel_is_All_You_Need_Weakly_Supervised_Medical_CVPRF_2026_paper.html),
    CVPR Findings 2026.
11. Richard Petersen et al., [Weakly-Supervised Lung Nodule Segmentation via
    Training-Free Guidance of 3D Rectified Flow](https://arxiv.org/abs/2604.08313),
    arXiv 2604.08313, submitted to MICCAI 2026 as of the survey date.
12. Francisco Caetano et al., [Symmetrical Flow Matching: Unified Image
    Generation, Segmentation, and Classification with Score-Based Generative
    Models](https://ojs.aaai.org/index.php/AAAI/article/view/37236), AAAI 2026.
13. Lea Bogensperger et al., [FlowSDF: Flow Matching for Medical Image
    Segmentation Using Distance Transforms](https://doi.org/10.1007/s11263-025-02373-y),
    IJCV 2025.
14. Xianglin Qiu et al., [Bias-Resilient Weakly Supervised Semantic
    Segmentation Using Normalizing Flows](https://openaccess.thecvf.com/content/ICCV2025/html/Qiu_Bias-Resilient_Weakly_Supervised_Semantic_Segmentation_Using_Normalizing_Flows_ICCV_2025_paper.html),
    ICCV 2025.

## 13. Survey limitations

- Search-index coverage is incomplete, especially for papers released close to
  the survey date or indexed under medical localization rather than WSSS.
- Several 2026 works have only recent publisher or arXiv records; venue and
  version metadata must be rechecked before citation in a submission.
- Author-reported scores were not reproduced in this repository.
- This survey does not establish that a new Flow Matching method will work; it
  identifies an empirical gap that must pass controlled validation.
