# MCTTA: Multi-class Token Transformer Adapter for Weakly Supervised Semantic Segmentation

Dingjie PENG*
*Waseda University


## 🔍 Abstract

**Weakly Supervised Semantic Segmentation (WSSS)** offers a compelling alternative to fully supervised approaches by using image-level labels instead of dense pixel annotations. However, most existing WSSS methods depend on a single-structure design, which lacks effective hierarchical feature extraction and multi-level feature fusion.

To address these challenges, we propose **MCTTA (Multi-class Token Transformer Adapter)** — a novel framework that enhances the MCTformer+ baseline with graph-based image priors and hierarchical architectural design. MCTTA incorporates several innovative modules:

* **Spatial Prior Grapher (SPG)**
* **Class Token Projection (CTP)**
* **Cross Attention Block**
* **Split Weighted Softmax**
* **Multi-scale Feature Fusion Strategy** (inspired by Fully Supervised Semantic Segmentation)

MCTTA improves Class Activation Map (CAM) quality and final segmentation predictions, achieving **state-of-the-art performance** on both **PASCAL VOC 2012** and **MS COCO 2014** datasets. The proposed architecture shows strong performance in single-stage and multi-stage setups, and our ablation studies validate the importance of each component.

---

## 📁 Project Structure

```
MCTTA/
├── models/
├── datasets/
├── configs/
├── utils/
├── train.py
├── infer.py
├── generate_cam.py
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/MCTTA.git
cd MCTTA

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📦 Dataset Preparation

We support training and evaluation on **PASCAL VOC 2012** and **MS COCO 2014** datasets. Follow the instructions below to prepare each dataset.

### 📌 PASCAL VOC 2012

1. **Download images and annotations**:

   * Download the [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) dataset and extract it.

2. **Download image-level labels** (e.g., from [WSSS datasets](https://github.com/xiaohan2012/wsssaliency)):

   * Place the label files under `datasets/VOC2012/labels/`.

3. **Directory structure** should look like:

```
datasets/VOC2012/
├── JPEGImages/
├── SegmentationClass/
├── ImageSets/
│   └── Segmentation/
├── labels/
└── ... 
```

4. **Preprocessing** (if required):

```bash
python scripts/preprocess_voc.py --data_root datasets/VOC2012
```

---

### 📌 MS COCO 2014

1. **Download images**:

   * Train: [train2014.zip](http://images.cocodataset.org/zips/train2014.zip)
   * Val: [val2014.zip](http://images.cocodataset.org/zips/val2014.zip)

2. **Download annotations**:

   * [annotations\_trainval2014.zip](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)

3. **Directory structure** should look like:

```
datasets/COCO2014/
├── images/
│   ├── train2014/
│   └── val2014/
├── annotations/
│   └── instances_train2014.json
│   └── instances_val2014.json
└── labels/
```

4. **Preprocessing** (if required):

```bash
python scripts/preprocess_coco.py --data_root datasets/COCO2014
```

---

## 🏋️ Training

```bash
python train.py --config configs/train_mctta_voc.yaml
```

For COCO:

```bash
python train.py --config configs/train_mctta_coco.yaml
```

---

## 🧪 Inference

```bash
python infer.py --config configs/infer_voc.yaml
```

---

## 🗺️ CAM Generation

```bash
python generate_cam.py --config configs/cam_voc.yaml
```

---

## 📊 Evaluation

```bash
python evaluate.py --config configs/eval_voc.yaml
```

---

## 📈 Results

### PASCAL VOC 2012

| Method               | mIoU (val) | mIoU (test) |
| -------------------- | ---------- | ----------- |
| MCTTA (single-stage) | XX.X%      | XX.X%       |
| MCTTA (multi-stage)  | XX.X%      | XX.X%       |

### MS COCO 2014

| Method               | mIoU (val) |
| -------------------- | ---------- |
| MCTTA (single-stage) | XX.X%      |
| MCTTA (multi-stage)  | XX.X%      |

> Full results and comparison tables are available in the paper.

---

## 📜 Citation

If you find this work useful, please consider citing:

```bibtex
@article{your2025mctta,
  title={MCTTA: Multi-class Token Transformer Adapter for Weakly Supervised Semantic Segmentation},
  author={Your Name and Others},
  journal={CVPR/ICCV/ECCV},
  year={2025}
}
```

---

## 🤝 Acknowledgements

This project builds on the [MCTformer+](https://github.com/your-base-repo/MCTformerPlus) architecture and draws inspiration from recent advances in weakly supervised segmentation. We thank the authors of prior work for their contributions.

---

Let me know if you'd like to auto-generate the `requirements.txt` file or include demo notebooks.
