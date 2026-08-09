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
├── data/
├── configs/
├── misc/
├── tools
├── scripts_coco
├── scripts_voc
├── README.md
└── ...
```

---

## ⚙️ Installation

### Three independent Conda environments are required

TGCA, MoRe, and CTI use different Python, PyTorch, CUDA, and `timm` versions. Do **not** install all three methods into one shared environment. Create one independent environment for each method:

* [`tgca-repro`: How to build the TGCA environment](#how-to-build-the-tgca-environment)
* [`more-repro`: How to build the MoRe environment](#how-to-build-the-more-environment)
* [`cti-repro`: How to build the CTI environment](#how-to-build-the-cti-environment)

For a fair comparison, run a host's vanilla baseline and its TGCA variant in the same host-specific environment. Record the complete environment, repository commit, command, seed, and checkpoint for every reported result.

These environments target Linux with an NVIDIA GPU. Check that the installed CUDA runtime is compatible with the machine's NVIDIA driver before training.

### How to build the TGCA environment

The current TGCA repository is based on MCTTA/MCTG. Its [`requirements.txt`](requirements.txt) combines PyTorch 2.1.0 with an incompatible torchvision version, so install the compatible PyTorch stack first and exclude the two conflicting lines when installing the remaining dependencies:

```bash
# Clone the repository
git clone --recurse-submodules https://github.com/jungletada/TGCA.git
cd TGCA

# Create the independent TGCA/MCTTA environment
conda create -n tgca-repro python=3.9 -y
conda activate tgca-repro
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install the remaining dependencies without replacing PyTorch or torchvision
grep -vE '^(torch|torchvision)==' requirements.txt > /tmp/tgca-requirements.txt
python -m pip install -r /tmp/tgca-requirements.txt
```

The legacy dependency pins still require validation and will be replaced by a clean, tested environment specification before public release.

### How to build the MoRe environment

MoRe requires Python 3.7, its older PyTorch stack, and a separately compiled bilateral-filter extension. Its [`requirements.txt`](hosts/MoRe/requirements.txt) is a reference list rather than a valid pip requirements file.

```bash
conda create -n more-repro python=3.7 -y
conda activate more-repro
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch -y
conda install numpy==1.21.5 pydensecrf==1.0rc3 -c conda-forge -y
python -m pip install einops==0.6.0 imageio==2.26.0 matplotlib==3.5.3 \
  mmcv==1.7.1 scikit-image==0.19.3 scikit-learn==1.0.2 \
  tensorboard==2.11.2 texttable==1.6.7 timm==0.6.12 tqdm==4.65.0
```

Then build the [bilateral-filter Python extension](https://github.com/meng-tang/rloss/tree/master/pytorch#build-python-extension-module) inside `more-repro`. MoRe currently hard-codes an extension build path, so verify or update that path locally before running its training scripts.

### How to build the CTI environment

CTI provides an exported [`environment.yml`](hosts/CTI/environment.yml) containing Python 3.9, PyTorch 2.0.1, torchvision 0.15.2, and CUDA 11.7. Remove its machine-specific prefix and give the environment a descriptive name before creating it:

```bash
sed -e '/^prefix:/d' -e 's/^name: ddpm$/name: cti-repro/' \
  hosts/CTI/environment.yml > /tmp/cti-repro.yml
conda env create -f /tmp/cti-repro.yml
conda activate cti-repro
```

The CTI README describes a different PyTorch version from the exported environment. Use the internally consistent exported stack above as the initial reproduction environment, and record any compatibility changes needed to reproduce the published baseline.

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
data/VOCdevkit/VOC2012/
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
data/MSCOCO/
├── train2014/
├── val2014/
├── annotations/
│   └── instances_train2014.json
│   └── instances_val2014.json
└── ImageLabel
```



## For VOC 
### 1. 🏋️ Generate Class Avtivation Map ()

```bash
bash scripts_voc/run_cam.sh 
```
### 2. 🏋️ Run Pixel Semantic Affinity (PSA) or (Inter-Pixel Relation) IRN
```bash
bash scripts_voc/run_psa.sh
```

### 3. 🧪 Alternative: Run direct method
```
bash scripts_voc/run_direct.sh
```

### 4. 🧪 Run WSSS using generated pseudo label
- Please install mmsegmentation and run the training and inference in MMSeg  

## For COCO
### Same as VOC
```bash
bash scripts_coco/run_cam.sh

bash scripts_coco/run_psa.sh

bash scripts_coco/run_direct.sh
```


## 📜 Citation

If you find this work useful, please consider citing:

```bibtex
@article{peng2024structural,
  title={Structural Relation Multi-class Token Transformer for Weakly Supervised Semantic Segmentation},
  author={PENG, Dingjie and KAMEYAMA, Wataru},
  journal={IEICE Transactions on Information and Systems},
  year={2024},
  publisher={The Institute of Electronics, Information and Communication Engineers}
}
```

---

## 🤝 Acknowledgements

This project builds on the [MCTformer+](https://github.com/your-base-repo/MCTformerPlus) architecture and draws inspiration from recent advances in weakly supervised segmentation. We thank the authors of prior work for their contributions.

---
