"""Deterministic image-label-only probe selection, separate GT mini-CAM set."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .writer import result_path, sha256, write_json


def semantic_transform(resolution):
    return transforms.Compose([
        transforms.Resize(int(resolution * 512 / 448), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resolution), transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    ])


def mask_transform(mask, resolution):
    # Same geometry as semantic_transform; categorical labels use nearest.
    mask = transforms.functional.resize(mask, int(resolution * 512 / 448), interpolation=transforms.InterpolationMode.NEAREST)
    return np.asarray(transforms.functional.center_crop(mask, resolution)).copy()


def label_path(root, dataset):
    return Path(root) / 'ImageLabel' / ('cls_labels.npy' if dataset == 'VOC12' else 'COCO_cls_labels.npy')


def labels_for(ids, root, dataset):
    labels = np.load(label_path(root, dataset), allow_pickle=True).item()
    y = np.asarray([labels[name if dataset == 'VOC12' else name + '.jpg'] for name in ids], dtype=np.float32)
    if y.ndim != 2 or not np.isin(y, [0, 1]).all():
        raise ValueError('Expected binary image-level class labels')
    return y


def read_ids(path):
    ids = Path(path).read_text().splitlines()
    if not ids or len(ids) != len(set(ids)) or any(not i or '/' in i or '\\' in i or i in {'.', '..'} for i in ids):
        raise ValueError('Image IDs must be unique bare identifiers')
    return ids


def stratified_indices(labels, size=256, minimum=8, seed=0):
    """Greedy rare-label coverage followed by seeded fill, no global RNG use."""
    y = np.asarray(labels, dtype=np.int64)
    if minimum < 0 or size < 1 or size > len(y) or (y.sum(0) < minimum).any():
        raise ValueError('Requested fixed-set coverage is infeasible')
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    counts = np.zeros(y.shape[1], dtype=np.int64)
    selected = []
    available = np.ones(len(y), dtype=bool)
    rarity = 1 / y.sum(0).clip(1)
    while (counts < minimum).any() and len(selected) < size:
        scores = y @ ((minimum - counts).clip(0) * rarity)
        scores[~available] = -1
        index = int(order[np.argmax(scores[order])])
        selected.append(index)
        available[index] = False
        counts += y[index]
    if (counts < minimum).any():
        raise ValueError('Cannot meet per-class minimum within requested probe size')
    selected.extend(int(i) for i in order[available[order]][:size - len(selected)])
    return np.asarray(sorted(selected))  # fixed source-list order


def build_sets(root, dataset, train_list, masks, output, probe_size=256, mini_size=500, minimum=8, seed=0):
    output = result_path(output)
    ids = read_ids(train_list)
    y = labels_for(ids, root, dataset)
    selected = stratified_indices(y, probe_size, minimum, seed)
    probe = [ids[i] for i in selected]
    probe_names = set(probe)
    # Only file existence, not pixels, is consulted to form the separate set.
    candidates = [name for name in ids if name not in probe_names and (Path(masks) / f'{name}.png').is_file()]
    if len(candidates) < mini_size:
        raise ValueError('Insufficient mask-backed images disjoint from the probe')
    pick = set(np.random.default_rng(seed).choice(len(candidates), mini_size, replace=False).tolist())
    mini = [name for i, name in enumerate(candidates) if i in pick]
    output.mkdir(parents=True, exist_ok=False)
    for name, values in [('probe_train_ids.txt', probe), ('mini_cam_train_ids.txt', mini)]:
        (output / name).write_text('\n'.join(values) + '\n')
    write_json(output / 'manifest.json', {
        'dataset': dataset, 'data_root': str(Path(root).resolve()), 'seed': seed,
        'source_list': str(Path(train_list).resolve()), 'source_list_sha256': sha256(train_list),
        'labels_sha256': sha256(label_path(root, dataset)), 'minimum_per_class': minimum,
        'probe_size': probe_size, 'mini_size': mini_size,
        'probe_set_sha256': sha256(output / 'probe_train_ids.txt'),
        'mini_set_sha256': sha256(output / 'mini_cam_train_ids.txt'),
        'probe_class_counts': y[selected].sum(0).astype(int).tolist(),
        'semantic_masks_read_for_probe': False, 'sets_disjoint': True,
    })
    return output / 'probe_train_ids.txt', output / 'mini_cam_train_ids.txt'


class ProbeDataset(Dataset):
    """Never loads masks; explicit train image directory avoids path heuristics."""
    def __init__(self, root, dataset, ids_path, resolution):
        self.ids = read_ids(ids_path)
        self.labels = labels_for(self.ids, root, dataset)
        self.image_dir = Path(root) / ('JPEGImages' if dataset == 'VOC12' else 'train2014')
        self.transform = semantic_transform(resolution)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        with Image.open(self.image_dir / f'{self.ids[index]}.jpg') as image:
            x = self.transform(image.convert('RGB'))
        return x, torch.from_numpy(self.labels[index]), self.ids[index]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset', choices=['VOC12', 'COCO'], required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--train-list', required=True)
    parser.add_argument('--mask-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    build_sets(args.data_root, args.dataset, args.train_list, args.mask_dir, args.output)


if __name__ == '__main__':
    main()
