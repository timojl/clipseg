import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as nnf
from torchvision import transforms


def _resolve_dataset_root(dataset_root):
    if dataset_root:
        candidates = [Path(os.path.expanduser(dataset_root))]
    else:
        repo_root = Path(__file__).resolve().parents[1]
        candidates = []

        env_root = os.environ.get('CLIPSEG_DATASET_ROOT')
        if env_root:
            candidates.append(Path(os.path.expanduser(env_root)))

        candidates.extend([
            repo_root / 'dataset',
            repo_root.parent / 'dataset',
            Path('E:/Projects/Segmentation/dataset'),
        ])

    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        checked.append(str(candidate))
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        'Could not locate the disease dataset. Checked: '
        + ', '.join(checked)
        + '. Set dataset_root in the experiment config or CLIPSEG_DATASET_ROOT.'
    )


def _clean_prompt(conversations):
    if not conversations:
        raise ValueError('Sample is missing conversations.')

    prompt = conversations[0].get('value', '').strip()
    prompt = prompt.replace('<image>\n', '', 1).strip()
    prompt = prompt.replace('<image>', '', 1).strip()

    if not prompt:
        raise ValueError('Sample prompt is empty after cleanup.')

    return prompt


class DiseasePromptSegmentation(object):

    def __init__(self, split, dataset_root=None, image_size=352, normalize=True, limit_samples=None):
        super().__init__()

        split_map = {'train': 'train.json', 'val': 'test.json', 'test': 'test.json'}
        if split not in split_map:
            raise ValueError(f'Unsupported split {split!r}. Expected one of {sorted(split_map)}')

        self.dataset_root = _resolve_dataset_root(dataset_root)
        self.image_size = image_size
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ) if normalize else None

        annotation_path = self.dataset_root / split_map[split]
        with open(annotation_path, 'r', encoding='utf-8') as handle:
            self.samples = json.load(handle)

        if limit_samples is not None:
            self.samples = self.samples[:int(limit_samples)]

    def __len__(self):
        return len(self.samples)

    def _load_image(self, relpath):
        image = Image.open(self.dataset_root / relpath).convert('RGB')
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image = nnf.interpolate(
            image.unsqueeze(0),
            (self.image_size, self.image_size),
            mode='bilinear',
            align_corners=True,
        )[0]

        if self.normalize is not None:
            image = self.normalize(image)

        return image

    def _load_mask(self, relpath):
        mask = Image.open(self.dataset_root / relpath)
        mask = torch.from_numpy(np.array(mask) > 0).float()
        mask = nnf.interpolate(
            mask.view(1, 1, *mask.shape),
            (self.image_size, self.image_size),
            mode='nearest',
        )[0]
        return mask

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = _clean_prompt(sample.get('conversations', []))
        image = self._load_image(sample['image'])
        mask = self._load_mask(sample['masks'][0])

        return (image, prompt), (mask, torch.zeros(0), index)
