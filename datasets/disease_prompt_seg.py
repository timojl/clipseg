import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as nnf
from torchvision import transforms

PROMPT_CANDIDATES = (
    'the diseased region',
    'the lesion region',
    'the infected area',
    'the symptomatic region',
    'the lesion on the leaf',
    'the lesion on the fruit surface',
)


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


def _sample_prompt():
    return random.choice(PROMPT_CANDIDATES)


def _extract_class_name(sample):
    conversations = sample.get('conversations', [])
    if len(conversations) > 1:
        answer = conversations[1].get('value', '')
        marker = 'symptoms of '
        if marker in answer:
            return answer.split(marker, 1)[1].split('.', 1)[0].strip()

    return sample['id'].rsplit('_', 1)[0].replace('_', ' ')


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

        all_class_names = set()
        for filename in {'train.json', 'test.json'}:
            annotation_path = self.dataset_root / filename
            if annotation_path.is_file():
                with open(annotation_path, 'r', encoding='utf-8') as handle:
                    all_class_names.update(_extract_class_name(sample) for sample in json.load(handle))

        self.class_names = sorted(all_class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

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
        prompt = _sample_prompt()
        image = self._load_image(sample['image'])
        mask = self._load_mask(sample['masks'][0])
        class_idx = self.class_to_idx[_extract_class_name(sample)]

        return (image, prompt), (mask, torch.zeros(0), class_idx)
