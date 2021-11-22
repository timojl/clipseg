from collections import defaultdict

from torchvision.transforms.functional import InterpolationMode
import numpy as np
import os
import torch
import json
from os.path import join, dirname, isfile, expanduser, realpath, basename
from PIL import Image
from general_utils import get_from_repository
from datasets.utils import blend_image_segmentation
from general_utils import log


IMAGENET_MEAN, IMAGENET_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def random_crop_slices(origin_size, target_size):
    """Gets slices of a random crop. """
    assert origin_size[0] >= target_size[0] and origin_size[1] >= target_size[1], f'actual size: {origin_size}, target size: {target_size}'

    offset_y = torch.randint(0, origin_size[0] - target_size[0] + 1, (1,)).item()  # range: 0 <= value < high
    offset_x = torch.randint(0, origin_size[1] - target_size[1] + 1, (1,)).item()

    return slice(offset_y, offset_y + target_size[0]), slice(offset_x, offset_x + target_size[1])


def find_crop_old(seg, image_size, iterations=1000, min_frac=None, best_of=None):

    best_crops = []
    min_sum = 0
    
    if min_frac is not None:
        min_sum = seg.sum() * min_frac
    
    for _ in range(iterations):
        sl_y, sl_x = random_crop_slices(seg.shape, image_size)
        seg_ = seg[sl_y, sl_x]

        if seg_.sum() > min_sum:

            if best_of is None:
                return sl_y, sl_x, False
            else:
                best_crops += [(seg_.sum(), sl_y, sl_x)]
                if len(best_crops) >= best_of:
                    best_crops.sort(key=lambda x:x[0], reverse=True)
                    sl_y, sl_x = best_crops[0][1:]
                    
                    return sl_y, sl_x, False
    else:
        return sl_y, sl_x, True


def bbox_overlap(a, b):
    # (y, x, height, width)

    overlap_y = max(0, (a[0] + a[2] - b[0]) if a[0] < b[0] else (b[0] + b[2] - a[0]))
    overlap_x = max(0, (a[1] + a[3] - b[1]) if a[1] < b[1] else (b[1] + b[3] - a[1]))
    return overlap_y * overlap_x


def find_crop(seg, image_size, iterations=100, min_frac_coverage=None, min_frac_crop=None, best_of=None, 
              operating_size=None, bounding_box=False):


    best_crops = []
    best_crop_not_ok = float('-inf'), None, None
    best_bbox_not_ok = float('-inf'), None, None

    seg = seg.astype('bool')

    if operating_size is not None:
        gap = min(seg.shape) // operating_size
        seg = seg[::gap, ::gap]
        image_size_s = image_size[0] // gap, image_size[1] // gap
    else:
        gap = 1
        image_size_s = image_size
    
    if min_frac_crop is not None:
        #min_sum = seg.sum() * min_frac
        # min_sum = seg.shape[0] * seg.shape[1] * min_frac
        min_sum_crop = image_size[0] * image_size[1] * min_frac_crop
    
    min_sum_coverage = (seg.sum() * min_frac_coverage) if min_frac_coverage is not None else 0

    if bounding_box:
        proj_x = np.where(np.array(seg).max(1))[0]
        proj_y = np.where(np.array(seg).max(0))[0]

        # y, x, height, width
        bbox = proj_x[0], proj_y[0], proj_x[-1] - proj_x[0], proj_y[-1] - proj_y[0]

    else:
        bbox = None

    # print(min_sum_coverage, min_sum_crop, seg.shape, image_size)

    for iteration in range(iterations):
        # sl_y, sl_x = random_crop_slices(seg.shape, image_size_s)

        off_y = torch.randint(0, seg.shape[0] - image_size_s[0] + 1, (1,)).item()  # range: 0 <= value < high
        off_x = torch.randint(0, seg.shape[1] - image_size_s[1] + 1, (1,)).item()
        sl_y = slice(max(0, off_y * gap - 1), max(0, off_y * gap - 1) + image_size[0])
        sl_x = slice(max(0, off_x * gap - 1), max(0, off_x * gap - 1) + image_size[1])

        compute_seg_sum = True
        if bbox is not None:
            # try to avoid to compute seg sum
            overlap = bbox_overlap((off_y, off_x, image_size_s[0], image_size_s[1]), bbox)
            if overlap < min_sum_coverage or overlap < min_sum_crop:
                compute_seg_sum = False
                if overlap > best_bbox_not_ok[0]:
                    best_bbox_not_ok = overlap, sl_y, sl_x

                # print('skip', overlap, image_size, min_sum_crop)

        if compute_seg_sum:
            seg_ = seg[sl_y, sl_x]
            sum_seg_ = seg_.sum()

            #sl_y = slice(off_y * gap -1, off_y * gap -1 + image_size[0])
            #sl_x = slice(off_x * gap -1, off_x * gap -1 + image_size[0])

            if sum_seg_ > min_sum_coverage and sum_seg_ > min_sum_crop:

                if best_of is None:
                    return sl_y, sl_x, False
                else:
                    best_crops += [(sum_seg_, sl_y, sl_x)]
                    if len(best_crops) >= best_of:
                        best_crops.sort(key=lambda x:x[0], reverse=True)
                        sl_y, sl_x = best_crops[0][1:]
                        
                        return sl_y, sl_x, False

            else:
                if sum_seg_ > best_crop_not_ok[0]:
                    best_crop_not_ok = sum_seg_, sl_y, sl_x
        
    else:
        if best_crop_not_ok[1] is not None:
            # return best segmentation found (if available)
            return best_crop_not_ok[1:] + (True,) 
        else:
            # otherwise use the best bbox
            return best_bbox_not_ok[1:] + (True,) 



def find_crop_zoom(seg, crop_size_range, iterations=100, min_frac_coverage=None, min_frac_crop=None, bounding_box=False):

    best_crop_not_ok = float('-inf'), None, None
    best_bbox_not_ok = float('-inf'), None, None
    seg = seg.astype('bool')
    
    min_sum_coverage = seg.sum() * min_frac_coverage if min_frac_coverage is not None else 0

    if bounding_box:
        proj_x = np.where(np.array(seg).max(1))[0]
        proj_y = np.where(np.array(seg).max(0))[0]

        # y, x, height, width
        bbox = proj_x[0], proj_y[0], proj_x[-1] - proj_x[0], proj_y[-1] - proj_y[0]

    else:
        bbox = None

    # print(min_sum_coverage, min_sum_crop, seg.shape, image_size)

    skipped = 0

    for iteration in range(iterations):
        # sl_y, sl_x = random_crop_slices(seg.shape, image_size_s)

        if crop_size_range[1] != crop_size_range[0]:
            image_size = torch.randint(crop_size_range[0], crop_size_range[1], (1,)).item()
        else:
            image_size = crop_size_range[0]

        image_size = (image_size, image_size)

        min_sum_crop = image_size[0] * image_size[1] * min_frac_crop if min_frac_crop is not None else 0

        if seg.shape[0] - image_size[0] > 1:
            off_y = torch.randint(0, seg.shape[0] - image_size[0] + 1, (1,)).item()  # range: 0 <= value < high
        else:
            off_y = 0

        if seg.shape[1] - image_size[1] > 1:
            off_x = torch.randint(0, seg.shape[1] - image_size[1] + 1, (1,)).item()
        else: 
            off_x = 0
        sl_y = slice(max(0, off_y - 1), max(0, off_y - 1) + image_size[0])
        sl_x = slice(max(0, off_x - 1), max(0, off_x - 1) + image_size[1])

        compute_seg_sum = True
        if bbox is not None:
            # try to avoid to compute seg sum
            overlap = bbox_overlap((off_y, off_x, image_size[0], image_size[1]), bbox)
            if overlap < min_sum_coverage or overlap < min_sum_crop:
                compute_seg_sum = False
                skipped = True
                if overlap > best_bbox_not_ok[0]:
                    best_bbox_not_ok = overlap, sl_y, sl_x

                # print('skip', overlap, image_size, min_sum_crop)

        if compute_seg_sum:
            seg_ = seg[sl_y, sl_x]
            sum_seg_ = seg_.sum()

            if sum_seg_ > min_sum_coverage and sum_seg_ > min_sum_crop:
                return sl_y, sl_x, False, iteration, skipped
            else:
                if sum_seg_ > best_crop_not_ok[0]:
                    best_crop_not_ok = sum_seg_, sl_y, sl_x
        
    else:
        # print('fail')
        if best_crop_not_ok[1] is not None:
            # return best segmentation found (if available)
            return best_crop_not_ok[1:] + (True, iterations, skipped) 
        else:
            # otherwise use the best bbox
            return best_bbox_not_ok[1:] + (True, iterations, skipped) 


def scale_to_bound(tensor_shape, bounds, interpret_as_max_bound=False):

    sf = bounds[0] / tensor_shape[0], bounds[1] / tensor_shape[1]

    lt = sf[0] < sf[1] if interpret_as_max_bound else sf[1] < sf[0]
    target_size = (bounds[0], int(tensor_shape[1] * sf[0])) if lt else (int(tensor_shape[0] * sf[1]), bounds[1])

    # make sure do not violate the bounds
    if interpret_as_max_bound:
        target_size = min(target_size[0], bounds[0]), min(target_size[1], bounds[1])
    else:
        target_size = max(target_size[0], bounds[0]), max(target_size[1], bounds[1])

    return target_size


def tensor_resize(tensor, target_size_or_scale_factor, interpret_as_min_bound=False, interpret_as_max_bound=False,
                  channel_dim=None, interpolation='bilinear', autoconvert=False, keep_channels_last=False):
    """
    Resizes a tensor (e.g. an image) along two dimensions while
    dimension `channel_dim` remains constant.

    Args:
        tensor: The input tensor
        target_size_or_scale_factor: Depending on the type...
         - a tuple (int, int) it specifies the target size.
           One dimension can be set to None if interpret_as_bound is not set.
         - a float it specifies the scale factor
        channel_dim: dimension of the color channel
        interpolation: Used interpolation mode

    Returns:
        The resized tensor

    Best performance is obtained when `channel_index` is 2.
    """

    assert type(tensor) == np.ndarray

    import cv2
    CV_INTERPOLATIONS = {'nearest': cv2.INTER_NEAREST, 'bilinear': cv2.INTER_LINEAR, 'cubic': cv2.INTER_CUBIC}
    # BROKEN_DTYPES = {'uint8'}  # datatypes that cause weird results for more than 3 channels
    # msg =
    # assert , msg
    BROKEN_DTYPES = {'uint8'}
    if tensor.dtype.name in BROKEN_DTYPES and tensor.ndim > 2 and tensor.shape[2] > 3:
        log.warning('For {} more than three channels give wrong results. Maybe a bug in OpenCV?'.format(BROKEN_DTYPES))

    if autoconvert:
        if tensor.dtype.name in {'int64', 'int32'}:
            log.important('Data is converted from ' + tensor.dtype.name + ' to int16, information might be lost.')
            tensor = tensor.astype('int16')
        elif tensor.dtype.name in {'float16'}:
            log.important('Data is converted from ' + tensor.dtype.name + ' to float32.')
            tensor = tensor.astype('float32')
        elif tensor.dtype.name in {'bool'}:
            log.important('Data is converted from ' + tensor.dtype.name + ' to uint8.')
            tensor = tensor.astype('uint8')
    else:
        if tensor.dtype.name not in {'uint8', 'int16', 'uint16', 'float32', 'float64'}:
            raise TypeError('unsupported datatype (by opencv): ' + tensor.dtype.name)

    if len(tensor.shape) == 2 and channel_dim is not None:
        log.warning('A 2d array is passed to tensor_resize, specifying channel_dim has no effect.')
        channel_dim = None

    # if channel index is not 2 then transpose such that it is.
    if channel_dim == 0:
        tensor = tensor.transpose([1, 2, 0])
    elif channel_dim == 1:
        tensor = tensor.transpose([0, 2, 1])

    if type(target_size_or_scale_factor) in {tuple, list}:
        # scale_factor = None

        if interpret_as_max_bound or interpret_as_min_bound:
            assert not interpret_as_min_bound or not interpret_as_max_bound

            target_size = scale_to_bound(tensor.shape, bounds=target_size_or_scale_factor,
                                         interpret_as_max_bound=interpret_as_max_bound)

        else:

            if target_size_or_scale_factor[0] is None:
                target_size = int(tensor.shape[0] * target_size_or_scale_factor[1] / tensor.shape[1]), target_size_or_scale_factor[1]
            elif target_size_or_scale_factor[1] is None:
                target_size = target_size_or_scale_factor[0], int(tensor.shape[1] * target_size_or_scale_factor[0] / tensor.shape[0])

            else:
                target_size = target_size_or_scale_factor
    elif type(target_size_or_scale_factor) == float:
        scale_factor = target_size_or_scale_factor
        target_size = int(tensor.shape[0] * scale_factor), int(tensor.shape[1] * scale_factor)
    else:
        raise ValueError('target_size must be either a int, float or a tuple of int.')

    log.detail('Resize tensor of shape', tensor.shape, 'and type', tensor.dtype, 'to target size', target_size)
    tensor = cv2.resize(tensor, (target_size[1], target_size[0]), interpolation=CV_INTERPOLATIONS[interpolation])

    if not keep_channels_last:
        if channel_dim == 0:
            tensor = tensor.transpose([2, 0, 1])
        elif channel_dim == 1:
            tensor = tensor.transpose([0, 2, 1])

    return tensor



class LVIS_OneShotBase(object):

    repository_files = ['LVIS_OneShot3b.tar']
    # repository_files = [
    #     ('LVIS_OneShot/dataset_new.json', 'dataset_new.json'),
    #     'LVIS_OneShot/precomputed_new.tar',
    # ]

    @staticmethod
    def check_data_integrity(data_path):

        print(len(os.listdir(join(data_path, 'all'))) == 815)
        print(isfile(join(data_path, 'all_splits.json')))

        return all([
            len(os.listdir(join(data_path, 'all'))) == 815,
            isfile(join(data_path, 'all_splits.json')),
            # len(os.listdir(join(self.data_path(), 'train'))) == 739,
            # len(os.listdir(join(self.data_path(), 'val'))) == 20,
            # len(os.listdir(join(self.data_path(), 'test'))) == 56,
        ])

    def __init__(self, split, split_mode='pascal_test', aug=0, mask='separate', replace=False, image_size=(400, 400), 
                 with_class_label=False, text_class_labels=False, min_area=None,
                 normalize=False, add_bar=True, reduce_factor=None, category_diversity=None, negative_prob=0.0,
                 min_frac_s=None, min_frac_q=None, pre_crop_scale=None, pre_crop_image_size=None, 
                 crop_best_of=None, balanced_sampling=False, seed=None, fix_find_crop=False, cache=None):
        """
            mask: e.g. separate, overlay or text_label
        """
        super().__init__()
        self.split = split
        self.aug = aug

        self.balanced_sampling = balanced_sampling

        self.add_bar = add_bar
        self.mask = mask
        self.image_size = image_size if type(image_size) in {list, tuple} else [image_size]*2

        self.find_crop = find_crop_old if not fix_find_crop else find_crop

        # Scale to a certain minimum size to avoid sampling from small images
        if isinstance(pre_crop_image_size, float):
            self.pre_crop_image_size = [int(pre_crop_image_size * s) for s in self.image_size]
        elif type(pre_crop_image_size) in {list, tuple} and len(pre_crop_image_size) == 2:
            self.pre_crop_image_size = pre_crop_image_size
        elif type(pre_crop_image_size) in {list, tuple} and pre_crop_image_size[0] in {'crop', 'sample'}:
            self.pre_crop_image_size = list(pre_crop_image_size)
            assert len(pre_crop_image_size) == 3
        elif pre_crop_image_size is None:
            self.pre_crop_image_size = None
        else:
            self.pre_crop_image_size = [pre_crop_image_size]*2
        
        print(self.pre_crop_image_size)

        assert self.pre_crop_image_size is None or (type(self.pre_crop_image_size) == list and len(self.pre_crop_image_size) in {2, 3})

        self.seed = seed
        self.replace = replace

        self.text_class_labels = text_class_labels
        self.with_class_label = with_class_label

        self.negative_prob = negative_prob
        self.min_frac_s = min_frac_s  # minimal visible support object fraction
        self.min_frac_q = min_frac_q  # minimal visible query object fraction
        self.pre_crop_scale = pre_crop_scale
        self.crop_best_of = crop_best_of  # chose the best of N crops

        self.exceed_count = 0

        self.data_path = lambda : expanduser('~/datasets/LVIS_OneShot3')

        # TODO: fix the integrity check function
        get_from_repository('LVIS_OneShot3', ['LVIS_OneShot3b.tar'], integrity_check=self.check_data_integrity)
        
        if normalize:
            from torchvision.transforms import Compose, Normalize
            value_scale = 255
            mean = [item * value_scale for item in [0.485, 0.456, 0.406]]
            std = [item * value_scale for item in [0.229, 0.224, 0.225]]
            self.norm_transform = Compose([Normalize(mean=mean, std=std)])
        else:
            self.norm_transform = None

        self.category_info = json.load(open(join(dirname(realpath(__file__)), 'category_info.json')))
        self.category_names = {c['id']: c['name'] for c in self.category_info}

        if aug == 1:

            import albumentations as A
            albu_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.HueSaturationValue(p=0.25, hue_shift_limit=10*aug, sat_shift_limit=15*aug, val_shift_limit=10*aug),
                A.RandomBrightnessContrast(p=0.25, brightness_limit=0.05, contrast_limit=0.05),
            ])
            def aug_transform(image, mask):
                out = albu_transform(image=image, mask=mask)
                out['image'] = torch.from_numpy(out['image'] / 255).permute(2,0,1)
                return out

            self.aug_transform = aug_transform            

        elif aug == 'flip':
            from torchvision.transforms import functional as tf
            from torch.distributions import Uniform

            def aug_transform(image, mask):

                image = torch.from_numpy(image).permute(2,0,1).float() / 255
                mask = torch.from_numpy(mask)
                if torch.rand(1).item() > 0.5:
                    image = tf.hflip(image)
                    mask = tf.hflip(mask)

                return {'image': image, 'mask': mask.numpy()}
            self.aug_transform = aug_transform


        elif aug == '1new':
            from torchvision.transforms import functional as tf
            from torch.distributions import Uniform

            def aug_transform(image, mask):

                image = torch.from_numpy(image).permute(2,0,1).float() / 255
                mask = torch.from_numpy(mask)
                if torch.rand(1).item() > 0.5:
                    image = tf.hflip(image)
                    mask = tf.hflip(mask)
                
                image = tf.adjust_brightness(image, Uniform(0.5, 1.5).sample())
                image = tf.adjust_contrast(image, Uniform(0.6, 1.4).sample())

                # image = tf.adjust_hue(image, Uniform(-0.05, 0.05).sample())

                if torch.rand(1).item() > 0.5:
                    image = tf.adjust_saturation(image, Uniform(0, 1.5).sample())

                return {'image': image, 'mask': mask.numpy()}
            self.aug_transform = aug_transform

        elif aug == 2:
            import albumentations as A
            albu_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.HueSaturationValue(p=1, hue_shift_limit=10*aug, sat_shift_limit=15*aug, val_shift_limit=10*aug),
                A.RandomBrightnessContrast(p=0.2, brightness_limit=0.1*aug, contrast_limit=0.1*aug),
                A.RandomFog(0.3, 0.6, p=0.2),
                A.RandomShadow(p=0.2),
                A.ElasticTransform(p=0.1, alpha=10, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GaussNoise(p=0.2, var_limit=(0.0, 20*aug))
            ])
            def aug_transform(image, mask):
                out = albu_transform(image=image, mask=mask)
                out['image'] = torch.from_numpy(out['image'] / 255).permute(2,0,1)
                return out

            self.aug_transform = aug_transform

        elif aug is None or aug == 0:
            def aug_transform(image, mask):
                return {
                    'image': torch.from_numpy(image).permute(2,0,1).float() / 255, 
                    'mask': mask
                }
            self.aug_transform = aug_transform
        else:
            raise ValueError('invalid value for aug')

        assert split_mode in {'pascal_test', 'pascal_test02', 'rand100', 'old_wrong'}
        split_categories = json.load(open(join(dirname(realpath(__file__)), 'all_splits.json')))
        self.categories = split_categories[split_mode][split if split != 'train_fixed' else 'train']


        log.info('SPLIT', split_mode, split, 'AUG', aug, 'REDUCE', reduce_factor)
        
        self.samples = dict()
        for c in self.categories:
            self.samples[c] = []
            for file in os.listdir(join(self.data_path(), 'all', str(c))):
                if file.endswith('.jpg'):
                    self.samples[c] += [join(self.data_path(), 'all', str(c), file[:-4])]

        if min_area is not None:
            import pickle
            
            area_per_sample = pickle.load(open('lvis_sample_areas.pickle', 'rb'))
            self.samples = {k: [s for s in v if area_per_sample[(k, basename(s))] > min_area] for k, v in self.samples.items()}

        # self.categories = os.listdir(join(self.data_path(), split))
        # self.samples = dict()
        # for c in self.categories:
        #     self.samples[c] = []
        #     for file in os.listdir(join(self.data_path(), split, c)):
        #         if file.endswith('.jpg'):
        #             self.samples[c] += [join(self.data_path(), split, c, file[:-4])]
        
        # remove categories with less than 2 samples
        removed = []
        for c in self.categories:
            if len(self.samples[c]) < 2:
                del self.samples[c]
                removed += [c]
        log.info(f'removed {[self.category_info[int(c)]["name"] for c in removed]}')
        self.categories = [c for c in self.categories if c not in removed]

        log.info(f'{len(self.categories)} categories remain')

        if reduce_factor is not None:
            self.samples = {c: samples[:max(2, int(len(samples) * reduce_factor))] for c, samples in self.samples.items()}

        if category_diversity is not None and split == 'train':

            target_samples, n_categories = category_diversity
            n_categories = len(self.categories) if n_categories is None else n_categories

            np.random.seed(123 if self.seed is None else self.seed)
            def sample_sum(cat_ids):
                return sum([len(self.samples[c]) for c in cat_ids])

            def samples_per_cat(c):
                return len(self.samples[c])

            print(len(self.categories), n_categories)
            # samples_per_cat = {c: len(self.samples[c]) for c in self.categories}
            cat_ids = np.random.choice(self.categories, n_categories, replace=False)

            while sample_sum(cat_ids) > target_samples and len(cat_ids) != len(self.categories):
                largest_i = sorted([(i, samples_per_cat(c)) for i, c in enumerate(cat_ids)], key=lambda x: x[1])[-1][0]
                new_cat = np.random.choice(list(set(self.categories).difference(cat_ids)), 1)[0]
                cat_ids[largest_i] = new_cat

            while sample_sum(cat_ids) < target_samples and len(cat_ids) != len(self.categories):
                smallest_i = sorted([(i, samples_per_cat(c)) for i, c in enumerate(cat_ids)], key=lambda x: x[1])[0][0]
                new_cat = np.random.choice(list(set(self.categories).difference(cat_ids)), 1)[0]
                cat_ids[smallest_i] = new_cat

            self.categories = cat_ids

            assert len(self.categories) == len(set(self.categories))
            
            keep, i = {c: 0 for c in self.categories}, 0
            while i < target_samples:
                for c in self.categories:
                    if keep[c] < samples_per_cat(c):
                        keep[c] += 1
                        i += 1

                    if i >= target_samples:
                        break 

            for c in self.categories:
                assert len(self.samples[c]) >= keep[c]
                self.samples[c] = self.samples[c][:keep[c]]
                
            log.info(sample_sum(self.categories))

        # self.samples = {c: v for c in self.categories for v in self.samples[c] if len(v) >= 2}
        assert all(len(self.samples[c]) > 1 for c in self.categories)

        if self.balanced_sampling:
            self.category_weights = [1/len(self.categories) for _ in self.categories]
        else:
            self.category_weights = [len(self.samples[c]) for c in self.categories]
            sum_samples = sum(self.category_weights)
            self.category_weights = [w / sum_samples for w in self.category_weights]

        self.category_weights = torch.tensor(self.category_weights)

        # just an arbitrary number of sample. For val and test the set should be identical
        if split in {'val', 'test', 'train_fixed'}:  # precompute for consistency
            np.random.seed(123 if self.seed is None else self.seed)

            # Note, this will affect another train dataset.
            torch.manual_seed(123 if self.seed is None else self.seed)

            self.sample_ids = []
            for _ in range({'val': 2000, 'test': 10000, 'train_fixed': 50000}[split]):

                category = self.categories[torch.multinomial(self.category_weights, 1).item()]
                si1, si2 = torch.multinomial(torch.ones(len(self.samples[category])), 2, replacement=False)
                # si1, si2 = [self.samples[category][i] for i in sample_indices]
                # si1, si2 = np.random.choice(len(self.samples[category]), 2, replace=False)

                if self.negative_prob > 0 and np.random.rand() < self.negative_prob:
                    category_q = self.categories[torch.multinomial(self.category_weights, 1).item()]
                    si2 = np.random.choice(len(self.samples[category_q]), 1, replace=False)[0]
                else:
                    category_q = category

                self.sample_ids += [(category, category_q, si1, si2)]    

        elif split == 'train':
            torch.manual_seed(123 if self.seed is None else self.seed)
            self.sample_ids = [None] * 50000

        log.info('total number of samples', sum(len(self.samples[c]) for c in self.categories))

        if cache is not None:
            # cache a fixed number of samples per category

            self.samples = {k: v[:cache] for k, v in self.samples.items()}

            # self.samples = {k: self.samples[k] for k in list(self.samples.keys())[:10]}
            print('to be cached', sum([len(v) for v in self.samples.values()]))

            self.cached_img = {k: [np.array(Image.open(fn + '.jpg')) for fn in v] for k, v in self.samples.items()}
            self.cached_seg = {k: [np.array(Image.open(fn + '-seg.png')) for fn in v] for k, v in self.samples.items()}
        else:
            self.cached_img = None


    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):

        sid = self.sample_ids[idx]

        if sid is None:
            # sample category

            category = int(torch.multinomial(self.category_weights, 1, replacement=True)[0])
            # category = int(torch.multinomial(torch.ones(len(self.category_weights)), 1, replacement=True)[0])
            category = self.categories[category]
            category_q = category
            si1, si2 = torch.multinomial(torch.ones(len(self.samples[category])), 2, replacement=self.replace)
            si1, si2 = int(si1), int(si2)

            # category = np.random.choice(self.categories, p=self.category_weights)
            # si1, si2 = np.random.choice(len(self.samples[category]), 2, replace=self.replace)
        else:
            category, category_q, si1, si2 = sid

        if self.split == 'train' and self.negative_prob > 0:
            if torch.rand(1).item() < self.negative_prob:
                category_q = self.categories[torch.multinomial(self.category_weights, 1, replacement=True).item()]
                si2 = torch.multinomial(torch.ones(len(self.samples[category_q])), 1, replacement=self.replace).item()

        sample_s = self.samples[category][si1]
        sample_q = self.samples[category_q][si2]

        if self.cached_img is not None:
            img_s = self.cached_img[category][si1]
            seg_s = self.cached_seg[category][si1].astype('uint8')
            img_q = self.cached_img[category_q][si2]
            seg_q = self.cached_seg[category][si2].astype('uint8')

            if category != category_q:
                seg_q = np.zeros((img_q.shape[0], img_q.shape[1]))

        else:
            img_s = np.array(Image.open(sample_s + '.jpg'))
            seg_s = np.array(Image.open(sample_s + '-seg.png')).astype('uint8')
            
            img_q = np.array(Image.open(sample_q + '.jpg'))

            # print(img_s.shape)

            if category == category_q:
                seg_q = np.array(Image.open(sample_q + '-seg.png')).astype('uint8')
            else:
                seg_q = np.zeros((img_q.shape[0], img_q.shape[1]))

        assert seg_q.sum() > 0 or category != category_q, f'category {category_q}: {sample_q}'
        assert seg_s.sum() > 0, f'category {category}: {sample_s}'

        if img_s.ndim == 2:
            img_s = np.dstack([img_s]*3)

        if img_q.ndim == 2:
            img_q = np.dstack([img_q]*3)            

        # seg = np.array(seg).astype('uint8')
        # print(filename, seg.mean(), seg.sum())

        # print(seg.max(), seg.dtype)

        if self.pre_crop_image_size is None or self.pre_crop_image_size[0] != 'crop':

            # make sure the images are larger than the crop sizes
            new_min_size = (int(self.image_size[0]*1.1), int(self.image_size[1]*1.1))
            if img_q.shape[0] < self.image_size[0] or img_q.shape[1] < self.image_size[1]:
                assert img_q.shape[:2] == seg_q.shape[:2]
                
                img_q = tensor_resize(img_q, new_min_size, interpret_as_min_bound=True)
                seg_q = tensor_resize(seg_q, new_min_size, interpret_as_min_bound=True)

            if img_s.shape[0] < self.image_size[0] or img_s.shape[1] < self.image_size[1]:
                assert img_s.shape[:2] == seg_s.shape[:2]
                img_s = tensor_resize(img_s, new_min_size, interpret_as_min_bound=True)
                seg_s = tensor_resize(seg_s, new_min_size, interpret_as_min_bound=True)

        # IMPORTANT: This was not used in the original MaRF code.
        # avoid too large images
        if self.pre_crop_scale is not None:
            
            if type(self.pre_crop_scale) in {int, float}:
                pc_s, pc_q = self.pre_crop_scale, self.pre_crop_scale
            elif type(self.pre_crop_scale) in {list, tuple}:
                # sample uniformly
                pc_s = self.pre_crop_scale[0] + torch.rand(1).item() * (self.pre_crop_scale[1] - self.pre_crop_scale[0])
                pc_q = self.pre_crop_scale[0] + torch.rand(1).item() * (self.pre_crop_scale[1] - self.pre_crop_scale[0])

            # +1 should avoid a rare bug where images become smaller than image_size
            new_max_size_s = (int(self.image_size[0]*pc_s + 1), int(self.image_size[1]*pc_s + 1))
            new_max_size_q = (int(self.image_size[0]*pc_q + 1), int(self.image_size[1]*pc_q + 1))
            img_q = tensor_resize(img_q, new_max_size_q, interpret_as_max_bound=True)
            seg_q = tensor_resize(seg_q, new_max_size_q, interpret_as_max_bound=True)

            img_s = tensor_resize(img_s, new_max_size_s, interpret_as_max_bound=True)
            seg_s = tensor_resize(seg_s, new_max_size_s, interpret_as_max_bound=True)


        need_to_crop = True

        if self.pre_crop_image_size is not None and self.pre_crop_image_size[0] == 'crop':
            # crop on full-resolution and rescale then
            assert self.image_size[0] == self.image_size[1]
            assert seg_s.shape[0] == seg_s.shape[1]

            # define crop size range by target (crop) image size
            #crop_size_range = int(self.pre_crop_image_size[1] * self.image_size[0]), int(self.pre_crop_image_size[2] * self.image_size[0])
            #crop_size_range = min(seg_s.shape[0], crop_size_range[0]), min(seg_s.shape[0], crop_size_range[1])

            # define crop size range by image
            crop_size_range = int(self.pre_crop_image_size[1] * seg_s.shape[0]), int(self.pre_crop_image_size[2] * seg_s.shape[0])

            sl_s_y, sl_s_x, _, _, _ = find_crop_zoom(
                seg_s, crop_size_range, iterations=500, 
                min_frac_coverage=self.min_frac_s, min_frac_crop=self.min_frac_s,
                bounding_box=True
            )

            if category == category_q:
                sl_q_y, sl_q_x, exceed, _, _ = find_crop_zoom(
                    seg_q, crop_size_range, iterations=500,
                    min_frac_coverage=self.min_frac_q, min_frac_crop=self.min_frac_s,
                    bounding_box=True
                )
            else:
                sl_q_y, sl_q_x = slice(0, self.image_size[0]), slice(0, self.image_size[1])

            seg_q = seg_q[sl_q_y, sl_q_x]
            seg_s = seg_s[sl_s_y, sl_s_x]

            img_q = img_q[sl_q_y, sl_q_x]
            img_s = img_s[sl_s_y, sl_s_x]            

            from torchvision.transforms.functional import resize

            img_q = resize(torch.from_numpy(img_q).permute(2,0,1), self.image_size).permute(1,2,0).numpy()
            seg_q = resize(torch.from_numpy(seg_q).unsqueeze(0), self.image_size, interpolation=InterpolationMode.NEAREST).numpy()[0]

            img_s = resize(torch.from_numpy(img_s).permute(2,0,1), self.image_size).permute(1,2,0).numpy()
            seg_s = resize(torch.from_numpy(seg_s).unsqueeze(0), self.image_size, interpolation=InterpolationMode.NEAREST).numpy()[0]


            #img_q = tensor_resize(img_q, self.image_size, interpret_as_min_bound=True)
            #seg_q = tensor_resize(seg_q, self.image_size, interpret_as_min_bound=True)      

            #img_s = tensor_resize(img_s, self.image_size, interpret_as_min_bound=True)
            #seg_s = tensor_resize(seg_s, self.image_size, interpret_as_min_bound=True)                

            need_to_crop = False

        elif self.pre_crop_image_size is not None:

            if self.pre_crop_image_size[0] == 'sample':
                pc_s = torch.distributions.Uniform(self.pre_crop_image_size[1], self.pre_crop_image_size[2]).sample()
                pc_q = torch.distributions.Uniform(self.pre_crop_image_size[1], self.pre_crop_image_size[2]).sample()

                # +1 should avoid a rare bug where images become smaller than image_size
                pre_crop_size_s = (int(self.image_size[0]*pc_s + 1), int(self.image_size[1]*pc_s + 1))
                pre_crop_size_q = (int(self.image_size[0]*pc_q + 1), int(self.image_size[1]*pc_q + 1))      

            else:
                pre_crop_size_s = self.pre_crop_image_size
                pre_crop_size_q = self.pre_crop_image_size

            img_q = tensor_resize(img_q, pre_crop_size_q, interpret_as_min_bound=True)
            seg_q = tensor_resize(seg_q, pre_crop_size_q, interpret_as_min_bound=True)      

            img_s = tensor_resize(img_s, pre_crop_size_s, interpret_as_min_bound=True)
            seg_s = tensor_resize(seg_s, pre_crop_size_s, interpret_as_min_bound=True)                      

        if need_to_crop:
            if category == category_q:
                sl_q_y, sl_q_x, exceed = self.find_crop(seg_q, self.image_size, iterations=200,
                                                        min_frac_coverage=self.min_frac_q, min_frac_crop=self.min_frac_s,
                                                        best_of=self.crop_best_of, bounding_box=True)
                if exceed:
                    self.exceed_count += 1
            else:
                sl_q_y, sl_q_x = slice(0, self.image_size[0]), slice(0, self.image_size[1])

            sl_s_y, sl_s_x, exceed = self.find_crop(seg_s, self.image_size, iterations=200, 
                                                    min_frac_coverage=self.min_frac_s, min_frac_crop=self.min_frac_s,
                                                    best_of=self.crop_best_of, bounding_box=True)

            # if exceed:
            #     self.exceed_count += 1

            if self.exceed_count > 500:
                raise ValueError('More than 500 exceeds')

            if self.exceed_count % 50 == 49:
                print(f'exceeds: {self.exceed_count}')

            seg_q = seg_q[sl_q_y, sl_q_x]
            seg_s = seg_s[sl_s_y, sl_s_x]

            if seg_s.sum() == 0:
                print('Empty support image:', sid, sample_s, self.samples[category][si1])

            img_q = img_q[sl_q_y, sl_q_x]
            img_s = img_s[sl_s_y, sl_s_x]

        seg_q = seg_q.astype('float32')

        t_s = self.aug_transform(image=img_s, mask=seg_s)
        img_s, seg_s = t_s['image'], t_s['mask']
        t_q = self.aug_transform(image=img_q, mask=seg_q)
        img_q, seg_q = t_q['image'], t_q['mask']

        if self.add_bar:
            for (_img, _seg) in [(img_s, seg_s), (img_q, seg_q)]:
                if torch.rand(1)[0] < 0.5:
                    k = int(25 + torch.rand(1)[0] * 50)
                    if torch.rand(1)[0] < 0.5:
                        if _seg[k:-k].sum() / _seg.sum() > 0.5: 
                            _seg[:k] = 0
                            _seg[-k:] = 0
                            _img[:, :k] = 0
                            _img[:, -k:] = 0
                    else:
                        if _seg[:, k:-k].sum() / _seg.sum() > 0.5: 
                            _seg[:, :k] = 0
                            _seg[:, -k:] = 0
                            _img[:, :, :k] = 0
                            _img[:, :, -k:] = 0

        seg_q = seg_q.reshape((1, ) + seg_q.shape)
        # seg_s = seg_s.reshape((1, ) + seg_s.shape)

        if self.norm_transform is not None:
            img_q = self.norm_transform(img_q * 255).numpy()
            img_s = self.norm_transform(img_s * 255).numpy()
        else:
            img_q = (img_q * 255).numpy()
            img_s = (img_s * 255).numpy()

        mask = self.mask

        if mask == 'mix_text_overlay':
            mask = 'text_label' if torch.rand(1).item() > 0.5 else 'overlay'

        label_name = self.category_names[int(category)]
        label_name = label_name.replace('_', ' ')

        if mask == 'text_label':
            # DEPRECATED
            img_s_masked = [int(category)]
        elif mask == 'text':
            img_s_masked = [label_name]      
        # elif mask == 'text_and_overlay':
        #     img_s_masked = [label_name] + blend_image_segmentation(img_s, seg_s, mode='overlay')
        # elif mask == 'text_and_highlight':
        #     img_s_masked = [label_name] + blend_image_segmentation(img_s, seg_s, mode='highlight')
        # elif mask == 'text_and_blur_highlight':
        #     img_s_masked = [label_name] + blend_image_segmentation(img_s, seg_s, mode='blur_highlight')
        elif mask is None:
            img_s_masked = [img_s]
        else:
            if mask.startswith('text_and_'):
                mask = mask[9:]
                label_add = [label_name]
            else:
                label_add = []

            img_s_masked = label_add + blend_image_segmentation(img_s, seg_s, mode=mask)

        #img_q = img_q.transpose([2,0,1])
        #img_s_masked[0] = img_s_masked[0].transpose([2,0,1])
        # img_s_masked[1] = img_s_masked[1].astype('float32')

        # print(img_q.shape, img_s_masked[0].shape)

        img_q = img_q.astype('float32')
        # img_s = img_s.astype('float32')

        if self.with_class_label:
            if self.text_class_labels:
                label = (torch.zeros(0), label_name) if self.with_class_label else ()
            else:
                label = (torch.zeros(0), int(category)) if self.with_class_label else ()

        else:
            label = ()


        # data_x: query image, (support image, support mask) | (label, blended support image)
        # data_y: ground truth, validity mask (ignored in this dataset), class id

        return (img_q, *img_s_masked), (seg_q,) + label


# for legacy code:
LVIS_OneShot3 = LVIS_OneShotBase


# split_mode: pascal_test
#   split: train
#   mask: text_and_blur3_highlight
#   prompt: shuffle+
#   image_size: 224
#   add_bar: False
#   normalize: True
#   pre_crop_image_size: 400
  
#   min_frac_s: 0.07
#   min_frac_q: 0.07
#   min_area: 0.1



class LVIS_OneShot(LVIS_OneShotBase):

    def __init__(self, split, split_mode='pascal_test', mask='separate', image_size=224, aug=0, normalize=True, with_class_label=False, 
                 reduce_factor=None, balanced_sampling=False, pre_crop_image_size=1.7, min_frac=0.03, min_area=0.05, negative_prob=0, cache=None):
        
        super().__init__(split=split, split_mode=split_mode, mask=mask, image_size=image_size, aug=aug, normalize=normalize, reduce_factor=reduce_factor,
                         balanced_sampling=balanced_sampling, pre_crop_image_size=pre_crop_image_size, negative_prob=negative_prob,
                         min_frac_s=min_frac, min_frac_q=min_frac, min_area=min_area, add_bar=False, with_class_label=with_class_label,
                         fix_find_crop=True, cache=cache)


class LVIS_Affordance(LVIS_OneShotBase):

    def __init__(self, split, affordance, use_prompt=None, split_mode='pascal_test', mask='separate', image_size=224, aug=0, normalize=True, with_class_label=False, 
                 reduce_factor=None, balanced_sampling=False, 
                 pre_crop_image_size=1.001, min_frac=0.02, min_area=0.03, # <- these arguments are different from the standard LVISOneShot
                 negative_prob=0, seed=None):
        
        super().__init__(split=split, split_mode=split_mode, mask=mask, image_size=image_size, aug=aug, normalize=normalize, reduce_factor=reduce_factor,
                         balanced_sampling=balanced_sampling, pre_crop_image_size=pre_crop_image_size, negative_prob=negative_prob,
                         min_frac_s=min_frac, min_frac_q=min_frac, min_area=min_area, add_bar=False, with_class_label=with_class_label,
                         fix_find_crop=True, seed=seed)                         

        if type(affordance) == str:
            affordance = [affordance]

        self.affordance = affordance
        self.affordances_table = {

            # affordances
            'sit on': ('armchair', 'sofa', 'loveseat', 'deck_chair', 'rocking_chair', 'highchair', 'deck_chair', 'folding_chair', 'chair', 'recliner', 'wheelchair'),
            'drink from': ('bottle', 'beer_bottle', 'water_bottle', 'wine_bottle', 'thermos_bottle'),
            'ride on': ('horse', 'pony', 'motorcycle'),

            # abilities
            'can fly': ('eagle', 'jet_plane', 'airplane', 'fighter_jet', 'bird', 'duck', 'gull', 'owl', 'seabird', 'pigeon', 'goose', 'parakeet'),
            'can be driven': ('minivan', 'bus_(vehicle)', 'cab_(taxi)', 'jeep', 'ambulance', 'car_(automobile)'),
            'can swim': ('duck', 'duckling', 'water_scooter', 'penguin', 'boat', 'kayak', 'canoe'),

            # meronymy
            'has wheels': ('dirt_bike', 'car_(automobile)', 'wheelchair', 'motorcycle', 'bicycle', 'cab_(taxi)',  'minivan', 
                           'bus_(vehicle)', 'cab_(taxi)', 'jeep', 'ambulance'),
            'has legs': ('armchair', 'sofa', 'loveseat', 'deck_chair', 'rocking_chair', 'highchair', 'deck_chair', 'folding_chair', 
                         'chair', 'recliner', 'wheelchair', 'horse', 'pony', 'eagle', 'bird', 'duck', 'gull', 'owl', 'seabird', 
                         'pigeon', 'goose', 'parakeet', 'dog', 'cat', 'flamingo', 'penguin', 'cow', 'puppy', 'sheep', 'black_sheep', 
                         'ostrich',
                         'ram_(animal)', 'chicken_(animal)', 'person'),

            'has ears': ('horse', 'pony', 'dog', 'cat', 'flamingo', 'penguin', 'cow', 'puppy', 'sheep', 'black_sheep', 'person'),

           # 'has wings': (),

            # categories
            # 'is an electric device': ('television_set'),
            #'vehicle': (),
            #'bird': (),
            #'flying things': (),
        }

        # shrink affordaces table based on specified affordances
        self.affordances_table = {k: v for k, v in self.affordances_table.items() if k in self.affordance}

        self.prompts = {k: k for k in self.affordances_table.keys()}

        # map: category -> [aff1, aff2, ...]
        self.aff_by_category = {self.category_names[int(c)]: [] for c in self.categories}
        for k, classes in self.affordances_table.items():
            for c in classes:
                # since aff_by_category is based on acually occurring categories, there is no guarantee c is in.
                if c in self.aff_by_category:  
                    self.aff_by_category[c] += [k]

        self.aff_by_category = {k.replace('_', ' '): v for k,v in self.aff_by_category.items()}


        self.prompt_map = {c: self.prompts[aff] for aff in affordance for c in self.affordances_table[aff]}

        old_categories, old_category_weights = self.categories, self.category_weights

        # modify LVIS to incorporate affordances
        self.categories = [c for c in self.categories if self.category_names[int(c)] in self.prompt_map]
 
        self.category_weights = torch.ones(len(self.categories))
        # self.samples = {k: v for k,v in self.samples.items() if k in self.categories}

        self.sample_ids = [(k, k, s, s) for k in self.categories for s in range(len(self.samples[k]))]

        if negative_prob > 0:
            for i in range(len(self.sample_ids)):
                if np.random.rand() < negative_prob:
                    k1, k2, si1, si2 = self.sample_ids[i]
                    
                    while True:
                        category_q = old_categories[torch.multinomial(old_category_weights, 1).item()]

                        # category is ok if not in prompt map...
                        if self.category_names[int(category_q)] not in self.prompt_map:
                            break
                        else:
                            # or if prompt is different from old one
                            new_aff = self.prompt_map[self.category_names[int(category_q)]]
                            aff = self.prompt_map[self.category_names[int(k1)]]
                            if new_aff != aff:
                                break
                    
                    # print(self.category_names[int(k1)], self.category_names[int(category_q)])
                    si2 = np.random.choice(len(self.samples[category_q]), 1, replace=False)[0]
                    self.sample_ids[i] = (k1, category_q, si1, si2)

        
        # self.category_names = {int(c): self.prompt_map[self.category_names[int(c)]] if use_prompt is None else use_prompt for c in self.categories}

        self.mask = 'text'

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        (img_q, label), data_y = out

        possible_labels = self.aff_by_category[label]
        label = possible_labels[torch.randint(0, len(possible_labels), (1,)).item()]

        return (img_q, label), data_y