import glob
import os

import numpy as np
import pandas as pd

# from . import ClsBaseDataset
from .utils import DatasetDescriptor
from ..transforms import build_transforms

from PIL import Image, ImageOps
import torch
from torch.utils import data


class ClsBaseDataset(data.Dataset):
    """
    Base class for segmentation dataset.
    Arguments:
        root: Str, root directory.
        split: Str, data split, e.g. train/val/test.
        is_train: Bool, for training or testing.
        crop_size: Tuple, crop size.
        mirror: Bool, whether to apply random horizontal flip.
        min_scale: Float, min scale in scale augmentation.
        max_scale: Float, max scale in scale augmentation.
        scale_step_size: Float, step size to select random scale.
        mean: Tuple, image mean.
        std: Tuple, image std.
    """
    def __init__(self,
                 root,
                 split,
                 is_train=True,
                 crop_size=(513, 1025),
                 mirror=True,
                 min_scale=0.5,
                 max_scale=2.,
                 scale_step_size=0.25,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.root = root
        self.split = split
        self.is_train = is_train

        self.crop_h, self.crop_w = crop_size

        self.mirror = mirror
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_step_size = scale_step_size

        self.mean = mean
        self.std = std

        self.pad_value = tuple([int(v * 255) for v in self.mean])

        # ======== override the following fields ========
        self.ignore_label = 255
        self.label_pad_value = (self.ignore_label, )
        self.label_dtype = 'uint8'

        # list of image filename (required)
        self.img_list = []
        # list of label filename (required)
        self.ann_list = []
        # list of instance dictionary (optional)
        self.ins_list = []

        self.has_instance = False
        self.label_divisor = 1000

        self.raw_label_transform = None
        self.pre_augmentation_transform = None
        self.transform = None
        self.target_transform = None

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        # TODO: handle transform properly when there is no label
        # print("get item, index: {}".format( index ))
        dataset_dict = {}
        assert os.path.exists(self.img_list[index]), 'Path does not exist: {}'.format(self.img_list[index])
        print("to read image {}".format( self.img_list[index] ))
        image = self.read_image(self.img_list[index], 'RGB')
        # print("got image")
        if not self.is_train:
            # Do not save this during training.
            dataset_dict['raw_image'] = image.copy()

        if self.ann_list is not None:
            # assert os.path.exists(self.ann_list[index]), 'Path does not exist: {}'.format(self.ann_list[index])
            # print("to read ann {}".format( self.ann_list[index] ))
            # label = self.read_label(self.ann_list[index], self.label_dtype)
            label = self.ann_list[index]
            # print("got ann")
        else:
            label = None

        # raw_label = label.copy()
        # if self.raw_label_transform is not None:
        #     raw_label = self.raw_label_transform(raw_label, self.ins_list[index])['semantic']
        # if not self.is_train:
            # Do not save this during training
        #     dataset_dict['raw_label'] = raw_label
        size = image.shape
        dataset_dict['raw_size'] = np.array(size)
        # To save prediction for official evaluation.
        # name = os.path.splitext(os.path.basename(self.ann_list[index]))[0]
        # TODO: how to return the filename?
        # dataset_dict['name'] = np.array(name)
        #print("to aug transform...")
        # Resize and pad image to the same size before data augmentation.
        #if self.pre_augmentation_transform is not None:
        #    image, label = self.pre_augmentation_transform(image, label)
        #    size = image.shape
        #    dataset_dict['size'] = np.array(size)
        #else:
        #    dataset_dict['size'] = dataset_dict['raw_size']
        dataset_dict['size'] = dataset_dict['raw_size']
        #print("to transform ... image {}".format( size ))
        # Apply data augmentation.
        if self.transform is not None:
            image, label = self.transform(image, label)

        dataset_dict['image'] = image
        #if not self.has_instance:
        dataset_dict['semantic'] = torch.as_tensor(label.astype('long'))
        return dataset_dict
        #print("to generate target...")
        # Generate training target.
        #if self.target_transform is not None:
        #    label_dict = self.target_transform(label, self.ins_list[index])
        #    for key in label_dict.keys():
         #       dataset_dict[key] = label_dict[key]
        #print("got item, image: {}".format( dataset_dict['image'].size() ))
        # return dataset_dict

    @staticmethod
    def read_image(file_name, format=None):
        image = Image.open(file_name)

        # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format == "BGR":
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)
        if format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]
        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)
        return image

    @staticmethod
    def read_label(file_name, dtype='uint8'):
        # In some cases, `uint8` is not enough for label
        label = Image.open(file_name)
        return np.asarray(label, dtype=dtype)

    def reverse_transform(self, image_tensor):
        """Reverse the normalization on image.
        Args:
            image_tensor: torch.Tensor, the normalized image tensor.
        Returns:
            image: numpy.array, the original image before normalization.
        """
        dtype = image_tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=image_tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=image_tensor.device)
        image_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
        image = image_tensor.mul(255)\
                            .clamp(0, 255)\
                            .byte()\
                            .permute(1, 2, 0)\
                            .cpu().numpy()
        return image

    @staticmethod
    def train_id_to_eval_id():
        return None


class PavDataset(ClsBaseDataset):
    def __init__(self,
                 root,
                 split,
                 is_train=True,
                 crop_size=(513, 1025),
                 mirror=True,
                 min_scale=0.5,
                 max_scale=2.,
                 scale_step_size=0.25,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 semantic_only=False,
                 ignore_stuff_in_offset=False,
                 small_instance_area=0,
                 small_instance_weight=1,
                 **kwargs):
        super(PavDataset, self).__init__(root, split, is_train, crop_size, mirror, min_scale, max_scale,
                                         scale_step_size, mean, std)
        self.num_classes = 4
        # for lb_idx in range(self.num_classe):
        # self.img_list = self._get_files('image', self.split)


        # self.root = './dataset/cls/'
        csv_path = os.path.join(self.root, 'train.csv')
        assert os.path.exists( csv_path ), 'Path does not exist: train.csv'
        print("csv_path: {}".format(csv_path))
        csv_fs = glob.glob(csv_path)[0]
        print("csv_fs: {}".format(csv_fs))

        csv_data = pd.read_csv(csv_fs)

        for img in csv_data['image_path']:
            fn1 = img.split('/')[0]
            fn2 = img.split('/')[1]
            fn2 = fn2.split('.pgm')[0]
            pattern = '%s_%s*.png' %(fn1, fn2)
            fpath = os.path.join(self.root, self.split, '*', pattern)
            fns = glob.glob(fpath)[0]
            assert os.path.exists( fns ), 'Path does not exist: {}'.format(fns)
            print("fns: {}".format(fns))
            self.img_list.append( fns )
            
        for label in csv_data['label']:
            self.ann_list.append( label )

        self.transform = build_transforms(self, is_train)





