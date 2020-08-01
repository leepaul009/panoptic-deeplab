# ------------------------------------------------------------------------------
# Builds dataloader.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import logging

import torch
import numpy as np

from .datasets import Cityscapes, CityscapesPanoptic, COCOPanoptic, PavDataset
from . import samplers
from segmentation.utils.comm import get_world_size
from segmentation.utils.env import seed_all_rng

import torchvision, os

def build_dataset_from_cfg(config, is_train=True):
    """Builds dataset from configuration file.
    Args:
        config: the configuration file.
        is_train: Bool, training or testing, it automatically handles data augmentation.

    Returns:
        A torch Dataset.
    """
    dataset_map = {
        'cityscapes': Cityscapes,
        'cityscapes_panoptic': CityscapesPanoptic,
        'coco_panoptic': COCOPanoptic,
        'pav_dataset' : PavDataset,
    }

    dataset_cfg = {
        'cityscapes': dict(
            root=config.DATASET.ROOT,
            split=config.DATASET.TRAIN_SPLIT if is_train else config.DATASET.TEST_SPLIT,
            is_train=is_train,
            crop_size=config.DATASET.CROP_SIZE if is_train else config.TEST.CROP_SIZE,
            mirror=config.DATASET.MIRROR,
            min_scale=config.DATASET.MIN_SCALE,
            max_scale=config.DATASET.MAX_SCALE,
            scale_step_size=config.DATASET.SCALE_STEP_SIZE,
            mean=config.DATASET.MEAN,
            std=config.DATASET.STD
        ),
        'cityscapes_panoptic': dict(
            root=config.DATASET.ROOT,
            split=config.DATASET.TRAIN_SPLIT if is_train else config.DATASET.TEST_SPLIT,
            is_train=is_train,
            crop_size=config.DATASET.CROP_SIZE if is_train else config.TEST.CROP_SIZE,
            mirror=config.DATASET.MIRROR,
            min_scale=config.DATASET.MIN_SCALE,
            max_scale=config.DATASET.MAX_SCALE,
            scale_step_size=config.DATASET.SCALE_STEP_SIZE,
            mean=config.DATASET.MEAN,
            std=config.DATASET.STD,
            semantic_only=config.DATASET.SEMANTIC_ONLY,
            ignore_stuff_in_offset=config.DATASET.IGNORE_STUFF_IN_OFFSET,
            small_instance_area=config.DATASET.SMALL_INSTANCE_AREA,
            small_instance_weight=config.DATASET.SMALL_INSTANCE_WEIGHT
        ),
        'coco_panoptic': dict(
            root=config.DATASET.ROOT,
            split=config.DATASET.TRAIN_SPLIT if is_train else config.DATASET.TEST_SPLIT,
            min_resize_value=config.DATASET.MIN_RESIZE_VALUE,
            max_resize_value=config.DATASET.MAX_RESIZE_VALUE,
            resize_factor=config.DATASET.RESIZE_FACTOR,
            is_train=is_train,
            crop_size=config.DATASET.CROP_SIZE if is_train else config.TEST.CROP_SIZE,
            mirror=config.DATASET.MIRROR,
            min_scale=config.DATASET.MIN_SCALE,
            max_scale=config.DATASET.MAX_SCALE,
            scale_step_size=config.DATASET.SCALE_STEP_SIZE,
            mean=config.DATASET.MEAN,
            std=config.DATASET.STD,
            semantic_only=config.DATASET.SEMANTIC_ONLY,
            ignore_stuff_in_offset=config.DATASET.IGNORE_STUFF_IN_OFFSET,
            small_instance_area=config.DATASET.SMALL_INSTANCE_AREA,
            small_instance_weight=config.DATASET.SMALL_INSTANCE_WEIGHT
        ),
        'pav_dataset': dict(
            root=config.DATASET.ROOT,
            split=config.DATASET.TRAIN_SPLIT if is_train else config.DATASET.TEST_SPLIT,
            is_train=is_train,
            crop_size=config.DATASET.CROP_SIZE if is_train else config.TEST.CROP_SIZE,
            mirror=config.DATASET.MIRROR,
            min_scale=config.DATASET.MIN_SCALE,
            max_scale=config.DATASET.MAX_SCALE,
            scale_step_size=config.DATASET.SCALE_STEP_SIZE,
            mean=config.DATASET.MEAN,
            std=config.DATASET.STD,
            semantic_only=config.DATASET.SEMANTIC_ONLY,
            ignore_stuff_in_offset=config.DATASET.IGNORE_STUFF_IN_OFFSET,
            small_instance_area=config.DATASET.SMALL_INSTANCE_AREA,
            small_instance_weight=config.DATASET.SMALL_INSTANCE_WEIGHT
        ),
    }
    print("to create dataset: {}".format(config.DATASET.DATASET))
    dataset = dataset_map[config.DATASET.DATASET](
        **dataset_cfg[config.DATASET.DATASET]
    )
    print("create dataset")
    return dataset


def build_dataset_from_cfg1(config, is_train=True):

    train_dataset_path = os.path.join(config.DATASET.ROOT, 'train')
    trans = torchvision.transforms.Compose([
            # torchvision.transforms.RandomResizedCrop(input_image_size),
            torchvision.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.ColorJitter(brightness=0.4,
            #                       contrast=0.4,
            #                       saturation=0.4,
            #                       hue=0),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    dataset = torchvision.datasets.ImageFolder(train_dataset_path, trans)
    return dataset


def build_train_loader_from_cfg(config):
    """Builds dataloader from configuration file.
    Args:
        config: the configuration file.

    Returns:
        A torch Dataloader.
    """
    num_workers = get_world_size()
    images_per_batch = config.TRAIN.IMS_PER_BATCH
    assert (
            images_per_batch % num_workers == 0
    ), "TRAIN.IMS_PER_BATCH ({}) must be divisible by the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    assert (
            images_per_batch >= num_workers
    ), "TRAIN.IMS_PER_BATCH ({}) must be larger than the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    images_per_worker = images_per_batch // num_workers

    # dataset = build_dataset_from_cfg(config, is_train=True)
    print("To build dataset from cfg 1")
    dataset = build_dataset_from_cfg1(config, is_train=True)
    print("built dataset with len: {}".format( len(dataset) ))

    sampler_name = config.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == "TrainingSampler":
        sampler = samplers.TrainingSampler(len(dataset), shuffle=config.DATALOADER.TRAIN_SHUFFLE)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_worker, drop_last=True
    )
    # drop_last so the batch always have the same size
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=config.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        worker_init_fn=worker_init_reset_seed,
    )
    print("to return data_loader")
    return data_loader


def build_test_loader_from_cfg(config):
    """Builds dataloader from configuration file.
    Args:
        config: the configuration file.

    Returns:
        A torch Dataloader.
    """
    dataset = build_dataset_from_cfg(config, is_train=False)

    sampler = samplers.InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=config.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
    )

    return data_loader


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)
