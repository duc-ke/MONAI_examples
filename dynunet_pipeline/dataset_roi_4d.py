# -*- coding: utf-8 -*-
import csv
import torch
import torch.distributed as dist
from monai.data import (
    CacheDataset,
    DataLoader,
    partition_dataset,
)
# from transforms_roi import get_train_transform, get_validation_transform, get_test_transform
from transforms_roi_4d import get_train_transform, get_validation_transform, get_test_transform

# train_aids = [
#     '00_0',
#     '01_d',
#     '02_d',
#     '03_d',
#     '04_d',
#     '05_d',
#     '06_d',
#     '07_d',
#     '08_d',
#     '09_d',
#     '10_d',
#     '11_dn',
#     '12_dn',
#     '13_dn',
#     '14_dn',
#     '15_dn',
#     '16_dn',
#     '17_dn',
#     '18_dn',
#     '19_dn',
#     '20_dn'
# ]

# 몇개만 테스트용으로 사용하자.
train_aids = [
    '00_0',
    '01_d',
    '02_d',
    '03_d',
]

val_aids = [
    "00_0",
    # "01_d",
]

def get_train_data(id_file, data_dir, 
                   image_file_pattern, label_file_pattern,
                   mask_file_pattern):
    f = open(id_file)
    rows = csv.DictReader(f)
    data = []
    for row in rows:
        id = row["id"]
        for aid in train_aids:
            data.append(
                {
                    "image": image_file_pattern.format(dir=data_dir, id=id, aid=aid),
                    "label": label_file_pattern.format(dir=data_dir, id=id, aid=aid),
                    # "brain": brain_file_pattern.format(dir=data_dir, id=id, aid=aid),
                    "mask": mask_file_pattern.format(dir=data_dir, id=id, aid=aid),
                }
            )
            print(image_file_pattern, data_dir, id, aid)
    return data

def get_val_data(id_file, data_dir, 
                 image_file_pattern, label_file_pattern,
                 mask_file_pattern):
    f = open(id_file)
    rows = csv.DictReader(f)
    data = []
    for row in rows:
        id = row["id"]
        for aid in val_aids:
            data.append(
                {
                    "image": image_file_pattern.format(dir=data_dir, id=id, aid=aid),
                    "label": label_file_pattern.format(dir=data_dir, id=id, aid=aid),
                    # "brain": brain_file_pattern.format(dir=data_dir, id=id, aid=aid),
                    "mask": mask_file_pattern.format(dir=data_dir, id=id, aid=aid),
                }
            )
            print(image_file_pattern, data_dir, id, aid)
    return data

def get_test_data(id_file, data_dir, 
                 image_file_pattern, mask_file_pattern):
    f = open(id_file)
    rows = csv.DictReader(f)
    data = []
    for row in rows:
        id = row["id"]
        for aid in val_aids:
            data.append(
                {
                    "image": image_file_pattern.format(dir=data_dir, id=id, aid=aid),
                    # "label": label_file_pattern.format(dir=data_dir, id=id, aid=aid),
                    # "brain": brain_file_pattern.format(dir=data_dir, id=id, aid=aid),
                    "mask": mask_file_pattern.format(dir=data_dir, id=id, aid=aid),
                }
            )
            print(image_file_pattern, data_dir, id, aid)
    return data

def get_train_loader(
    data_dir,
    id_file,
    image_file_pattern,
    label_file_pattern,
    mask_file_pattern,
    batch_size,
    patch_size,
    num_samples,
    num_workers=8,
    cache_rate=1.0,
    multi_gpu_flag = False
):
    data = get_train_data(
        id_file, data_dir, 
        image_file_pattern, label_file_pattern,
        mask_file_pattern
    )

    transform = get_train_transform(patch_size, num_samples)
    # transform = get_train_transform("train", "01", (1, 1, 8))
    
    if multi_gpu_flag:
        data = partition_dataset(
            data=data,
            shuffle=True,
            num_partitions=dist.get_world_size(),
            even_divisible=True,
        )[dist.get_rank()]

    dataset = CacheDataset(
        data=data,
        transform=transform,
        num_workers=8,
        cache_rate=cache_rate,
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    return data_loader


def get_val_loader(
    data_dir,
    id_file,
    image_file_pattern,
    label_file_pattern,
    mask_file_pattern,
    batch_size,
    num_workers=1,
    multi_gpu_flag = False
):
    data = get_val_data(
        id_file, data_dir, 
        image_file_pattern, label_file_pattern,
        mask_file_pattern
    )

    transform = get_validation_transform()

    if multi_gpu_flag:
        data = partition_dataset(
            data=data,
            shuffle=False,
            num_partitions=dist.get_world_size(),
            even_divisible=False,
        )[dist.get_rank()]

    dataset = CacheDataset(data=data, transform=transform, num_workers=4)

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return data_loader


def get_test_loader(
    data_dir,
    id_file,
    image_file_pattern,
    mask_file_pattern,
    batch_size,
    num_workers=1,
    multi_gpu_flag = False
):
    data = get_test_data(
        id_file, data_dir, 
        image_file_pattern,
        mask_file_pattern
    )

    transform = get_test_transform()

    # if torch.cuda.is_available() and dist.get_world_size() > 1:
    if multi_gpu_flag:
        data = partition_dataset(
            data=data,
            shuffle=False,
            num_partitions=dist.get_world_size(),
            even_divisible=False,
        )[dist.get_rank()]

    dataset = CacheDataset(data=data, transform=transform, num_workers=4)

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return data_loader
