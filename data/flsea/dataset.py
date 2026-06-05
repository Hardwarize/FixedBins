from depth_estimation.utils.data import (
    InputTargetDataset,
    IntPILToTensor,
    FloatPILToTensor,
    MutualRandomFactor,
    ReplaceInvalid,
    MutualRandomHorizontalFlip,
    # MutualRandomVerticalFlip,
)
from torchvision import transforms

import csv


def get_flsea_dataset(
    split="dataset_with_matched_features", train=False, shuffle=False, device="cpu"
):

    if train:
        csv_files = [
            "./red_sea_features_paths_npy.csv"
        ]
    
    else:
        csv_files = [
            "./canyons_features_paths_npy.csv"
        ]
    # filenames
    rgb_depth_priors_tuples = []
    for csv_file in csv_files:
        try:
            lines = csv.reader(open(csv_file).read().splitlines())
            rgb_depth_priors_tuples += [i.replace('/teamspace/studios/this_studio/', '/kaggle/input/datasets/guillermolvarez/optimized-flsea/') for i in lines]
        except FileNotFoundError:
            print(f"{csv_file} not found, skipping...")

    # transforms
    if train:
        input_transform = transforms.Compose(
            [
                IntPILToTensor(type="uint8", device=device),
                transforms.ColorJitter(brightness=0.1, hue=0.05),
            ]
        )
        target_transform = transforms.Compose(
            [
                FloatPILToTensor(device=device),
                ReplaceInvalid(value="max"),
            ]
        )
        all_transform = transforms.Compose([MutualRandomHorizontalFlip()])
        target_samples_transform = transforms.Compose(
            [
                MutualRandomFactor(factor_range=(0.8, 1.2)),
            ]
        )

    # if not train
    else:
        input_transform = transforms.Compose(
            [IntPILToTensor(type="uint8", device=device)]
        )
        target_transform = transforms.Compose(
            [
                FloatPILToTensor(device=device),
                ReplaceInvalid(value="max"),
            ]
        )
        all_transform = None
        target_samples_transform = None

    dataset = InputTargetDataset(
        rgb_depth_priors_tuples=rgb_depth_priors_tuples,
        input_transform=input_transform,
        target_transform=target_transform,
        all_transform=all_transform,
        target_samples_transform=target_samples_transform,
        max_priors=200,
        shuffle=shuffle,
    )

    return dataset
