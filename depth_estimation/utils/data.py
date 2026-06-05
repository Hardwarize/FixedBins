import torch
from torchvision.transforms.functional import hflip, vflip
from PIL import Image
import time

import numpy as np

import pandas as pd
import random
from os.path import exists

from .depth_prior import get_depth_prior_from_features


class InputTargetDataset_old:
    """Parameters:
    - rgb_depth_priors_tuples: List of filepath tuples of form (rgb, depth, sparse priors)
    - input_transform: Transform to apply to the input RGB image, returns torch Tensor
    - target_transform: Transform to apply to the target depth image, returns torch Tensor
    - all_transform: Transform to apply to both input, target and mask image (and depth samples as well), returns list of torch Tensors
    - target_samples_transform: Transfrom to apply to both target and depth samples, returns list of torch Tensors
    - max_priors: max number of priors to subsample
    - shuffle: shuffle dataset"""

    def __init__(
        self,
        rgb_depth_priors_tuples,
        input_transform,
        target_transform,
        all_transform=None,
        target_samples_transform=None,
        max_priors=200,
        shuffle=False,
    ) -> None:

        # file paths
        self.path_tuples = rgb_depth_priors_tuples

        # transforms applied to input and target
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.all_transform = all_transform

        # random shuffle tuples
        if shuffle:
            random.shuffle(self.path_tuples)

        # depth_samples
        self.target_samples_transform = target_samples_transform
        self.max_priors = max_priors

        # checking dataset for missing files
        if not check_dataset(self.path_tuples):
            print("WARNING, dataset has missing files!")
            # exit(1)

        print(f"Dataset with {len(self)} tuples.")

    def __len__(self):
        return len(self.path_tuples)

    def __getitem__(self, idx):
        # Using perf_counter for high-resolution, precise benchmarking
        t_start = time.perf_counter()

        # --- 1. get filenames ---
        ## TIME_FEATURE:t0 = time.perf_counter()
        input_fn = self.path_tuples[idx][0]
        target_fn = self.path_tuples[idx][1]
        depth_samples_fn = self.path_tuples[idx][2]
        ## TIME_FEATURE:t1 = time.perf_counter()
        ## TIME_FEATURE:print(f"[Time] [Item: {idx}] 1. Get filenames: {t1 - t0:.6f}s")

        # --- 2. read imgs ---
        ## TIME_FEATURE:t2 = time.perf_counter()
        input_img = np.load(input_fn)#Image.open(input_fn)#.resize((640, 480))
        target_img = np.load(target_fn)#Image.open(target_fn)#.resize((320, 240), resample=Image.NEAREST)
        ## TIME_FEATURE:t3 = time.perf_counter()
        ## TIME_FEATURE:print(f"[Time] [Item: {idx}] 2. Read & resize images: {t3 - t2:.6f}s")

        # --- 3. apply input/target transforms ---
        ## TIME_FEATURE:t4 = time.perf_counter()
        input_img = self.input_transform(input_img)
        target_img, mask = self.target_transform(target_img)
        ## TIME_FEATURE:t5 = time.perf_counter()
        ## TIME_FEATURE:print(f"[Time] [Item: {idx}] 3. Input & Target transforms: {t5 - t4:.6f}s")

        # --- 4. check if depth map has at least one valid value ---
        ## TIME_FEATURE:t6 = time.perf_counter()
        if not mask.any():
            print(
                f"File {target_fn} has no valid depth values, trying other image as substitution ..."
            )
            random_idx = np.random.randint(0, len(self))
            return self[random_idx]  # recursion
        ## TIME_FEATURE:t7 = time.perf_counter()
        ## TIME_FEATURE:print(f"[Time] [Item: {idx}] 4. Mask validity check: {t7 - t6:.6f}s")

        # --- 5. read sparse depth priors ---
        ## TIME_FEATURE:t8 = time.perf_counter()
        depth_samples = read_features(depth_samples_fn, self.max_priors, device=target_img.device)
        ## TIME_FEATURE:t9 = time.perf_counter()
        ## TIME_FEATURE:print(f"[Time] [Item: {idx}] 5. Read sparse depth priors: {t9 - t8:.6f}s")

        # --- 6. check if features has at least one entry ---
        ## TIME_FEATURE:t10 = time.perf_counter()
        if depth_samples is None:
            print("Depth priors is None, trying other image as substitution ...")
            random_idx = np.random.randint(0, len(self))
            return self[random_idx]  # recursion
        ## TIME_FEATURE:t11 = time.perf_counter()
        ## TIME_FEATURE: print(f"[Time] [Item: {idx}] 6. Feature validity check: {t11 - t10:.6f}s")

        # --- 7. get dense parametrization from sparse priors ---
        ## TIME_FEATURE:t12 = time.perf_counter()
        parametrization = get_depth_prior_from_features(
            features=depth_samples.unsqueeze(0),  # add batch dimension
            height=240,
            width=320,
        ).squeeze(0)
        ## TIME_FEATURE:t13 = time.perf_counter()
        ## TIME_FEATURE:print(f"[Time] [Item: {idx}] 7. Dense parametrization: {t13 - t12:.6f}s")

        # --- 8. apply target + prior transform ---
        ## TIME_FEATURE:t14 = time.perf_counter()
        if self.target_samples_transform is not None:
            target_img, parametrization = self.target_samples_transform(
                [target_img, parametrization]
            )
        ## TIME_FEATURE:t15 = time.perf_counter()
        ## TIME_FEATURE:print(f"[Time] [Item: {idx}] 8. Target + prior transform: {t15 - t14:.6f}s")

        # --- 9. apply mutual transforms ---
        ## TIME_FEATURE:t16 = time.perf_counter()
        tensor_list = [input_img, target_img, mask, parametrization]
        if self.all_transform is not None:
            tensor_list = self.all_transform(tensor_list)
        ## TIME_FEATURE:t17 = time.perf_counter()
        ## TIME_FEATURE:print(f"[Time] [Item: {idx}] 9. Mutual transforms: {t17 - t16:.6f}s")
        
        ## TIME_FEATURE:print(f"[Time] [Item: {idx}] ---> TOTAL __getitem__ time: {t17 - t_start:.6f}s\n")

        return tensor_list


import time
import torch
import numpy as np
import random
from tqdm import tqdm # Para ver el progreso de carga en RAM

class InputTargetDataset:
    """Parameters:
    - rgb_depth_priors_tuples: List of filepath tuples of form (rgb, depth, sparse priors)
    - input_transform: Transform to apply to the input RGB image, returns torch Tensor
    - target_transform: Transform to apply to the target depth image, returns torch Tensor
    - all_transform: Transform to apply to both input, target and mask image...
    - target_samples_transform: Transfrom to apply to both target and depth samples...
    - max_priors: max number of priors to subsample
    - shuffle: shuffle dataset"""

    def __init__(
        self,
        rgb_depth_priors_tuples,
        input_transform,
        target_transform,
        all_transform=None,
        target_samples_transform=None,
        max_priors=200,
        shuffle=False,
    ) -> None:

        self.path_tuples = rgb_depth_priors_tuples
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.all_transform = all_transform
        self.target_samples_transform = target_samples_transform
        self.max_priors = max_priors

        if shuffle:
            random.shuffle(self.path_tuples)

        # checking dataset for missing files
        if not check_dataset(self.path_tuples):
            print("WARNING, dataset has missing files!")

        # --- CARGA DEL DATASET EN RAM ---
        self.cache = []
        print(f"Iniciando carga de {len(self.path_tuples)} archivos en RAM. Esto puede tardar...")

        for input_fn, target_fn, depth_samples_fn in self.path_tuples:
            try:
                # 1. Cargamos el target primero para validarlo
                target_img = np.load(target_fn)
                
                # Transformación anticipada: Si el mapa de profundidad es todo 0, lo descartamos.
                # Esto ahorra RAM y evita hacer chequeos recursivos en el __getitem__
                if not (target_img > 0.0).any():
                    continue

                # 2. Cargamos el input
                input_img = np.load(input_fn)

                # 3. Almacenamos en memoria conservando el formato puro de numpy
                self.cache.append({
                    'input_img': input_img,
                    'target_img': target_img,
                    'depth_samples_fn': depth_samples_fn # Dejamos el path para leerlo al vuelo
                })
            except Exception as e:
                print(f"Error cargando los archivos: {input_fn} o {target_fn}. Excepción: {e}")
                continue

        print(f"Carga completada. Muestras válidas retenidas en RAM: {len(self.cache)}")

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        t_start = time.perf_counter()

        # --- 1. Recuperar datos de la RAM ---
        data = self.cache[idx]
        
        # CRÍTICO: Usar .copy() para no modificar los arrays de la caché de RAM 
        # cuando apliquemos transformaciones in-place como ReplaceInvalid
        input_img = data['input_img'].copy()
        target_img = data['target_img'].copy()
        depth_samples_fn = data['depth_samples_fn']

        # --- 2. apply input/target transforms ---
        input_img = self.input_transform(input_img)
        target_img, mask = self.target_transform(target_img)

        # La comprobación "if not mask.any(): return self[random_idx]" ya no es necesaria
        # porque filtramos los inválidos en el __init__

        # --- 3. read sparse depth priors ---
        depth_samples = read_features(depth_samples_fn, self.max_priors, device=target_img.device)

        # --- 4. check if features has at least one entry ---
        if depth_samples is None:
            # Si da error en features, buscamos otro aleatorio
            random_idx = np.random.randint(0, len(self))
            return self[random_idx]

        # --- 5. get dense parametrization from sparse priors ---
        parametrization = get_depth_prior_from_features(
            features=depth_samples.unsqueeze(0),
            height=240,
            width=320,
        ).squeeze(0)

        # --- 6. apply target + prior transform ---
        if self.target_samples_transform is not None:
            target_img, parametrization = self.target_samples_transform(
                [target_img, parametrization]
            )

        # --- 7. apply mutual transforms ---
        tensor_list = [input_img, target_img, mask, parametrization]
        if self.all_transform is not None:
            tensor_list = self.all_transform(tensor_list)
        
        return tensor_list



class MutualRandomHorizontalFlip:
    """Randomly flips an input RGB imape and corresponding depth target horizontally with probability p.\\
    (Either both are transformed or neither of them)"""

    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, tensors):

        do_flip = torch.rand(1) < self.p

        # flip
        if do_flip:

            for i in range(len(tensors)):

                tensors[i] = hflip(tensors[i])

        return tensors


class MutualRandomVerticalFlip:
    """Randomly flips an input RGB imape and corresponding depth target vertically with probability p.\\
    (Either both are transformed or neither of them)"""

    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, tensors):
        do_flip = torch.rand(1) < self.p

        # flip
        if do_flip:
            for i in range(len(tensors)):

                tensors[i] = vflip(tensors[i])

        return tensors


class IntPILToTensor:
    """Converts an int PIL img to a torch float tensor in range [0,1]."""

    def __init__(self, type="uint8", custom_divider=None, device="cpu") -> None:

        self.device = device

        if type == "uint8":
            self.divider = 255
        elif type == "uint16":
            self.divider = 65535
        else:
            self.divider = 1

        if custom_divider is not None:
            self.divider = custom_divider  # ycb-video uses 10'000 as factor

    def __call__(self, img):

        # convert to np array
        img_np = np.array(img)

        # enforce dimension order: ch x H x W
        if img_np.ndim == 3:
            img_np = img_np.transpose((2, 0, 1))
        elif img_np.ndim == 2:
            img_np = img_np[np.newaxis, ...]

        # convert to tensor
        img_tensor = torch.from_numpy(img_np).to(self.device)

        # convert to float and divide by set divider
        img_tensor = img_tensor.float().div(self.divider)

        return img_tensor


class FloatPILToTensor:
    """Converts a float PIL img to a torch float tensor"""

    def __init__(self, device="cpu"):
        self.device = device

    def __call__(self, img):

        # convert to np array
        img_np = np.array(img)

        # enforce dimension order: channels x height x width
        if img_np.ndim == 2:
            img_np = img_np[np.newaxis, ...]

        # convert to tensor
        img_tensor = torch.from_numpy(img_np).to(self.device)

        return img_tensor


class MutualRandomFactor:
    """Multiply tensors by a random factor in given range."""

    def __init__(self, factor_range=(0.75, 1.25)) -> None:
        self.factor_range = factor_range

    def __call__(self, tensors):

        factor = (
            torch.rand(1).item() * (self.factor_range[1] - self.factor_range[0])
            + self.factor_range[0]
        )

        for i in range(len(tensors)):

            tensors[i][0, ...] *= factor

        return tensors


class ReplaceInvalid:
    """Replace invalid values (=0) of a tensor with a given vale."""

    def __init__(self, value=None):
        self.value = value

    def __call__(self, tensor):

        mask = get_mask(tensor)

        # if mask is empty, return None
        if not mask.any():
            print(
                "Mask is empty, meaning all depth values invalid. Returning unchanged."
            )

            return tensor, mask

        # change value of non valid pixels
        if self.value is not None:
            if self.value == "max":
                max = tensor[mask].max()
                tensor[~mask] = max
            elif self.value == "min":
                min = tensor[mask].min()
                tensor[~mask] = min
            else:
                tensor[~mask] = self.value

        return tensor, mask


def get_mask(depth):
    """Get mask depth > 0.0"""

    mask = depth.gt(0.0)

    return mask


def test_dataset():

    print("Testing InputTargetDataset class ...")

    # test specific imports
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    from data.example_dataset.dataset import get_example_dataset

    dataset = get_example_dataset()

    # dataloader
    dataloader = DataLoader(dataset, batch_size=2)

    for batch_id, data in enumerate(dataloader):

        rgb_imgs = data[0]
        d_imgs = data[1]
        masks = data[2]
        parametrizations = data[3]

        for i in range(rgb_imgs.size(0)):

            rgb_img = rgb_imgs[i, ...]
            d_img = d_imgs[i, ...]
            mask = masks[i, ...]
            nn_parametrization = parametrizations[i, 0, ...].unsqueeze(0)
            prob_parametrization = parametrizations[i, 1, ...].unsqueeze(0)

            print(f"d range: [{d_img.min()}, {d_img.max()}]")

            plt.figure(f"rgb img {i}")
            plt.imshow(rgb_img.permute(1, 2, 0))
            plt.figure(f"d img {i}")
            plt.imshow(d_img.permute(1, 2, 0))
            plt.figure(f"mask {i}")
            plt.imshow(mask.permute(1, 2, 0))
            plt.figure(f"parametrization, NN {i}")
            plt.imshow(nn_parametrization.permute(1, 2, 0))
            plt.figure(f"parametrization, Probability {i}")
            plt.imshow(prob_parametrization.permute(1, 2, 0))

        plt.show()

        break  # only check first batch

    print("Testing DataSet class done.")


# run as "python -m depth_estimation.utils.data" from repo root
if __name__ == "__main__":
    test_dataset()
