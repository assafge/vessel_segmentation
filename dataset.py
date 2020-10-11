
import os.path
import numpy as np
from glob import glob
import torch
from typing import List, Dict
import cv2
import matplotlib.pyplot as plt
import random


def list_images(folder_path: str, pattern='') -> List[str]:
    """return a list of png files"""
    files_list = glob(os.path.join(folder_path, '*' + pattern))
    # files_list.sort(key=file_name_order)
    return files_list



def base_collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption)."""
    images, labels = zip(*data)
    # Merge arrays (from tuple of 3D to 4D).
    images = np.stack(images, 0)
    labels = np.stack(labels, 0)
    im_tensor = torch.as_tensor(data=np.ascontiguousarray(images), dtype=torch.float32)
    lbl_tensor = torch.as_tensor(data=np.ascontiguousarray(labels), dtype=torch.long)
    # lbl_tensor = torch.as_tensor(data=np.ascontiguousarray(labels), dtype=torch.float32)
    # if im_tensor.ndim < lbl_tensor.ndim:
    #     im_tensor = im_tensor.unsqueeze(1)
    return im_tensor, lbl_tensor

class VesselSegmentation(torch.utils.data.Dataset):
    def __init__(self, out_dir: str, patch_size: int, seed=42, shuffle=True, train_split=0.8):
        self.data_map = {}
        self.data: List = []  # list of patches
        self.train_split = train_split
        self.train_idx: np.ndarray = None
        self.test_idx: np.ndarray = None
        self.train_split = train_split
        self.shuffle = shuffle
        self.seed = seed
        self.patch_size = patch_size
        self.datasets = {
            'chase_db1': {
                'img_dir': 'data/CHASEDB1',
                'gt_dir': 'data/CHASEDB1',
                'img_suffix': '.jpg',
                'gt_suffix': '_2ndHO.png'},
            'drive': {
                'img_dir': 'data/DRIVE/training/images',
                'gt_dir': 'data/DRIVE/training/1st_manual',
                'img_suffix': 'training.tif',
                'gt_suffix': 'manual1.gif'},
            'hrf': {
                'img_dir': 'data/HRF/images',
                'gt_dir': 'data/HRF/manual1',
                'img_suffix': '.jpg',
                'gt_suffix': '.tif'},
            'iostar': {
                'img_dir': 'data/IOSTAR/image',
                'gt_dir': 'data/IOSTAR/GT',
                'img_suffix': '.jpg',
                'gt_suffix': '_GT.tif'
            }
        }
        self.prepare_data(out_dir)

    def prepare_data(self, out_dir):
        last_idx = 0
        for dataset_name, prm in self.datasets.items():
            for img_path in list_images(prm['img_dir'], prm['img_suffix']):
                ref_path = img_path.replace(prm['img_suffix'], prm['gt_suffix'])
                ref_path = ref_path.replace(prm['img_dir'], prm['gt_dir'])

                if os.path.exists(ref_path):
                    self.data.append((img_path, ref_path))
                else:
                    print()
            print('added {} images from {} dataset'.format(len(self.data) - last_idx, dataset_name))
            last_idx = len(self.data)
        assert len(self.data) > 0, 'dataset is empty'
        if self.shuffle:
            # random.seed = self.seed
            random.shuffle(self.data)
        train_size = int(len(self.data) * self.train_split)
        self.train_idx = np.arange(train_size)
        self.test_idx = np.arange(train_size, len(self.data))
        with open(os.path.join(out_dir, 'test_images.txt'), 'w') as f:
            for idx in self.test_idx:
                f.write(self.data[idx][0] + os.linesep)
        print('wrote test images list to test_images.txt')


    def random_sample_patch(self, im, lbl, do_transpose=True, debug=False):
        y0, y1, x0, x1 = 0, im.shape[0], 0, im.shape[1]
        sy = random.randint(y0, y1 - self.patch_size)
        sx = random.randint(x0, x1 - self.patch_size)

        im_p = im[sy:sy + self.patch_size, sx:sx + self.patch_size]
        lbl_p = lbl[sy:sy + self.patch_size, sx:sx + self.patch_size]

        if debug:
            r2g = im_p[:, :, 0] / im_p[:, :, 1]
            r2b = im_p[:, :, 0] / im_p[:, :, 2]
            r2g_l = lbl_p[:, :, 0] / lbl_p[:, :, 1]
            r2b_l = lbl_p[:, :, 0] / lbl_p[:, :, 2]
            R2G = np.mean(cv2.absdiff(r2g, r2g_l))
            R2B = np.mean(cv2.absdiff(r2b, r2b_l))
            # print(R2G, R2B)
            if R2G > 1 or R2B > 1:
                print(sy, sx)
                f, a = plt.subplots(2, 2)
                a[0, 0].imshow(im_p.astype(np.uint8))
                a[0, 1].imshow(im.astype(np.uint8))
                a[1, 0].imshow(lbl)
                a[1, 1].imshow(im)
                f.suptitle('R2G={:.3f}  | R2B={:.3f}'.format(R2G, R2B))
                plt.show()

        if do_transpose:
            im_p = np.transpose(im_p, (2, 0, 1))
        return im_p.astype(np.float32) / 255, lbl_p / 255

    def __getitem__(self, item):

        img_path, ref_path = self.data[item]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        sample, label = self.random_sample_patch(img, ref)

        return sample, label

    def __len__(self):
        return len(self.data)

    def get_data_loaders(self, batch_size):
        train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.Subset(
            self, indices=self.train_idx), batch_size=batch_size, collate_fn=base_collate_fn,
            # pin_memory=True, num_workers=1, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.Subset(
            self, indices=self.test_idx), batch_size=batch_size, collate_fn=base_collate_fn,
            # pin_memory=False, shuffle=False,
            )
        return train_loader, test_loader
