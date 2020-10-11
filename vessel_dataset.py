import torch
import numpy as np
import cv2
from glob import glob
import os
import random


data_root = '/home/assaf/data/vessel_segmentation/data'


class VesselSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, out_dir: str, train_split=0.8, augmentation=None, preprocessing=None):
        self.train_split = train_split
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.train_idx: np.ndarray = None
        self.test_idx: np.ndarray = None
        self.images_fps = []
        self.masks_fps = []
        # self.patch_size = patch_size
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
                'gt_suffix': 'manual1.png'},
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
            },
            'stare': {
                'img_dir': 'data/STARE/stare-images',
                'gt_dir': 'data/STARE/labels-ah',
                'img_suffix': '.ppm',
                'gt_suffix': '.ah.ppm'
            }
        }

        last_idx = 0
        for dataset_name, prm in self.datasets.items():
            for img_path in glob(os.path.join(prm['img_dir'], '*' + prm['img_suffix'])):
                ref_path = img_path.replace(prm['img_suffix'], prm['gt_suffix'])
                ref_path = ref_path.replace(prm['img_dir'], prm['gt_dir'])

                if os.path.exists(ref_path):
                    self.images_fps.append(img_path)
                    self.masks_fps.append(ref_path)
            print('added {} images from {} dataset'.format(len(self.images_fps) - last_idx, dataset_name))
            last_idx = len(self.images_fps)
        assert len(self.images_fps) > 0, 'dataset is empty'
        train_size = int(len(self) * train_split)
        ind = np.arange(len(self))
        random.seed(42)
        random.shuffle(ind)
        self.train_idx = ind[:train_size]
        self.test_idx = ind[train_size:]
        with open(os.path.join(out_dir, 'test_images.txt'), 'w') as f:
            for idx in self.test_idx:
                f.write(f'{self.images_fps[idx]}, {self.masks_fps[idx]}{os.linesep}')
        print('wrote test images list to test_images.txt')

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=2).astype('float32')

        # extract certain classes from mask (e.g. cars)
        # masks = [(mask == v) for v in self.class_values]
        # mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        else:
            image = image.transpose(2,1,0).astype('float32')
            mask = mask.transpose(2,1,0).astype('float32')

        return image, mask

    def __len__(self):
        return len(self.images_fps)

    def get_data_loaders(self, batch_size):
        train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.Subset(
            self, indices=self.train_idx), batch_size=batch_size,
            num_workers=4, shuffle=True
            # pin_memory=True, num_workers=1, s
        )
        test_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.Subset(
            self, indices=self.test_idx), batch_size=batch_size,
            num_workers=2,
            # pin_memory=False, shuffle=False,
            )
        return train_loader, test_loader

