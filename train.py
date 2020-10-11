import torch
import numpy as np
import argparse
from datetime import datetime
from shutil import rmtree
import os
import segmentation_models_pytorch as smp
import albumentations as albu
from vessel_dataset import VesselSegmentationDataset
from torch.utils.tensorboard import SummaryWriter
from models.Unet_milesial import UNet
import matplotlib.pyplot as plt


data_root = '/home/assaf/data/vessel_segmentation/'


def save_checkpoint(root, model, epoch, optimizer, better):
    if better:
        fpath = os.path.join(root, 'best_checkpoint.pth')
    else:
        fpath = os.path.join(root, 'last_checkpoint.pth')
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, fpath)


def train(device, model, criterion, optimizer, train_loader, test_loader, epochs, root):
    model.zero_grad()
    best_loss = 2 ** 16
    running = True
    epoch = 0
    train_writer = SummaryWriter(os.path.join(root, 'train'))
    test_writer = SummaryWriter(os.path.join(root, 'test'))

    while epoch <= epochs and running:
        model.train()
        train_loss = 0
        for x, y in train_loader:
            inputs, labels = x.to(device), y.to(device)
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            train_loss += loss.item() / len(train_loader)
            optimizer.step()
        train_writer.add_scalar(tag='loss', scalar_value=train_loss, global_step=epoch)

        model.eval()
        test_loss = 0
        pixelwize_rank = 0
        for x, y in test_loader:
            inputs, labels = x.to(device), y.to(device)
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            test_loss += loss.item() / len(test_loader)
            optimizer.step()
            if outputs.shape == labels.shape:
                pixelwize_rank += int(
                    torch.sum((outputs > 0.5) == (labels> 0.5))) / labels.data.nelement() * 100 / len(test_loader)
            else:
                pixelwize_rank += int(
                    torch.sum((outputs.argmax(dim=1) > 0.5) == (labels > 0.5))) / labels.data.nelement() * 100 / len(test_loader)
        test_writer.add_scalar(tag='loss', scalar_value=test_loss, global_step=epoch)
        test_writer.add_scalar(tag='pixel wise accuracy', scalar_value=pixelwize_rank, global_step=epoch)

        print(f'epoch {epoch}/{epochs} train loss {train_loss:.4} test loss {test_loss:.4}')
        if test_loss < best_loss:
            best_loss = test_loss
        save_checkpoint(root, model, epoch, optimizer, test_loss == best_loss)

        epoch += 1

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_long_tensor(x, **kargs):
    return (np.squeeze(x, axis=2) / 255).astype(np.longlong)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')/255


def get_preprocessing(long_target):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    if long_target:
        _transform = [albu.Lambda(image=to_tensor, mask=to_long_tensor),]
    else:
        _transform = [albu.Lambda(image=to_tensor, mask=to_tensor), ]

    return albu.Compose(_transform)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu_index', help='index of gpu (if exist, torch indexing) use -1 for cpu', type=int, default=0)
    parser.add_argument('architecture', help='segmentation head architecture',
                        choices=['UNet-non-fine-tune', 'Unet', 'Linknet', 'FPN', 'PSPNet', 'PAN', 'DeepLabV3', 'DeepLabV3+'])
    parser.add_argument('encoder', help='backbone encoder (trained on imagenet)',
                                          choices=smp.encoders.get_encoder_names() + ['16', '32', '64'])
    args = parser.parse_args()

    if int(args.gpu_index) >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.gpu_index))
        print('using device: ', torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print('using cpu')
    model_dir = os.path.join(data_root, 'trained_models', 
                             '_'.join([args.architecture, args.encoder,  datetime.now().strftime("%d%b%H%M")]))
    if os.path.isdir(model_dir):
        print("root directory is already exist - will delete the previous and create new")
        rmtree(model_dir)
    os.makedirs(model_dir)
    print('writing results to directory: %s' % model_dir)

    if args.architecture == 'FPN':
        long_target = False

        model = smp.FPN(
            encoder_name=args.encoder,
            encoder_weights='imagenet',
            activation='sigmoid',
            classes=1,
        )

        loss = smp.utils.losses.DiceLoss()
    elif args.architecture == 'Unet':
        long_target = False

        model = smp.Unet(
            encoder_name=args.encoder,
            encoder_weights='imagenet',
            activation='sigmoid',
            classes=1,
        )
        loss = smp.utils.losses.DiceLoss()
    elif args.architecture == 'UNet':
        long_target = True
        model = UNet(n_channels=3, n_classes=2, scale_channels=int(args.encoder))
        loss = torch.nn.CrossEntropyLoss()

    # preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, 'imagenet')

    dataset = VesselSegmentationDataset(
        out_dir=model_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(long_target),
    )
    train_loader, test_loader = dataset.get_data_loaders(batch_size=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1E-4)

    model.to(device)
    loss.to(device)
    train(model=model,
          optimizer=optimizer,
          criterion=loss,
          device=device, train_loader=train_loader, test_loader=test_loader, epochs=300, root=model_dir)
