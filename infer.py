import torch
import argparse
import os
import segmentation_models_pytorch as smp
from models.Unet_milesial import UNet
import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_path')
    parser.add_argument('image_path')
    parser.add_argument('ref_path')
    parser.add_argument('-g', '--gpu_index', help='index of gpu if exist (torch indexing), -1 for cpu', type=int, default=-1)
    args = parser.parse_args()

    if int(args.gpu_index) >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.gpu_index))
        print('using device: ', torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print('using cpu')

    cp_path = os.path.join(args.model_path, 'best_checkpoint.pth')
    print('loading checkpoint', cp_path)
    checkpoint = torch.load(cp_path, map_location=device)
    epoch = checkpoint['epoch']

    if 'FPN' in args.model_path:
        model = smp.FPN(
            encoder_name='_'.join(os.path.basename(args.model_path).split('_')[1:-1]),
            encoder_weights='imagenet',
            activation='sigmoid',
            classes=1,
        )
        loss = smp.utils.losses.DiceLoss()
    elif 'Unet' in args.model_path:
        long_target = False

        model = smp.Unet(
            encoder_name=os.path.basename(args.model_path).split('_')[1],
            encoder_weights='imagenet',
            activation='sigmoid',
            classes=1,
        )
        loss = smp.utils.losses.DiceLoss()
    elif 'UNet' in args.model_path:
        long_target = True
        model = UNet(n_channels=3, n_classes=2, scale_channels=64)
        loss = torch.nn.CrossEntropyLoss()

    model.load_state_dict(checkpoint['model_state_dict'])


    model.to(device)

    im = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    print(im.shape)
    factor = 32
    if im.shape[0] % factor != 0 or im.shape[1] % factor != 0:
        h = ((im.shape[0] // factor) + 1) * factor
        w = ((im.shape[1] // factor) + 1) * factor
        top = (h - im.shape[0]) // 2
        bottom = h - im.shape[0] - top
        left = (w - im.shape[1]) // 2
        right = w - im.shape[1] - left
        pad = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)
    else:
        pad = im
    in_img = pad.transpose(2, 1, 0).astype(np.float32) / 255
    inputs = torch.from_numpy(in_img).float().to(device)
    inputs = inputs.unsqueeze(0)
    out = model(inputs)
    res = out.cpu().detach().numpy().squeeze(0)
    if 'UNet' not in args.model_path:
        res = res.squeeze(0).transpose(1,0)
    else:
        res = res.argmax(axis=0).transpose(1,0)
    if args.ref_path is not None:
        ref = cv2.imread(args.ref_path, cv2.IMREAD_GRAYSCALE)
        if pad is im:
            pad_ref = cv2.copyMakeBorder(ref, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)
        else:
            pad_ref = ref
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
        ax1.imshow(im), ax1.set_title('Original Image')
        net_name = os.path.basename(args.model_path)
        if len(net_name) > 20:
            net_name = '_'.join(os.path.basename(args.model_path).split('_')[:-1])
        ax2.imshow(res), ax2.set_title(net_name + ' result')
        ax3.imshow(pad_ref), ax3.set_title('GT')
        plt.show()
