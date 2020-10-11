# Vessels Segmentation

Vessels Segmentation is a PyTorch based library for training and evaluating segmentation networks of fundus images.

## Datasets
* CHASEDB1 
* DRIVE
* HRF
* IOSTAR
* STARE

The datasets are free to use and can be easily downloaded. 

## Train

"train.py" has two positional arguments, execution of the form:
$ python train.py <architecture> <encoder>
```bash

$ $ python train.py -h
usage: train.py [-h] [-g GPU_INDEX]
                {UNet-non-fine-tune,Unet,Linknet,FPN,PSPNet,PAN,DeepLabV3,DeepLabV3+}
                {resnet18,resnet34,resnet50,resnet101,resnet152,resnext50_32x4d,resnext101_32x4d,resnext101_32x8d,resnext101_32x16d,resnext101_32x32d,resnext101_32x48d,dpn68,dpn68b,dpn92,dpn98,dpn107,dpn131,vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19,vgg19_bn,senet154,se_resnet50,se_resnet101,se_resnet152,se_resnext50_32x4d,se_resnext101_32x4d,densenet121,densenet169,densenet201,densenet161,inceptionresnetv2,inceptionv4,efficientnet-b0,efficientnet-b1,efficientnet-b2,efficientnet-b3,efficientnet-b4,efficientnet-b5,efficientnet-b6,efficientnet-b7,mobilenet_v2,xception,timm-efficientnet-b0,timm-efficientnet-b1,timm-efficientnet-b2,timm-efficientnet-b3,timm-efficientnet-b4,timm-efficientnet-b5,timm-efficientnet-b6,timm-efficientnet-b7,timm-efficientnet-b8,timm-efficientnet-l2,16,32,64}


positional arguments:
  {UNet-non-fine-tune,Unet,Linknet,FPN,PSPNet,PAN,DeepLabV3,DeepLabV3+}
                        segmentation head architecture
  {resnet18,resnet34,resnet50,resnet101,resnet152,resnext50_32x4d,resnext101_32x4d,resnext101_32x8d,resnext101_32x16d,resnext101_32x32d,resnext101_32x48d,dpn68,dpn68b,dpn92,dpn98,dpn107,dpn131,vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19,vgg19_bn,senet154,se_resnet50,se_resnet101,se_resnet152,se_resnext50_32x4d,se_resnext101_32x4d,densenet121,densenet169,densenet201,densenet161,inceptionresnetv2,inceptionv4,efficientnet-b0,efficientnet-b1,efficientnet-b2,efficientnet-b3,efficientnet-b4,efficientnet-b5,efficientnet-b6,efficientnet-b7,mobilenet_v2,xception,timm-efficientnet-b0,timm-efficientnet-b1,timm-efficientnet-b2,timm-efficientnet-b3,timm-efficientnet-b4,timm-efficientnet-b5,timm-efficientnet-b6,timm-efficientnet-b7,timm-efficientnet-b8,timm-efficientnet-l2,16,32,64}
                        backbone encoder (trained on imagenet)

optional arguments:
  -h, --help            show this help message and exit
  -g GPU_INDEX, --gpu_index GPU_INDEX
                        index of gpu (if exist, torch indexing) use -1 for cpu (default: 0)

```
```
$ python train.py FPN se_resnext50_32x4d
using device:  TITAN V
writing results to directory: trained_models/FPN_se_resnext50_32x4d_11Oct_1039
added 28 images from chase_db1 dataset
added 20 images from drive dataset
added 30 images from hrf dataset
added 30 images from iostar dataset
added 20 images from stare dataset
wrote test images list to test_images.txt
epoch 0/300 train loss 0.7881 test loss 0.7228
epoch 1/300 train loss 0.6387 test loss 0.5598
epoch 2/300 train loss 0.5157 test loss 0.5055
epoch 3/300 train loss 0.4877 test loss 0.4634
...
```
## Inference

```
$ python infer.py -h
usage: infer.py [-h] [-g GPU_INDEX] model_path image_path ref_path

positional arguments:
  model_path
  image_path
  ref_path

optional arguments:
  -h, --help            show this help message and exit
  -g GPU_INDEX, --gpu_index GPU_INDEX
                        index of gpu if exist (torch indexing), -1 for cpu (default: -1)
```
```
$ python infer.py ~/data/vessel_segmentation/trained_models/FPN_se_resnext50_32x4d_11Oct1039 ~/data/vessel_segmentation/data/STARE/stare-images/im0077.ppm ~/data/vessel_segmentation/data/STARE/labels-ah/im0077.ah.ppm -g 0
using device:  TITAN V
loading checkpoint /home/assaf/data/vessel_segmentation/trained_models/FPN_se_resnext50_32x4d_11Oct1039/best_checkpoint.pth
```

![Alt text](plot_example.png?raw=true "example of inference plot")
