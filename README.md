# ATM (Adversarial Tight Match)
Maximum Density Divergence for Domain Adaptation published on IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)

Authors: Jingjing Li, Erpeng Chen, Zhengming Ding, Lei Zhu, Ke Lu and Heng Tao Shen



# ATM implemneted in PyTorch

## Prerequisites
- PyTorch >= 1.0.0 (with suitable CUDA and CuDNN version)
- torchvision >= 0.2.2
- Python3
- Numpy
- argparse
- PIL

## Training
Please use the following commands for different tasks. 

You can find more detailed commands samples in the *train.sh* file
```
SVHN->MNIST
python train_svhnmnist.py --mdd_weight 0.01 --epochs 50

USPS->MNIST
python train_uspsmnist.py --mdd_weight 0.01 --epochs 50 --task USPS2MNIST

MNIST->USPS
python train_uspsmnist.py --mdd_weight 0.01 --epochs 50 --task MNIST2USPS
```
```
Office-31

python train_image.py  --net ResNet50 --dset office --test_interval 500 --s_dset_path ../data/office/amazon_list.txt --t_dset_path ../data/office/webcam_list.txt
```
```
Office-Home

python train_image.py  --net ResNet50 --dset office-home --test_interval 2000 --s_dset_path ../data/office-home/Art.txt --t_dset_path ../data/office-home/Clipart.txt
```

```
Image-clef

python train_image.py  --net ResNet50 --dset image-clef --test_interval 500 --s_dset_path ../data/image-clef/b_list.txt --t_dset_path ../data/image-clef/i_list.txt
```

The adversarial learning part is inspired by CDAN.
