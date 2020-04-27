#!/usr/bin/env bash

#office
##a2w
#python train_image.py --test_interval 300 --epoch 9000 --use_seed True --torch_seed 893561550778924240 --torch_cuda_seed 1756436598284239 --left_weight 1 --right_weight 1 --mdd_weight 0.05 --entropic_weight 0 --log_name a2w.txt --s_dset_path ./data/office/amazon_list.txt --t_dset_path ./data/office/webcam_list.txt


#image_clef
#i2c
python train_image.py  --epoch 5000 --mdd_weight 0.025 --entropic_weight 0.5 --output_dir img-clef --log_name i2c.txt  --s_dset_path ./data/image-clef/i_list.txt --t_dset_path ./data/image-clef/c_list.txt --dset image-clef


#s2m
python train_svhnmnist.py --weight 0 --left_weight 1 --right_weight 1  --batch_size 64 --seed 40 --epochs 40 --mdd_weight 0.01 --entropic_weight 0


#u2m
python train_uspsmnist.py --task MNIST2USPS --epoch 40 --mdd_weight 0.005 --entropic_weight 0.1