#!/bin/bash
CUDA_VISIBLE_DEVICES='0' python3.10 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 0 --logdir exp/cifar10 --exclude_class 0
