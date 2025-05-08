#!/bin/bash
EXCLUDE=1
CUDA_VISIBLE_DEVICES='2' python3.10 -u data_only.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 0 --logdir exp_{$EXCLUDE}/cifar10 --exclude_class {$EXCLUDE}
CUDA_VISIBLE_DEVICES='2' python3.10 -u data_only.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 1 --logdir exp_{$EXCLUDE}/cifar10 --exclude_class {$EXCLUDE}
CUDA_VISIBLE_DEVICES='2' python3.10 -u data_only.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 2 --logdir exp_{$EXCLUDE}/cifar10 --exclude_class {$EXCLUDE}
CUDA_VISIBLE_DEVICES='2' python3.10 -u data_only.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 3 --logdir exp_{$EXCLUDE}/cifar10 --exclude_class {$EXCLUDE}
CUDA_VISIBLE_DEVICES='2' python3.10 -u data_only.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 4 --logdir exp_{$EXCLUDE}/cifar10 --exclude_class {$EXCLUDE}
CUDA_VISIBLE_DEVICES='2' python3.10 -u data_only.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 5 --logdir exp_{$EXCLUDE}/cifar10 --exclude_class {$EXCLUDE}
CUDA_VISIBLE_DEVICES='2' python3.10 -u data_only.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 6 --logdir exp_{$EXCLUDE}/cifar10 --exclude_class {$EXCLUDE}
CUDA_VISIBLE_DEVICES='2' python3.10 -u data_only.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 7 --logdir exp_{$EXCLUDE}/cifar10 --exclude_class {$EXCLUDE}
CUDA_VISIBLE_DEVICES='2' python3.10 -u data_only.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 8 --logdir exp_{$EXCLUDE}/cifar10 --exclude_class {$EXCLUDE}
CUDA_VISIBLE_DEVICES='2' python3.10 -u data_only.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 9 --logdir exp_{$EXCLUDE}/cifar10 --exclude_class {$EXCLUDE}
CUDA_VISIBLE_DEVICES='2' python3.10 -u data_only.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 10 --logdir exp_{$EXCLUDE}/cifar10 --exclude_class {$EXCLUDE}
CUDA_VISIBLE_DEVICES='2' python3.10 -u data_only.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 11 --logdir exp_{$EXCLUDE}/cifar10 --exclude_class {$EXCLUDE}
CUDA_VISIBLE_DEVICES='2' python3.10 -u data_only.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 12 --logdir exp_{$EXCLUDE}/cifar10 --exclude_class {$EXCLUDE}
CUDA_VISIBLE_DEVICES='2' python3.10 -u data_only.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 13 --logdir exp_{$EXCLUDE}/cifar10 --exclude_class {$EXCLUDE}
CUDA_VISIBLE_DEVICES='2' python3.10 -u data_only.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 14 --logdir exp_{$EXCLUDE}/cifar10 --exclude_class {$EXCLUDE}
CUDA_VISIBLE_DEVICES='2' python3.10 -u data_only.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 15 --logdir exp_{$EXCLUDE}/cifar10 --exclude_class {$EXCLUDE}
