# HDML-pytorch

Unofficial Pytorch implementation for the paper "[Hardness-Aware Deep Metric Learning" (CVPR 2019 Oral)](https://arxiv.org/abs/1903.05503)".

# Code Setup
- **Requirements**:
	- Pytorch
	- Torchvision
	- Tensorboard
	- Pillow
	- Pandas
  
- Download the [Cars196](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) dataset and place the `cars_ims` folder inside the `datasets` folder.
- Download GoogLeNet ImageNet weights: https://download.pytorch.org/models/googlenet-1378be20.pth
- Run: `` python main.py --experiment NAME | NAME.log ``

## Results
- Experiments were performed on the Cars196 dataset using Triplet Loss.
- Training logs can be found in the `logs` folder and online at [this tensorboard link](https://tensorboard.dev/experiment/7Ly8NMIqRxS39ehM6BqQUw/#scalars).
- **Recall Resuts**: Baseline vs HDML:
	- Recall@1:  60 / 65
	- Recall@2: 72 / 77
	- Recall@4: 82 / 85
	- Recall@8: 89 / 92


## TODO

- Benchmarking on other datasets.
- Adding support for N-Pair loss.

Official TensorFlow implementation can be found at: https://github.com/neka-nat/pytorch-hdml

