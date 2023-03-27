# Real-time Monte Carlo Denoising with Weight Sharing Kernel Prediction Network
The official pytorch implementation of our paper "Real-time Monte Carlo Denoising with Weight Sharing Kernel Prediction Network" in EGSR2021.
This respository contains the following organization:

## Dataset
Please download the BMFR data from https://github.com/maZZZu/bmfr/issues/2, and put it into dataset/folder. Then preprocess the data by running data_preprocess.py.
The DataLoader class is defined in dataset.py. Currently we load all the dataset into the RAM directly, and it can be modified with regular batch loading manner.

## Network
The network is defined in net.py. 

During training, we use a Sum-kernel-based method to implementation the identical computation of kernel construction and kernel applying described in our paper for convenience. Please refer to the file train.py.

During test, we firstly apply re-parameterization to our RepVGG Block, and then use a CUDA implementation of kernel construction and kernel applying for runtime speed. Please run CUSTOM_WSKP_auto/setup.py to install the CUDA extention library and refer to the file test.py.

## Citation
```
@inproceedings{fan2021real,
  title={Real-time Monte Carlo Denoising with Weight Sharing Kernel Prediction Network},
  author={Fan, Hangming and Wang, Rui and Huo, Yuchi and Bao, Hujun},
  booktitle={Computer Graphics Forum},
  volume={40},
  number={4},
  pages={15--27},
  year={2021},
  organization={Wiley Online Library}
}
```