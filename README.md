# Uncertainty-Evaluation

Evaluation framework for uncertainty-based neural networks.


## Setup
```
conda env create -f environment.yml
```

## Running
```
MODEL=sngp
OUTPUT_DIR=./output/

python main.py model=$MODEL output_dir=$OUTPUT_DIR
```


## Results

### CIFAR10: ResNet18
|                     |   test_acc1 |   test_loss |   test_nll |   test_tce |   test_mce |   test_SVHN_entropy_auroc |   test_SVHN_entropy_aupr |
|:--------------------|------------:|------------:|-----------:|-----------:|-----------:|--------------------------:|-------------------------:|
| resnet18            |      95.275 |    0.197757 |   0.197757 |  0.0695829 |  0.0447987 |                  0.880982 |                 0.869589 |
| resnet18_sngp       |      94.477 |    0.22009  |   0.22009  |  0.0738159 |  0.0460703 |                  0.863937 |                 0.85443  |

### CIFAR10: Wide-ResNet-28-10
|                     |   test_acc1 |   test_loss |   test_nll |   test_tce |   test_mce |   test_SVHN_entropy_auroc |   test_SVHN_entropy_aupr |
|:--------------------|------------:|------------:|-----------:|-----------:|-----------:|--------------------------:|-------------------------:|
| wideresnet2810      |      96.05  |    0.159374 |   0.159374 |  0.041665  |  0.021761  |                  0.892527 |                 0.884355 |
| wideresnet2810_mc   |      96.28  |    0.120258 |   0.120258 |  0.012486  |  0.014515  |                  0.918307 |                 0.905463 |
| wideresnet2810_sngp |      95.74  |    0.205887 |   0.205887 |  0.058210  |  0.025145	 |                  0.848497 |                 0.857889 |
| wideresnet2810_ens  |      96.61  |    0.114379 |   0.114379 |  0.018241  |  0.015642  |                  0.919876 |                 0.908626 |

## Ablations

### SNGP Ablation: CIFAR10

|                      |   test_acc1 |   test_loss |   test_nll |   test_tce |   test_mce |   test_SVHN_entropy_auroc |   test_SVHN_entropy_aupr |
|:---------------------|------------:|------------:|-----------:|-----------:|-----------:|--------------------------:|-------------------------:|
| default              |      94.477 |    0.22009  |   0.22009  |  0.0738159 |  0.0460703 |                  0.863937 |                 0.85443  |
| kernel_scale=10      |      95.202 |    0.19427  |   0.19427  |  0.0716748 |  0.0448753 |                  0.876433 |                 0.866152 |
| num_inducing=4096    |      94.908 |    0.210421 |   0.210421 |  0.069614  |  0.044008  |                  0.868013 |                 0.859462 |
| scale_features=False |      86.161 |    0.451635 |   0.451635 |  0.0669977 |  0.0473571 |                  0.799947 |                 0.746378 |


## References

TODO

