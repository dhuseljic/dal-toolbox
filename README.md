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

### SNGP 
Default Config:
```
name : resnet18_sngp
spectral_norm:
  use_spectral_norm: True
  coeff: 6
  n_power_iterations: 1
gp:
  kernel_scale: ...
  num_inducing: 1024
  normalize_input: False
  scale_random_features: False
  cov_momentum: -1
  ridge_penalty: 1
  mean_field_factor: 0.393 # = pi / 8
optimizer:
  lr: 0.08
  weight_decay: 3e-4
  momentum: .9
```
Adapt Kernel Scale:
|                      |   test_acc1 |   test_loss |   test_nll |   test_tce |   test_mce |   test_SVHN_entropy_auroc |   test_SVHN_entropy_aupr |
|:---------------------|------------:|------------:|-----------:|-----------:|-----------:|--------------------------:|-------------------------:|


### SNGP Ablation: CIFAR10

|                      |   test_acc1 |   test_loss |   test_nll |   test_tce |   test_mce |   test_SVHN_entropy_auroc |   test_SVHN_entropy_aupr |
|:---------------------|------------:|------------:|-----------:|-----------:|-----------:|--------------------------:|-------------------------:|
| default              |      94.477 |    0.22009  |   0.22009  |  0.0738159 |  0.0460703 |                  0.863937 |                 0.85443  |
| kernel_scale=10      |      95.202 |    0.19427  |   0.19427  |  0.0716748 |  0.0448753 |                  0.876433 |                 0.866152 |
| num_inducing=4096    |      94.908 |    0.210421 |   0.210421 |  0.069614  |  0.044008  |                  0.868013 |                 0.859462 |
| scale_features=False |      86.161 |    0.451635 |   0.451635 |  0.0669977 |  0.0473571 |                  0.799947 |                 0.746378 |

|          |   test_acc1 |   test_prec |   test_loss |   test_nll |   test_tce |   test_mce |   test_SVHN_entropy_auroc |   test_SVHN_conf_auroc |   test_SVHN_entropy_aupr |   test_SVHN_conf_aupr |
|:---------|------------:|------------:|------------:|-----------:|-----------:|-----------:|--------------------------:|-----------------------:|-------------------------:|----------------------:|
| scale5   |     94.6833 |    0.946959 |    0.323586 |   0.323586 |  0.162511  |  0.0543539 |                  0.843845 |               0.877071 |                 0.794999 |              0.868031 |
| scale10  |     95.19   |    0.951941 |    0.183397 |   0.183397 |  0.0236729 |  0.0170452 |                  0.950784 |               0.948381 |                 0.950499 |              0.947192 |
| scale20  |     95.0767 |    0.950844 |    0.173778 |   0.173778 |  0.0396597 |  0.019671  |                  0.948606 |               0.942049 |                 0.93784  |              0.92453  |
| scale50  |     95.429  |  nan        |    0.195884 |   0.195884 |  0.0461696 |  0.0218912 |                  0.894061 |               0.891966 |                 0.884355 |              0.877633 |
| scale100 |     95.1133 |    0.951254 |    0.189364 |   0.189364 |  0.0455852 |  0.0217968 |                  0.938106 |               0.933445 |                 0.922456 |              0.909393 |
| scale200 |     95.3733 |    0.953805 |    0.184243 |   0.184243 |  0.0415868 |  0.0201096 |                  0.946996 |               0.943015 |                 0.934876 |              0.924746 |

## References

TODO

