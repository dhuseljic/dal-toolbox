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
|               |   test_acc1 |   test_prec |   test_loss |   test_nll |   test_tce |   test_mce |   test_SVHN_entropy_auroc |   test_SVHN_entropy_aupr |
|:--------------|------------:|------------:|------------:|-----------:|-----------:|-----------:|--------------------------:|-------------------------:|
| deterministic |     95.1833 |    0.951844 |    0.200975 |   0.200975 |  0.0485284 |  0.0220805 |                  0.946721 |                 0.935976 |
| dropout       |     94.66   |    0.946542 |    0.160994 |   0.160994 |  0.0123387 |  0.0136697 |                  0.92705  |                 0.89594  |
| sngp          |     95.19   |    0.951941 |    0.183397 |   0.183397 |  0.0236729 |  0.0170452 |                  0.950784 |                 0.950499 |

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
Adapt Kernel Scale (10 reps):
|               |   test_acc1 |   test_prec |   test_loss |   test_nll |   test_tce |   test_mce |   test_SVHN_entropy_auroc |   test_SVHN_entropy_aupr |
|:--------------|------------:|------------:|------------:|-----------:|-----------:|-----------:|--------------------------:|-------------------------:|
| deterministic |      95.251 |    0.952563 |    0.197761 |   0.197761 |  0.0464164 |  0.0214368 |                  0.899446 |                 0.887948 |
| scale5        |      94.707 |    0.947173 |    0.31674  |   0.31674  |  0.155289  |  0.0519013 |                  0.851029 |                 0.806068 |
| scale10       |      94.946 |    0.949525 |    0.189047 |   0.189047 |  0.0185986 |  0.0164396 |                  0.951141 |                 0.950725 |
| scale20       |      95.084 |    0.950877 |    0.174552 |   0.174552 |  0.0377965 |  0.0194097 |                  0.939107 |                 0.92472  |
| scale50       |      95.099 |    0.951107 |    0.187391 |   0.187391 |  0.0436503 |  0.0206812 |                  0.936355 |                 0.921185 |
| scale100      |      95.093 |    0.951046 |    0.190403 |   0.190403 |  0.0441826 |  0.0217058 |                  0.932819 |                 0.91823  |
| scale200      |      95.262 |    0.952735 |    0.187278 |   0.187278 |  0.0437041 |  0.0208737 |                  0.937771 |                 0.924532 |

### SNGP Ablation: CIFAR10

|                      |   test_acc1 |   test_loss |   test_nll |   test_tce |   test_mce |   test_SVHN_entropy_auroc |   test_SVHN_entropy_aupr |
|:---------------------|------------:|------------:|-----------:|-----------:|-----------:|--------------------------:|-------------------------:|
| default              |      94.477 |    0.22009  |   0.22009  |  0.0738159 |  0.0460703 |                  0.863937 |                 0.85443  |
| kernel_scale=10      |      95.202 |    0.19427  |   0.19427  |  0.0716748 |  0.0448753 |                  0.876433 |                 0.866152 |
| num_inducing=4096    |      94.908 |    0.210421 |   0.210421 |  0.069614  |  0.044008  |                  0.868013 |                 0.859462 |
| scale_features=False |      86.161 |    0.451635 |   0.451635 |  0.0669977 |  0.0473571 |                  0.799947 |                 0.746378 |

## References

TODO

