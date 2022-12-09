# Experiments
Summary of some done experiments.

## General

### CIFAR10: ResNet18
|               |   acc1 |   nll |   tce |   mce |   auroc |
|:--------------|-------:|------:|------:|------:|--------:|
| deterministic | 95.300 | 0.193 | 0.029 | 0.025 |   0.936 |
| dropout       | 94.420 | 0.171 | 0.006 | 0.017 |   0.932 |
| sngp          | 95.094 | 0.161 | 0.006 | 0.018 |   0.961 |

### CIFAR10: Wide-ResNet-28-10
|               |   acc1 |   nll |   tce |   auroc |
|:--------------|-------:|------:|------:|--------:|
| deterministic | 96.446 | 0.128 | 0.016 |   0.932 |
| dropout       | 96.400 | 0.127 | 0.015 |   0.958 |
| ensemble      | 96.886 | 0.098 | 0.007 |   0.976 |
| sngp          | 96.292 | 0.123 | 0.005 |   0.960 |