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

### CIFAR10

### SNGP Ablation: CIFAR10

|                  |   test_acc1 |   test_loss |   test_nll |   test_tce |   test_mce |   test_SVHN_entropy_auroc |   test_SVHN_conf_auroc |   test_SVHN_entropy_aupr |   test_SVHN_conf_aupr |
|:-----------------|------------:|------------:|-----------:|-----------:|-----------:|--------------------------:|-----------------------:|-------------------------:|----------------------:|
| default          |      94.456 |       0.220 |      0.220 |      0.074 |      0.046 |                     0.864 |                  0.863 |                    0.855 |                 0.849 |
| num_inducing4096 |      94.897 |       0.211 |      0.211 |      0.069 |      0.044 |                     0.868 |                  0.867 |                    0.859 |                 0.854 |

## References

TODO

