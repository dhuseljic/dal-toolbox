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

### Resnet18-SNGP: CIFAR10

|                  |   test_acc1 |   test_loss |   test_nll |   test_tce |   test_mce |   test_SVHN_entropy_auroc |   test_SVHN_conf_auroc |   test_SVHN_entropy_aupr |   test_SVHN_conf_aupr |
|:-----------------|------------:|------------:|-----------:|-----------:|-----------:|--------------------------:|-----------------------:|-------------------------:|----------------------:|
| default          |     94.4556 |    0.219927 |   0.219927 |  0.0741822 |  0.0464066 |                  0.863924 |               0.863019 |                 0.854582 |              0.848949 |
| num_inducing4096 |     94.8967 |    0.211319 |   0.211319 |  0.0692637 |  0.043826  |                  0.868102 |               0.867269 |                 0.859247 |              0.854196 |
## References

TODO

