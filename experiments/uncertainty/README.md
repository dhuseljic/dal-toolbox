## Running Uncertainty Experiments
For detailed parameter choices and reproducing results take a look at the slurm folder.
```
python uncertainty.py \
    model=resnet18 \
    dataset=CIFAR10 \
    ood_datasets=\['SVHN'\] \
    output_dir=./output
```