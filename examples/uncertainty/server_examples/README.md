# Server Examples for Uncertainty Experiments

This example section aims to demonstrate how to use the DAL-Toolbox to investigate the impact of different methods on uncertainty metrics. For example, can we increase a model's ability for out-of-distribution detection when applying label-smoothing during training?

## Uncertainty Metrics

Next to the classic cross-entropy loss and classification accuracy other metrics exist that capture a models ability to express it's uncertainty. __Calibration-Error__ describes the model's ability to give reliable probabilities concerning its classification predictions. In contrast, the __AUROC__ when comparing predictions of in-domain and out-of-domain samples can be interpreted as the models ability to detect wether a given sample is part of the same domain of samples it trained on or not.

- Calibration

## Methods to improve Uncertainty Metrics
The general base model for each method described below is a ResNet-18. Building ontop of this, we provide the following methods to improve uncertainty metrics:
- Label Smoothing
- Mixup
- SNGP
- Ensembles
- MCDropOut

## Results
The following table shows results from experiments conducted on CIFAR10 with CIFAR100 and SVHN as out-of-distribution datasets.

__TBD. when experiments are finished...__


```
python uncertainty.py \
    model=resnet18 \
    dataset=CIFAR10 \
    ood_datasets=\['SVHN'\, 'CIFAR100'\] \
    output_dir=./output
```