# Deep Uncertainty Modeling and Active Learning

Framework for uncertainty-based neural networks and active learning.

## Setup
```
conda env create -f environment.yml
```

### Development
In order to develop it is best to use :
```
pip install -e .
```

## Running Uncertainty Experiments
For detailed parameter choices and reproducing results take a look at the slurm folder.
```
python uncertainty.py \
    model=resnet18 \
    dataset=CIFAR10 \
    ood_datasets=\['SVHN'\] \
    output_dir=./output
```

## Running AL Experiments
For detailed instructions take a look at the slurm folder.
```
TODO
```

## Examples Notebooks
Examples on how to use the respective models can be found here:
- [Deterministic](experiments/notebooks/2D-Examples/deterministic.ipynb)
- [Ensemble](experiments/notebooks/2D-Examples/ensemble.ipynb)
- [MC-Dropout](experiments/notebooks/2D-Examples/mc-dropout.ipynb)
- [SNGP](experiments/notebooks/2D-Examples/sngp.ipynb)


## References
[1] J. Liu, Z. Lin, S. Padhy, D. Tran, T. Bedrax Weiss, and B. Lakshminarayanan, “Simple and principled uncertainty estimation with deterministic deep learning via distance awareness,” Advances in Neural Information Processing Systems, vol. 33, pp. 7498–7512, 2020.  
[2] Y. Gal and Z. Ghahramani, “Dropout as a bayesian approximation: Representing model uncertainty in deep learning,” in International Conference on Machine Learning, pp. 1050–1059. 2016.  
[3] W. H. Beluch, T. Genewein, A. Nurnberger, and J. M. Kohler, “The Power of Ensembles for Active Learning in Image Classification,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9368–9377. 2018.  