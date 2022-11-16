# Uncertainty-Evaluation

Evaluation framework for uncertainty-based neural networks.


## Setup
```
conda env create -f environment.yml
```

## Running
For detailed instructions take a look at the slurm folder.
```
MODEL=sngp
OUTPUT_DIR=./output/

python main.py model=$MODEL output_dir=$OUTPUT_DIR
```

## 2D-Examples
Examples on how to use the respective models can be found here:
- [SNGP](notebooks/2D-Examples/ensemble.ipynb)
- [MC-Dropout](notebooks/2D-Examples/ensemble.ipynb)
- [Ensemble](notebooks/2D-Examples/ensemble.ipynb)

## References
[1] J. Liu, Z. Lin, S. Padhy, D. Tran, T. Bedrax Weiss, and B. Lakshminarayanan, “Simple and principled uncertainty estimation with deterministic deep learning via distance awareness,” Advances in Neural Information Processing Systems, vol. 33, pp. 7498–7512, 2020.  
[2] Y. Gal and Z. Ghahramani, “Dropout as a bayesian approximation: Representing model uncertainty in deep learning,” in International Conference on Machine Learning, pp. 1050–1059. 2016.  
[3] W. H. Beluch, T. Genewein, A. Nurnberger, and J. M. Kohler, “The Power of Ensembles for Active Learning in Image Classification,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9368–9377. 2018.  