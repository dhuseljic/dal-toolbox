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

## References
