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

|Model          | ACC	     |   NLL	   |   Brier	| TCE	    | ACE	        | AUPR (SVHN)	| AUROC (SVHN)	| AUPR (CIFAR100)	|AUROC (CIFAR100)   |
|---------------|------------|-------------|------------|-----------|---------------|---------------|---------------|-------------------|-------------------|
|Deterministic	|   0.9518	 |   0.1875	   | 0.0773  	| 0.0281  	| 0.0340  	    | 0.9642  	    | 0.9319  	    |0.8665  	        |0.8797             |
|Ensemble	    |   0.9607	 |   0.1309	   | 0.0605  	| 0.0087  	| 0.0170  	    | 0.9779  	    | 0.9582  	    |0.8970  	        |0.9086             |
|Labelsmoothing	|   0.9526	 |   0.2021	   | 0.0752  	| 0.0424  	| 0.0355  	    | 0.9360  	    | 0.8488  	    |0.8411  	        |0.8131             |
|MC-Dropout	    |   0.9445	 |   0.1698	   | 0.0824  	| 0.0089  	| 0.0153  	    | 0.9557  	    | 0.9202  	    |0.8812  	        |0.8978             |
|Mixup	        |   0.9557	 |   0.2081	   | 0.0746  	| 0.0659  	| 0.0292  	    | 0.9620  	    | 0.9151  	    |0.8515  	        |0.8358             |
|SNGP           |	0.9458	 |   0.1769	   | 0.0819  	| 0.0057  	| 0.0161  	    | 0.9779  	    | 0.9550  	    |0.8723  	        |0.8826             |