# Evaluation Scripts for Aleatoric Strategies

This folder contains the evaluation scripts for aleatoric strategies.  Here, we
employ a standard ResNet18 model which uses the selection of other AL strategies
(i.e., another model + uncertainty-based score).  This is to ensure that the
improvement in AL performance is due to the selection only, and not from
modifications of the standard model itself (e.g., mixup).