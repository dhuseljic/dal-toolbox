# Active Learning Scripts
This folder contains the active learning scripts for different models with
different aleatoric strategies.  The models we include are i) Standard Resnet18,
ii) Resnet18 using label smoothing, and iii) Resnet18 using mixup.  Introducing
these modifications improves the aleatoric uncertainty estimation of our models
which allows us, to obtain a selection with better aleatoric uncertainty.
The aleatoric strategies we consider are i) least confident, ii) margin, and iii) entropy.
