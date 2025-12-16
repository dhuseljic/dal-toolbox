# Examples on how to use the DAL-Toolbox

This folder contains a minimal example on how to use the different tools provided by our DAL-Toolbox. The primary toolsets contained in the Toolbox are

- __active_learning__: Contains all kinds of active learning strategies ready to be used like random, BADGE, BAIT...
- __datasets__: Contains different kinds of datasets and the ActiveLearningDataModule, a key component for active learning.
- __metrics__: Contains metrics of interest to track throughout active learning, like accuracy, calibration, AUC...
- __models__: Contains various kinds of model types that can work with different backbones like deterministic, laplace, ensemble...

Recommended imports and usage is shown in the [active_learning.py"-file](./active_learning.py)". In addition, we provide a minimal [slurm script](./slurm_example.sh) that is ready to be run and extended once you replace the placeholders with your individual settings.
