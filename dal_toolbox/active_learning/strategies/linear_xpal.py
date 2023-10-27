# Linear Implementation of https://arxiv.org/pdf/2006.01732.pdf.
# Code partially taken from https://github.com/dakot/probal/blob/master/src/query_strategies/expected_probabilistic_active_learning.py
import logging
import time

import lightning as L
import numpy as np
import torch
from sklearn.utils import check_array
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .query import Query


class LinearXPAL(Query):
    """XPAL
    The expected probabilistic active learning (xPAL) strategy.
    Parameters
    ----------
    n_classes: int
        Number of classes.
    S: array-like, shape (n_samples, n_samples)
        Similarity matrix defining the similarities between all pairs of available samples, e.g., S[i,j] describes
        the similarity between the samples x_i and x_j.
        Default similarity matrix is the unit matrix.
    alpha_c: float | array-like, shape (n_classes)
        Prior probabilities for the Dirichlet distribution of the candidate samples.
        Default is 1 for all classes.
    alpha_x: float | array-like, shape (n_classes)
        Prior probabilities for the Dirichlet distribution of the samples in the evaluation set.
        Default is 1 for all classes.
    subset_size: int
        How much of the unlabeled dataset is taken into account.
    random_seed: numeric | np.random.RandomState
        Random seed for annotator selection.
    Attributes
    ----------
    n_classes_: int
        Number of classes.
    S_: array-like, shape (n_samples, n_samples)
        Similarity matrix defining the similarities between all pairs of available samples, e.g., S[i,j] describes
        the similarity between the samples x_i and x_j.
        Default similarity matrix is the unit matrix.
    alpha_c_: float | array-like, shape (n_classes)
        Prior probabilities for the Dirichlet distribution of the candidate samples.
        Default is 1 for all classes.
    alpha_x_: float | array-like, shape (n_classes)
        Prior probabilities for the Dirichlet distribution of the samples in the evaluation set.
        Default is 1 for all classes.
    """

    def __init__(self, alpha = 0.0, beta=0.0, subset_size=None, random_seed=None):
        super().__init__(random_seed)
        self.subset_size = subset_size
        self.alpha = alpha
        self.beta = beta

    def query(self, *, model, al_datamodule, acq_size, return_gains=False, **kwargs):
        trainer = L.Trainer(
            max_epochs=90,
            enable_checkpointing=False,
            accelerator="cpu",
            default_root_dir=None,
            enable_progress_bar=False,
            check_val_every_n_epoch=100,
            enable_model_summary=False,
        )

        logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(subset_size=self.subset_size)
        labeled_dataloader, labeled_indices = al_datamodule.labeled_dataloader()
        labels = torch.unique(al_datamodule.train_dataset.dataset.labels).tolist()
        num_samples = len(unlabeled_indices) + len(labeled_indices)
        f_L_predictions = trainer.predict(model, [unlabeled_dataloader, labeled_dataloader])
        f_L_unlabeled_predictions = torch.softmax(torch.cat([pred[0] for pred in f_L_predictions[0]]) + self.beta, dim=1)
        f_L_unlabeled_prediction_classes = f_L_unlabeled_predictions.argmax(-1)
        f_L_labeled_predictions = torch.softmax(torch.cat([pred[0] for pred in f_L_predictions[1]]) + self.beta, dim=1)
        f_L_labeled_prediction_classes = f_L_labeled_predictions.argmax(-1)

        gains = []

        fits = []
        # Calculate gain for each unlabeled sample
        for x, x_pred in tqdm(zip(unlabeled_indices, f_L_unlabeled_predictions), total=len(unlabeled_indices), desc="Canidate", leave=False, colour="blue"):
            gain = 0.0

            L_plus_indices = labeled_indices + [x]
            L_plus_data = al_datamodule.train_dataset[L_plus_indices]
            for candidate_label in tqdm(labels, desc="Canidate Label", leave=False, colour="green"):

                dataset = CustomDataset(L_plus_data, candidate_label)

                dloader = DataLoader(dataset=dataset, batch_size=len(L_plus_indices) // 10 + 1, shuffle=True)
                trainer = L.Trainer(max_epochs=90, enable_checkpointing=False, accelerator="cpu", default_root_dir=None,
                                    enable_progress_bar=False, check_val_every_n_epoch=100, enable_model_summary=False)

                model.reset_states()
                tic = time.perf_counter()

                trainer.fit(model, train_dataloaders=dloader)  # Create f_L_plus

                toc = time.perf_counter()
                fits.append(toc - tic)

                f_L_plus_predictions = trainer.predict(model, [unlabeled_dataloader, labeled_dataloader])
                f_L_plus_unlabeled_predictions = torch.softmax(torch.cat([pred[0] for pred in f_L_plus_predictions[0]]) + self.alpha, dim=1)
                f_L_plus_unlabeled_prediction_classes = f_L_plus_unlabeled_predictions.argmax(-1)
                f_L_plus_labeled_predictions = torch.softmax(torch.cat([pred[0] for pred in f_L_plus_predictions[1]]) + self.alpha, dim=1)
                f_L_plus_labeled_prediction_classes = f_L_plus_labeled_predictions.argmax(-1)

                sum_ = 0.0
                for label in labels:
                    unlabeled_gain = f_L_plus_unlabeled_predictions[:, label] * ((label != f_L_plus_unlabeled_prediction_classes).float() - (label != f_L_unlabeled_prediction_classes).float())
                    labeled_gain = f_L_plus_labeled_predictions[:, label] * ((label != f_L_plus_labeled_prediction_classes).float() - (label != f_L_labeled_prediction_classes).float())
                    sum_ += torch.sum(unlabeled_gain) + torch.sum(labeled_gain)

                sum_ /= num_samples
                gain += x_pred[candidate_label] * sum_
            gain = -gain
            gains.append(gain)

        top_gains, indices = torch.topk(torch.Tensor(gains), acq_size)

        actual_indices = [int(unlabeled_indices[i]) for i in indices]

        logging.getLogger("lightning.pytorch").setLevel(logging.INFO)
        print(f"Took on average {np.mean(fits)} seconds to fit linear model." )

        if return_gains:
            return actual_indices, top_gains
        return actual_indices


class CustomDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.data[1][-1] = label  # Overwrite true label

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx]
