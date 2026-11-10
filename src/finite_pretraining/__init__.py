"""Finite pretraining of aligned tied linear autoencoders."""

from .closure import Fit, fit_surrogate
from .pretraining import generate_dataset, spectral_encoder, train_lae
from .finetuning import Dynamics, predict, simulate, references

__all__ = [
    "Fit",
    "fit_surrogate",
    "generate_dataset",
    "spectral_encoder",
    "train_lae",
    "Dynamics",
    "predict",
    "simulate",
    "references",
]
