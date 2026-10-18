# Joint Angle Inspector — Model Trainer

[Русский](README.ru.md) | **English**

A Python research codebase for training and comparing machine-learning models on human exercise and pose data. Inputs are sequences of 11 joint angles or 44 expanded features.

## Included in this repository

- Includes recurrent and Transformer autoencoders, variational models, analysis tools, experiment tracking, and report generation.
- Core modules include `model_zoo.py`, `data_pipeline.py`, `experiment_manager.py`, and `run_experiment.py`.
- `requirements.txt` lists the Python dependencies; `readme.md` contains the longer system manual.

## Getting started

Create a Python environment, install `requirements.txt`, then consult `readme.md` for experiment and training commands.

## Notes

Model scores depend on the dataset split, preprocessing, and experiment settings. The README documents the codebase and does not imply clinical validation.
