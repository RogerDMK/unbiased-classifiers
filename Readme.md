# Final Project

## Overview
The repo includes models, explainability methods, utility functions, and analyses of both the COMPAS, Twitter hate speech and Jigsaw datasets. A conda environment to run the code can be constructed from requirements.txt

## Repository Structure
- `models.py` - Defines machine learning models with tunable hyperparameters, including seed configurations for reproducible results.
- `explainer.py` - Contains explainability methods to interpret model predictions and outputs.
- `utils.py` - Utility functions used across different components of the project.
- `compas.py` - Performs analysis on the COMPAS dataset, focusing on fairness and biases in risk assessments.
- `twitter.py` - Conducts hate speech detection and analysis on Twitter data.
- `jigsaw_utils.py` - Contains Jigsaw utils
- `jigsaw_visualizer.py` - Contains Jigsaw explainability code
- `jigsaw-pipeline` - Contains pipeline to run jigsaw analysis and models

For COMPAS and Twitter hatespeech models hyperparameters, including seeds, can be set within the files. The LIME and Shapely results for COMPAS data are in 3 heads, 5 heads, 10 heads, for each ensemble size respectively.
