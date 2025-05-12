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

## Evalutation on the COMPAS and Twitter dataset
1. COMPAS

```
python compas.py
```
2. Twitter

```
python twitter.py
```
## Running Jigsaw Toxicity Classification Experiments

First, you need to download the data (all_data.csv) from https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification and put it in the jigsaw_toxicity_bias folder (file is too big to upload to github).

1. Run the Full Pipeline
   This command runs the complete pipeline including data preprocessing, model training, and evaluation:

```
python jigsaw-pipeline.py --mode train --model_type both --epochs 3 --batch_size 32 --train_samples 25000 --test_samples 2000
```

## 2. Evaluate a Trained Model

To evaluate a previously trained model on the test set:

```
python jigsaw-pipeline.py --mode evaluate --model_type divdis --test_samples 2000
```

For evaluating the baseline model:

```
python jigsaw-pipeline.py --mode evaluate --model_type baseline --test_samples 2000
```

## 3. Generate Visualizations

To generate visualizations of model performance and bias metrics:

```
python jigsaw-pipeline.py --mode visualize --test_samples 2000
```

This will generate visualizations based on the previously saved results in bias_metrics_results.json.
