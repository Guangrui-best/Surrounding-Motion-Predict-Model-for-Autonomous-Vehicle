# Surrounding Motion Predict Model for Autonomous Vehicle



> Wenhao Cui, Guangrui Shen, Tieming Sun
> 
> EE 599 Deep Learning - Fall 2020

<p align="center"><img src="img/output_scene.gif" alt="Scene" width="500" /></p>

## Introduction

- Predict surrounding agents motions of the autonomous vehicle over 5s given their historical 1s positions
- Useful for planning self driving vehicle’s movement
- Deep learning techniques (CNN: Mixnet) + Ensemble Models
- Choose negative multi-log-likelihood as evaluate metric
- Full Information provided by [Kaggle](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/overview/description)

## Run Model
- Follow the instruction on [Lyft Website](https://self-driving.lyft.com/level5/data/) to download Dataset
- Use Jupyter Notebooks under directory "notebook" to run our model
- Or run python script under "code", first changing your path to dataset
- Competition can be found on [kaggle website](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/overview/description)
- Structure of this repo

```
- code - train.py
      |
       - test.py
      |
       - model.py
      |
       - utils.py

- data_model - pth
            |
             - metric

- notebook - train-cnn-nll.ipynb
          |
           - test-cnn.ipynb
```
