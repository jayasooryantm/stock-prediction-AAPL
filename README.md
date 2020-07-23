# stock-prediction-AAPL

## :warning: Warning!! Please do not use this model to perform trading. This model is built just for an educational purpose :warning:

### Table of Contents
* Overview
* Motivation
* Technical Aspect
* Requirements
* Run
* Technologies Used
* Credits

### Overview

A Machine Learning Model for Stock Market Prediction of AAPL (Apple Inc). Stock market prediction is the act of trying to determine the future value of company stock or other financial instrument traded on a financial exchange. An LSTM neural network which could predict the future stock price has been build and trained on top of TensorFlow framework. The LSTM model takes the previous 100 days stock price (closing price) as input and predict the next day price.

### Motivation

I have a keen interest in participating in stock market trading. Since I am not an expert in trading technique I could take any chances to lose my hard-earned money, I consider this as my very first version of trading insight dashboard to see how the stock market works. I would like to expand this project as a fully-fledged stock insight dashboard in future.

### Technical Aspect

* LSTM architecture has been used to build the model
* Tensorflow framework has used to coding
* Pandas library has used for feature engineering.


### Requirement

* Python (>=3.7) should be installed on your computer.
* Jupyter notebook or any other client that support jupyter notebook would be required to open the code.
* You should install the necessary libraries from the requirement.txt file
* GPU is preferred since you are training a neural network

### Run

You can use the StockPredictionModel folder to load the model.

1. Open a python file/jupyter notebook
1. Type the below code to predict the stock price:
(<x_values> should be replaced with suitable input values and make sure you have pointed to correct directory while loading the model)
```
import tensorflow as tf
stockPrice = tf.keras.models.load_model('StockPredictionModel')
stockPrice.predict(<x_values>)
```
### Technologies Used

* Python
* Tensorflow
* Sklearn
* Jupyter Notebook

### Credits

* Mentor - @krishnaik06
* Dataset - https://www.tiingo.com
