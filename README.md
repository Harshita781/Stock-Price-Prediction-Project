# Stock Price Prediction
Stock Market Analysis and Prediction is the project on technical analysis, visualization, and prediction using data provided by Google Finance. 
By looking at data from the stock market, particularly some giant technology stocks and others. Used pandas to get stock information,
visualized different aspects of it, and finally looked at a few ways of analyzing the risk of a stock, based on its previous performance history. 

![Stock-Price-Prediction](https://github.com/Harshita781/Stock-Price-Prediction-Project/blob/main/Images/1.png)
### Table of Contents
* [Installation](#installation)
* [Usage](#usage)
* [Documentation](#documentation)
* [Dependencies](#dependencies)
* [License](#license)

### Installation
Clone the git repository:
```console
$ git clone "https://github.com/Harshita781/Stock-Price-Prediction-Project.git"
```

Install necessary dependencies
```console
$ pip install -r requirements.txt
```

Go ahead and install as follows:
```console
$ python setup.py install
```

You may have to install TensorFlow:
```console
$ pip install tensorflow     # CPU
$ pip install tensorflow-gpu # GPU - Requires CUDA, CuDNN
```

### Usage
## Prediction Model

The app utilizes a Random Forest Classifier model trained on historical stock price data obtained from Yahoo Finance (via the `yfinance` library). The model predicts whether the stock price will increase or decrease based on features such as closing price, high, low, open, and volume.

##### a. Loading
Create a share object.
```python
>>> import bulbea as bb
>>> share = bb.Share('YAHOO', 'GOOGL')
>>> share.data
# Open        High         Low       Close      Volume  \
# Date                                                                     
# 2004-08-19   99.999999  104.059999   95.959998  100.339998  44659000.0   
# 2004-08-20  101.010005  109.079998  100.500002  108.310002  22834300.0   
# 2004-08-23  110.750003  113.479998  109.049999  109.399998  18256100.0   
# 2004-08-24  111.239999  111.599998  103.570003  104.870002  15247300.0   
# 2004-08-25  104.960000  108.000002  103.880003  106.000005   9188600.0
...
```
##### b. Preprocessing
Split your data set into training and testing sets.
```python
>>> from bulbea.learn.evaluation import split
>>> Xtrain, Xtest, ytrain, ytest = split(share, 'Close', normalize = True)
```

##### c. Modelling
```python
>>> import numpy as np
>>> Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))
>>> Xtest  = np.reshape( Xtest, ( Xtest.shape[0],  Xtest.shape[1], 1))

>>> from bulbea.learn.models import RNN
>>> rnn = RNN([1, 100, 100, 1]) # number of neurons in each layer
>>> rnn.fit(Xtrain, ytrain)
# Epoch 1/10
# 1877/1877 [==============================] - 6s - loss: 0.0039
# Epoch 2/10
# 1877/1877 [==============================] - 6s - loss: 0.0019
...
```

## Deployed Website

The app is already deployed and accessible online. You can try it out at [Stock Price Prediction App](https://stock-predictorv2.streamlit.app/#stock-predictor).

## Feedback and Contributions

Feedback, bug reports, and contributions are welcome! If you encounter any issues or have suggestions for improvement, please [open an issue](https://github.com/yourusername/stock-price-prediction-app/issues) or [submit a pull request](https://github.com/yourusername/stock-price-prediction-app/pulls).

## Acknowledgments

This project was inspired by the need for a simple and user-friendly tool to predict stock prices and visualize the results. Special thanks to the Streamlit and Yahoo Finance communities for their contributions and support.





