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

## Features

- **Prediction Model:** The app utilizes a Random Forest Classifier model to predict stock prices based on historical data.
- **Profit Calculation:** It includes a profit calculation function that estimates the profit or loss based on the predictions.
- **Advanced Settings:** Users can adjust advanced settings such as the start index and step size for the prediction range to customize their analysis.
- **Visualization:** The app provides interactive plots to visualize the predictions, actual stock values, and other relevant metrics.


## Deployed Website

The app is already deployed and accessible online. You can try it out at [Stock Price Prediction App](https://stock-price-prediction-by-harsh-ita.streamlit.app/).

## Feedback and Contributions

Feedback, bug reports, and contributions are welcome! If you encounter any issues or have suggestions for improvement, please [open an issue](https://github.com/Harshita781/Stock-Price-Prediction-Project/issues) or [submit a pull request](https://github.com/Harshita781/Stock-Price-Prediction-Project/pulls).

## Acknowledgments

This project was inspired by the need for a simple and user-friendly tool to predict stock prices and visualize the results. Special thanks to the Streamlit and Yahoo Finance communities for their contributions and support.





