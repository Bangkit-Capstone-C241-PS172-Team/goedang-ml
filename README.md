# Goedang-ML

## Overview
Goedang supports entrepreneurs by providing inventory management services that include price forecasts using machine learning technology. 
With this feature, Goedang will enable businesses to streamline their inventory operations, optimize stock levels, increase the accuracy of forecasts, 
and implement better decision-making processes so that they can drive growth and profitability.

## Data Understanding
The dataset used is agricultural company data, accessible at the following link:  [Dataset](https://docs.google.com/spreadsheets/d/18YY5jS95EebTn-KL7aOiyg-g07afwsVUZrsojaxfQTs/edit?usp=sharing)

## Result
In the system modeling, LSTM layers are used as follows:
```python
model = Sequential([
  LSTM(units = 64, return_sequences = True, input_shape = [None, 1], activation='tanh'),
  Dropout(0.2),
  LSTM(64, return_sequences=True, activation="tanh"),
  Dropout(0.2),
  LSTM(64, return_sequences=True, activation="tanh"),
  Dropout(0.2),
  LSTM(64, activation="tanh"),
  Dropout(0.2),
  Dense(1),
])
```
with loss: '0.0014" - mae: '0.0384' - val_loss: '7.5870e-04' - 'val_mae: 0.0261'

## Testing
To use the Testing Forecasting API, you can access it at the following link : `https://dep-prep-lh5lfcq2da-et.a.run.app/forecast`<br>
if with input as an example at the following link:

[Postman API](https://documenter.getpostman.com/view/36443503/2sA3XV7KAt)



