import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import seaborn as sns
import math
import datetime
import keras
import warnings
import tensorflow as tf

warnings.filterwarnings("ignore")

from datetime import date, timedelta
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def plot_series(
    x,
    y,
    format="-",
    start=0,
    end=None,
    title=None,
    xlabel=None,
    ylabel=None,
    legend=None,
):
    """
    Visualizes time series data

    Args:
      x (array of int) - contains values for the x-axis
      y (array of int or tuple of arrays) - contains the values for the y-axis
      format (string) - line style when plotting the graph
      start (int) - first time step to plot
      end (int) - last time step to plot
      title (string) - title of the plot
      xlabel (string) - label for the x-axis
      ylabel (string) - label for the y-axis
      legend (list of strings) - legend for the plot
    """

    # Setup dimensions of the graph figure
    plt.figure(figsize=(10, 6))

    # Check if there are more than two series to plot
    if type(y) is tuple:

        # Loop over the y elements
        for y_curr in y:

            # Plot the x and current y values
            plt.plot(x[start:end], y_curr[start:end], format)

    else:
        # Plot the x and y values
        plt.plot(x[start:end], y[start:end], format)

    # Label the x-axis
    plt.xlabel(xlabel)

    # Label the y-axis
    plt.ylabel(ylabel)

    # Set the legend
    if legend:
        plt.legend(legend)

    # Set the title
    plt.title(title)

    # Overlay a grid on the graph
    plt.grid(True)

    # Draw the graph on screen
    plt.show()


def load_data():
    """
    Loads the data from the CSV file

    Returns:
      df (dataframe) - contains the data from the CSV file
    """

    # Load the data from the CSV file
    df = pd.read_csv("/content/dataset.csv")

    # Return the dataframe
    return df

def preprocess_data(df):
    """
    Preprocesses the data

    Args:
      df (dataframe) - contains the data from the CSV file

    Returns:
      df (dataframe) - contains the preprocessed data
    """

    # Convert tipe data
    df['QTY'] = df['QTY'].str.strip().str.replace(',', '').astype(float)
    df['HARGA'] = df['HARGA'].str.strip().str.replace(',', '').astype(float)
    df['JUMLAH'] = df['JUMLAH'].str.strip().str.replace(',', '').astype(float)
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], format='mixed')
    df.set_index('TANGGAL', inplace=True)

    # Return the dataframe
    return df

def main():
    df = load_data()
    preprocess_data(df)
    # Plot price
    df['HARGA'].plot(grid = True)
    plt.title("Harga CRMTT", color = 'black', fontsize = 20)
    plt.xlabel('Tahun', color = 'black', fontsize = 15)
    plt.ylabel('Harga', color = 'black', fontsize = 15)
    plt.show()

main()