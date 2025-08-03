import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, BatchNormalization, LeakyReLU
from sklearn.utils import class_weight
from tensorflow.keras.metrics import Precision, Recall, AUC

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import os

def create_lstm_sequences(data, labels, window_length):
    X_seq, y_seq = [], []
    for i in range(len(data) - window_length):
        X_seq.append(data[i:i+window_length])
        y_seq.append(labels[i+window_length]) 
    return np.array(X_seq), np.array(y_seq)

def build_model_rnn(window_length, num_features):
    model = Sequential([
        Input(shape=(window_length, num_features)),
        LSTM(64),
        Dropout(0.2),
        Dense(32),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.01),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Has the limit in count
def simulate_trade(entry_price, highs, lows, direction, sl, tp):
    for i, (high, low) in enumerate(zip(highs, lows)):
        if direction == 1:
            if (high - entry_price) * 10_000 >= tp:
                return tp, i + 1
            elif (entry_price - low) * 10_000 >= sl:
                return -sl, i + 1
        elif direction == -1:
            if (entry_price - low) * 10_000 >= tp:
                return tp, i + 1
            elif (high - entry_price) * 10_000 >= sl:
                return -sl, i + 1
    final_price = highs[-1] if direction == 1 else lows[-1]
    result = (final_price - entry_price) * 10_000 * direction
    return result, len(highs)