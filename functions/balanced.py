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

def optimize_sl_tp_per_class(y, close_prices, highs, lows, sl_values, tp_values, class_to_direction, cost_per_trade, steps):
    sl_tp_map = {}
    for cls in [0, 1, 2]:
        best_profit = -np.inf
        best_pair = (12, 20)
        for sl in sl_values:
            for tp in tp_values:
                temp_profit = 0
                count = 0
                for idx, pred in enumerate(y):
                    if pred != cls:
                        continue
                    direction = class_to_direction.get(pred, 0)
                    if direction == 0:
                        continue
                    # Make sure we have enough future data
                    if idx + steps > len(y):
                        continue
                    entry = close_prices[idx]
                    highs_seq = highs[idx:idx+steps]
                    lows_seq = lows[idx:idx+steps]
                    outcome, _ = simulate_trade(entry, highs_seq, lows_seq, direction, sl, tp)
                    temp_profit += outcome
                    temp_profit -= cost_per_trade
                    count += 1
                if count > 0 and temp_profit > best_profit:
                    best_profit = temp_profit
                    best_pair = (sl, tp)
        sl_tp_map[cls] = {'sl': best_pair[0], 'tp': best_pair[1]}
    sl_tp_map[1] = {'sl': None, 'tp': None}  # no-trade
    return sl_tp_map

def estimate_avg_duration_per_class(y, close_prices, highs, lows, sl_tp_map, class_to_direction, steps):
    duration_by_class = {0: [], 1: [], 2: []}

    for idx, pred in enumerate(y):
        direction = class_to_direction.get(pred, 0)
        if direction == 0:
            continue
        sltp = sl_tp_map.get(pred, {'sl': None, 'tp': None})
        if sltp['sl'] is None or sltp['tp'] is None:
            continue
        # Make sure we have enough future data
        if idx + steps > len(close_prices):
            continue
        entry = close_prices[idx]
        highs_seq = highs[idx:idx+steps]
        lows_seq = lows[idx:idx+steps]
        _, duration = simulate_trade(entry, highs_seq, lows_seq, direction, sltp['sl'], sltp['tp'])
        duration_by_class[pred].append(duration)

    avg_duration_by_class = {
        k: round(np.mean(v)) if v else 7 for k, v in duration_by_class.items()
    }
    return avg_duration_by_class


# not being used yet

def relabel_data(df, sl_tp_map, avg_duration_by_class, class_to_direction):
    relabeled = []
    for i in range(len(df)):
        row = df.iloc[i]
        label = row['label']
        direction = class_to_direction.get(label, 0)
        if direction == 0:
            relabeled.append(2)  # No-trade
            continue

        sltp = sl_tp_map.get(label, {'sl': None, 'tp': None})
        horizon = avg_duration_by_class.get(label)
        if sltp['sl'] is None or sltp['tp'] is None:
            relabeled.append(2)
            continue

        highs_seq = df['High'].iloc[i:i+horizon].values
        lows_seq = df['Low'].iloc[i:i+horizon].values
        if len(highs_seq) < horizon or len(lows_seq) < horizon:
            relabeled.append(2)
            continue

        result, _ = simulate_trade(row['Close'], highs_seq, lows_seq, direction, sltp['sl'], sltp['tp'])
        if result > 0:
            relabeled.append(label)
        else:
            relabeled.append(2)  # No-trade if neither hit
    return np.array(relabeled)


# see this last part of the result > 0 because then it's sus