import os
import pandas as pd
import numpy as np
import ta

# this uses a buffer to sort of filter the volatility of the market
# it is a more conservative approach than the previous one
def label_and_confidence(i, entry, highs, lows, sl, tp, buffer_candles, max_steps):
    tp_index = None
    sl_index = None

    max_long_fav = 0
    max_long_adv = 0
    max_short_fav = 0
    max_short_adv = 0

    for j in range(i, i + max_steps):
        high = highs[j] * 10000
        low = lows[j] * 10000
        price = entry[i] * 10000

        # Long direction
        long_fav = high - price
        long_adv = price - low
        max_long_fav = max(max_long_fav, long_fav)
        max_long_adv = max(max_long_adv, long_adv)

        # Short direction
        short_fav = price - low
        short_adv = high - price
        max_short_fav = max(max_short_fav, short_fav)
        max_short_adv = max(max_short_adv, short_adv)

        # Track first TP/SL hits for long and short
        if tp_index is None and high >= price + tp:
            tp_index = j
        if sl_index is None and low <= price - sl:
            sl_index = j

        if tp_index is not None and sl_index is not None:
            break

    # Determine label
    if tp_index is None and sl_index is None:
        label = 0  # No movement
    elif tp_index is not None and sl_index is not None:
        if max(tp_index, sl_index) <= i + buffer_candles:
            label = 0  # Too volatile: neutral
        else:
            label = 1 if tp_index < sl_index else 0
    elif tp_index is not None:
        label = 1 # Long
    else:
        label = 0  # Short

    # Compute confidence based on label
    if label == 1:  # Long
        confidence = (max_long_fav - max_long_adv) / (tp + sl)
        confidence = float(np.clip(confidence, -1, 1))

    else:
        confidence = 0.0
    
    return label, confidence


def build_eurusd_dataset_new(filename):
    # ---- 1. Load raw M1 ASCII -----------------------------------------------
    file_path = os.path.join(os.getcwd(), filename)
    df = pd.read_csv(
        file_path,
        sep=',',
        header=None,
        names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    )
    df = df.astype({'Open': float, 'High': float, 'Low': float, 'Close': float, 'Volume': float})
    # Combine date and time columns
    # Combine and reformat date and time columns to 'YYYYMMDDHHMM00'
    df['DateTime'] = df['Date'].astype(str).str.replace('.', '', regex=False) + df['Time'].astype(str).str.replace(':', '', regex=False) + '00'
    # Optionally, set as index or parse as datetime if needed
    df.index = pd.to_datetime(df['DateTime'], format='%Y%m%d%H%M%S')
    df = df.drop(columns=['Date', 'Time',  'DateTime', 'Volume'])

    # ---- 2. Resample ----------------------------------------------------------
    def resample_ohlc(data, rule):
        return data.resample(rule).agg({
            "Open": "first",
            "High": "max",
            "Low":  "min",
            "Close":"last",
        }).dropna()

    m5  = resample_ohlc(df,  '5min')

    # -3. Add temporal features --------------------------------------------
    m5["hour"]          = m5.index.hour
    m5["dayofweek"]     = m5.index.dayofweek
    m5["mins_into_m15"] = m5.index.minute % 15
    m5["frac_into_m15"] = m5["mins_into_m15"] / 15

    m5 = m5.drop(m5.between_time("22:00", "07:00").index)

    # ---- 4. M5 Technical indicators -------------------------------------------
    m5["rsi"]  = ta.momentum.RSIIndicator(m5["Close"], 14).rsi()
    macd = ta.trend.MACD(m5["Close"])
    m5["macd"] = macd.macd_diff()
    m5["body"] = m5["Close"] - m5["Open"]
    m5["vol_local"] = m5["High"] - m5["Low"]

    # ATR (Average True Range)
    atr = ta.volatility.AverageTrueRange(high=m5["High"], low=m5["Low"], close=m5["Close"], window=14)
    m5["atr"] = atr.average_true_range()

    # ADX (Average Directional Index)
    adx = ta.trend.ADXIndicator(high=m5["High"], low=m5["Low"], close=m5["Close"], window=14)
    m5["adx"] = adx.adx()
    '''m5["adx_pos"] = adx.adx_pos()
    m5["adx_neg"] = adx.adx_neg()'''

    # ---- 4b. Add Rolling M15 (3 x 5m) features --------------------------------
    m5["roll_m15_high"]  = m5["High"].rolling(3).max()
    m5["roll_m15_low"]   = m5["Low"].rolling(3).min()
    m5["roll_m15_close"] = m5["Close"].rolling(3).apply(lambda x: x.iloc[-1], raw=False)
    m5["roll_m15_trend"] = m5["Close"] - m5["Close"].rolling(3).mean()
    m5["roll_m15_body"]  = (m5["Close"] - m5["Open"]).rolling(3).mean()

    # ---- 4c. Add Rolling H1 (12 x 5m) features --------------------------------
    m5["roll_h1_high"]  = m5["High"].rolling(12).max()
    m5["roll_h1_low"]   = m5["Low"].rolling(12).min()
    m5["roll_h1_close"] = m5["Close"].rolling(12).apply(lambda x: x.iloc[-1], raw=False)
    m5["roll_h1_vol"]   = m5["roll_h1_high"] - m5["roll_h1_low"]
    m5["roll_h1_relpos"] = (m5["Close"] - m5["roll_h1_low"]) / (m5["roll_h1_high"] - m5["roll_h1_low"] + 1e-6)
    m5["returns"] = m5["Close"].pct_change()
    m5["returns_mean"] = m5["Close"].rolling(12).mean()
    m5["volatility"] = m5["returns"].rolling(12).std()

    # ---- 5. Build future-range label -----------------------------------------
    # Assumes 5-min candles
    max_steps = 12  # max candles to look ahead, put six to b more conservative
    entry = m5['Open'].shift(-1).values
    highs = m5['High'].shift(-1).values
    lows = m5['Low'].shift(-1).values
    sl = 15
    tp = 15
    buffer_candles = 0
    labels = []
    confidences = []

    for i in range(len(m5) - max_steps - 1):
        label, conf = label_and_confidence(
            i,
            entry,
            highs,
            lows,
            sl,
            tp,
            buffer_candles,
            max_steps
        )
        labels.append(label)
        confidences.append(conf)


    m5 = m5.iloc[:len(labels)].copy()

    m5['label'] = labels
    m5['confidence'] = confidences

    # ---- 6. Drop NaNs caused by rolling / label shift ------------------------
    m5 = m5.dropna()

    m5 = m5.drop(m5.between_time("07:00", "08:10").index)

    return m5