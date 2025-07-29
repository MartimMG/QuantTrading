import os
import pandas as pd
import ta  
from datetime import timedelta
 

def build_eurusd_dataset_continuous(filename):
    # ---- 1. Load raw M1 ASCII -----------------------------------------------
    file_path = os.path.join(os.getcwd(), filename)
    df = pd.read_csv(
        file_path,
        sep=',',
        header=None,
        names=['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    )
    df = df.astype({'Open': float, 'High': float, 'Low': float, 'Close': float, 'Volume': float})
    df.index = pd.to_datetime(df['Date_Time'], format='%Y%m%d %H%M%S')
    df = df.drop(columns=['Date_Time'])

    # ---- 2. Resample ----------------------------------------------------------
    def resample_ohlc(data, rule):
        return data.resample(rule).agg({
            "Open": "first",
            "High": "max",
            "Low":  "min",
            "Close":"last",
            "Volume":"sum"
        }).dropna()

    m5  = resample_ohlc(df,  '5min')
    # m15 = resample_ohlc(df, '15min')
    # h1  = resample_ohlc(df,  '1h')

    # ---- 3. M5 Technical indicators -------------------------------------------
    m5["rsi"]  = ta.momentum.RSIIndicator(m5["Close"], 14).rsi()
    macd = ta.trend.MACD(m5["Close"])
    m5["macd"] = macd.macd_diff()
    m5["body"] = m5["Close"] - m5["Open"]
    m5["vol_local"] = m5["High"] - m5["Low"]

    # ---- 3b. Add Rolling M15 (3 x 5m) features --------------------------------
    m5["roll_m15_high"]  = m5["High"].rolling(3).max()
    m5["roll_m15_low"]   = m5["Low"].rolling(3).min()
    m5["roll_m15_close"] = m5["Close"].rolling(3).apply(lambda x: x.iloc[-1], raw=False)
    m5["roll_m15_trend"] = m5["Close"] - m5["Close"].rolling(3).mean()
    m5["roll_m15_body"]  = (m5["Close"] - m5["Open"]).rolling(3).mean()

    # ---- 3c. Add Rolling H1 (12 x 5m) features --------------------------------
    m5["roll_h1_high"]  = m5["High"].rolling(12).max()
    m5["roll_h1_low"]   = m5["Low"].rolling(12).min()
    m5["roll_h1_close"] = m5["Close"].rolling(12).apply(lambda x: x.iloc[-1], raw=False)
    m5["roll_h1_vol"]   = m5["roll_h1_high"] - m5["roll_h1_low"]
    m5["roll_h1_relpos"] = (m5["Close"] - m5["roll_h1_low"]) / (m5["roll_h1_high"] - m5["roll_h1_low"] + 1e-6)
    m5["returns"] = m5["Close"].pct_change()
    m5["returns_mean"] = m5["Close"].rolling(12).mean()
    m5["volatility"] = m5["returns"].rolling(12).std()

    # -4. Add temporal features --------------------------------------------
    m5["hour"]          = m5.index.hour
    m5["dayofweek"]     = m5.index.dayofweek
    m5["mins_into_m15"] = m5.index.minute % 15
    m5["frac_into_m15"] = m5["mins_into_m15"] / 15

    # ---- 5. Build future-range label -----------------------------------------
    # Assumes 5-min candles
    tp_pips=25
    sl_pips=20
    max_steps=12

    labels = []
    time_to_tp = []
    time_to_sl = []
    runups = []
    drawdowns = []
    soft_labels = []

    highs = m5['High'].values
    lows = m5['Low'].values
    entry_prices = m5['Open'].shift(-1).values
    
    for i in range(len(m5)):
        label = 1  # default = no trade
        tp_hit = False
        sl_hit = False
        max_up = 0
        max_down = 0
        tp_step = None
        sl_step = None

        entry_price = entry_prices[i]

        for step in range(1, max_steps - 1):
            if i + step >= len(m5):
                break

            high = highs[i + step]
            low = lows[i + step]

            up_pips = (high - entry_price) * 10000
            down_pips = (entry_price - low) * 10000

            max_up = max(max_up, up_pips)
            max_down = max(max_down, down_pips)

            if not tp_hit and up_pips >= tp_pips:
                tp_hit = True
                tp_step = step

            if not sl_hit and down_pips >= sl_pips:
                sl_hit = True
                sl_step = step

            if tp_hit or sl_hit:
                break

        # Label logic
        if tp_hit and not sl_hit:
            label = 2  # strong long
        elif sl_hit and not tp_hit:
            label = 0  # strong short
        elif tp_hit and sl_hit:
            if tp_step < sl_step:
                label = 2 
            else: label = 0

        # Soft label logic (optional)
        if tp_hit or sl_hit:
            score = (tp_step or max_steps) - (sl_step or max_steps)
            score /= max_steps
        else:
            score = max_up / tp_pips if max_up > max_down else -max_down / sl_pips

        # Append outputs
        labels.append(label)
        time_to_tp.append(tp_step * 5 if tp_step else "No hit")  # in minutes
        time_to_sl.append(sl_step * 5 if sl_step else "No hit")
        runups.append(max_up)
        drawdowns.append(max_down)
        soft_labels.append(score)

    m5["label"] = labels
    m5["time_to_tp"] = time_to_tp
    m5["time_to_sl"] = time_to_sl
    m5["runup"] = runups
    m5["drawdown"] = drawdowns
    m5["confidence"] = soft_labels

    # ---- 8. Drop NaNs caused by rolling / label shift ------------------------
    m5 = m5.dropna()
    m5_outside_overlap = m5.drop(m5.between_time("12:00", "16:00").index)
    m5_inside_overlap = m5.between_time("12:00", "16:00")

    return m5