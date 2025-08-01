import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import os


def generate_model_report_pdf(
    steps,
    extra_steps,
    window_indices,
    f1_per_window,
    profit_monetary_per_window,
    trades_per_window,
    volatility_per_window,
    local_vol_per_window,
    losing_profit,
    winning_profit,
    initial_account_balance,
    window_size,
    val_size,
    step,
    cost_per_trade,
    pip_value_per_standard_lot,
    risk_per_trade_percentage,
    winning_trades,
    losing_trades,
    profit_per_class,
    trades_per_class,
    threshold,
    train_distributions,
    val_distributions,
    report_filename
):

    total_profit = np.sum(profit_monetary_per_window)
    total_trades = np.sum(trades_per_window)
    avg_f1 = np.mean(f1_per_window)

    cumulative_balance = [initial_account_balance]
    for p in profit_monetary_per_window:
        cumulative_balance.append(cumulative_balance[-1] + p)
    cumulative_balance_arr = np.array(cumulative_balance)
    peak = cumulative_balance_arr[0]
    max_drawdown_percentage = 0
    if len(cumulative_balance_arr) > 1:
        for balance in cumulative_balance_arr:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak if peak != 0 else 0
            if drawdown > max_drawdown_percentage:
                max_drawdown_percentage = drawdown

    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 10})

    fig1, axes1 = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
    fig1.suptitle('Rolling Window Backtest Performance Metrics', fontsize=16)

    axes1[0, 0].plot(window_indices, profit_monetary_per_window, marker='o', linestyle='-', color='green', markersize=4)
    axes1[0, 0].set_title('Profit per Window (Dollars)')
    axes1[0, 0].set_xlabel('Window Index')
    axes1[0, 0].set_ylabel('Total Profit ($)')
    axes1[0, 0].grid(True)
    axes1[0, 0].axhline(0, color='gray', linestyle='--', linewidth=0.8)

    axes1[0, 1].plot(window_indices, trades_per_window, marker='o', linestyle='-', color='blue', markersize=4)
    axes1[0, 1].set_title('Number of Trades per Window')
    axes1[0, 1].set_xlabel('Window Index')
    axes1[0, 1].set_ylabel('Number of Trades')
    axes1[0, 1].grid(True)

    axes1[1, 0].plot(window_indices, f1_per_window, marker='o', linestyle='-', color='purple', markersize=4)
    axes1[1, 0].set_title('F1-score per Window (Weighted)')
    axes1[1, 0].set_xlabel('Window Index')
    axes1[1, 0].set_ylabel('F1-score')
    axes1[1, 0].grid(True)
    axes1[1, 0].set_ylim(0, 1)

    axes1[2, 0].plot(window_indices, volatility_per_window, marker='o', linestyle='-', color='orange', markersize=4)
    axes1[2, 0].set_title('Volatility per Window')
    axes1[2, 0].set_xlabel('Window Index')
    axes1[2, 0].set_ylabel('Volatility')
    axes1[2, 0].grid(True)
    axes1[2, 0].set_ylim(0, 0.0015)

    axes1[2, 1].plot(window_indices, local_vol_per_window, marker='o', linestyle='-', color='orange', markersize=4)
    axes1[2, 1].set_title('Local Volatility per Window')
    axes1[2, 1].set_xlabel('Window Index')
    axes1[2, 1].set_ylabel('Local Volatility')
    axes1[2, 1].grid(True)
    axes1[2, 1].set_ylim(0, 0.002)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot1_path = 'temp_metrics_plot.png'
    fig1.savefig(plot1_path)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(window_indices, cumulative_balance_arr[1:], color='darkgreen', linewidth=2)
    plt.title('Cumulative Account Balance Over Windows')
    plt.xlabel('Window Index')
    plt.ylabel(f'Cumulative Balance ($)')
    plt.grid(True)
    plt.axhline(initial_account_balance, color='orange', linestyle='--', linewidth=0.8, label='Initial Balance')
    plt.axhline(np.max(cumulative_balance_arr), color='blue', linestyle='--', linewidth=0.8, label='All-time High')
    plt.axhline(np.min(cumulative_balance_arr), color='red', linestyle='--', linewidth=0.8, label='All-time Low')
    plt.legend()
    plot2_path = 'temp_cumulative_plot.png'
    fig2.savefig(plot2_path)
    plt.close(fig2)

    # Add correlation heatmap of window statistics to the report
    df_stats = pd.DataFrame({
        'profit': profit_monetary_per_window,
        'num_trades': trades_per_window,
        'f1': f1_per_window,
        'volatility': volatility_per_window
    })

    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_stats.corr(), annot=True, cmap='coolwarm', ax=ax_corr)
    plt.title('Correlation Heatmap of Window Statistics')
    corr_plot_path = 'temp_corr_heatmap.png'
    fig_corr.savefig(corr_plot_path)
    plt.close(fig_corr)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Trading Model Performance Report", 0, 1, "C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", 0, 1, "C")
    pdf.ln(0.5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "1. Backtest Parameters", 0, 1, "L")
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 7, f"""
    - Window Size: {window_size} candles
    - Validation Size: {val_size} candles
    - Step Size: {step} candles
    - Cost per Trade: {cost_per_trade} pips
    - Pip Value (Standard Lot): ${pip_value_per_standard_lot}
    - Risk per trade (e.g., 0.1 for Mini Lot): {risk_per_trade_percentage}
    - Initial Account Balance: ${initial_account_balance:,.2f}
    - Extra Steps in the simulate trade: {extra_steps}
    - Total Steps in the prediction: {steps}
    - Threshold for trade decision: {threshold}
    """)
    pdf.ln(0.5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "2. Overall Performance Summary", 0, 1, "L")
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 7, f"""
    - Total Profit: ${total_profit:,.2f}
    - Total Trades: {total_trades:,}
    - Average F1-score (Weighted): {avg_f1:.3f}
    - Maximum Drawdown: {max_drawdown_percentage * 100:.2f}%
    - Final Account Balance: ${cumulative_balance_arr[-1]:,.2f}
    - All-time High Balance: ${np.max(cumulative_balance_arr):,.2f}
    - All-time Low Balance: ${np.min(cumulative_balance_arr):,.2f}
    - Winning Trades: {winning_trades:,}
    - Losing Trades: {losing_trades:,}
    - Winning Trade Profit: ${np.mean(winning_profit):,.2f} (Average)
    - Losing Trade Profit: ${np.mean(losing_profit):,.2f} (Average)
    """)
    pdf.ln(0.5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "3. Profit Per Predicted Class", 0, 1, "L")
    pdf.set_font("Arial", "", 10)

    for cls in sorted(profit_per_class.keys()):
        total_profit_cls = profit_per_class[cls]
        total_trades_cls = trades_per_class[cls]
        avg_profit_cls = total_profit_cls / total_trades_cls if total_trades_cls else 0
        pdf.cell(0, 7, f"Class {cls}: Trades = {total_trades_cls}, "
                           f"Total Profit = ${total_profit_cls:,.2f}, "
                           f"Avg Profit/Trade = ${avg_profit_cls:,.2f}", ln=1)

    pdf.ln(0.5)

    total_classified_trades = winning_trades + losing_trades
    win_rate = (winning_trades / total_classified_trades) * 100 if total_classified_trades else 0
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "4. Risk Metrics", 0, 1, "L")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 7, f"Win Rate: {win_rate:.2f}%", ln=1)

    returns = np.array(profit_monetary_per_window)
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe_ratio = avg_return / std_return if std_return > 0 else 0
    pdf.cell(0, 7, f"Sharpe Ratio: {sharpe_ratio:.2f}", ln=1)

    pdf.ln(2)

    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "3. Performance Visualizations", 0, 1, "L")
    pdf.ln(2)

    pdf.image(plot1_path, x=10, y=pdf.get_y(), w=180)
    pdf.ln(fig1.get_size_inches()[1] * 13)

    pdf.image(plot2_path, x=10, y=pdf.get_y(), w=180)
    pdf.ln(fig2.get_size_inches()[1] * 10)

    pdf.add_page()

    pdf.image(corr_plot_path, x=10, y=pdf.get_y(), w=180)
    pdf.ln(fig_corr.get_size_inches()[1] * 8)

    pdf.output(report_filename)

    os.remove(plot1_path)
    os.remove(plot2_path)
    os.remove(corr_plot_path)

    print(f"\nReport generated successfully: {report_filename}")
