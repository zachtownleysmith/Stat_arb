import pandas as pd
from stat_arb import gen_stat_arb_position


def simulate_stat_arb(prices, eigen_cnt=15, param_days=60, k_min=252/30, long_op=-1.25, long_cl=-0.5,
                      short_op=1.25, short_cl=0.75, slip=1/1000):

    data = prices

    # First Day is generating a trading signal only
    data_sub = data.iloc[0:252]
    generate = gen_stat_arb_position(data_sub, eigen_cnt=eigen_cnt, param_days=param_days, k_min=k_min,
                                     long_op=long_op, long_cl=long_cl, short_op=short_op, short_cl=short_cl)

    long_arbs = generate['long_pos']
    short_arbs = generate['short_pos']
    close_trades = generate['close_trades']
    trade_signal = generate['new_trades'].sum()
    open_trades = pd.DataFrame(data=None, columns=data.columns)
    portfolio_log = pd.DataFrame(data=None, columns=data.columns)
    cash_log = pd.DataFrame(data=None, columns=['Cash'])

    # Add new trades to dataframe of open positions.
    # Remove closed trades from dataframe of open positions
    # Execute net trade of open/close signals from last timestep (add in slippage assumption)
    for i in range(1, data.shape[0]-252):
        # print((i, i+252))
        print(f'Trading Day: {data.index[i+252]}')
        open_trades = pd.concat([open_trades, generate['new_trades']])

        # Close Positions and Add net position to current timestep trade signal
        for closed in close_trades:
            # print(f'Removing: {closed}')
            trade_signal -= open_trades.loc[closed]
            open_trades = open_trades.drop(closed)

            if closed in long_arbs:
                long_arbs.remove(closed)

            if closed in short_arbs:
                short_arbs.remove(closed)
        close_trades = []

        # Execute net trading signal for the day
        # Update cash and portfolio positions (including trading cost / slippage)
        data_sub = data.iloc[i:i + 252]
        trade_cost = abs(trade_signal).dot(data_sub.iloc[-1].transpose()).sum() * slip
        cash = trade_signal.dot(data_sub.iloc[-1].transpose()) * -1 - trade_cost
        portfolio_log.loc[data_sub.index[-1]] = trade_signal
        cash_log.loc[data_sub.index[-1]] = cash

        # Generate trading signal for tomorrow, based on today's price data
        generate = gen_stat_arb_position(data_sub, eigen_cnt=eigen_cnt, param_days=param_days, k_min=k_min,
                                         long_op=long_op, long_cl=long_cl, short_op=short_op, short_cl=short_cl,
                                         long_arbs=long_arbs, short_arbs=short_arbs)
        trade_signal = generate['new_trades'].sum()
        close_trades = generate['close_trades']

    # On Final Day, close out all open positions
    trade_signal = open_trades.sum() * -1
    trade_cost = abs(trade_signal).dot(data.iloc[-1].transpose()).sum() * slip
    cash = trade_signal.dot(data.iloc[-1].transpose()) * -1 - trade_cost
    portfolio_log.loc[data.index[-1]] = trade_signal
    cash_log.loc[data.index[-1]] = cash

    print('')
    print('Trading profit')
    print(cash_log.cumsum().iloc[-1])
    print('')

    output = {'portfolio_log': portfolio_log, 'cash_log': cash_log}
    return output


if __name__ == '__main__':
    df = pd.read_csv('djia_5yr.csv', index_col='Date', parse_dates=True)
    test = simulate_stat_arb(prices=df.iloc[-265:], eigen_cnt=2)


