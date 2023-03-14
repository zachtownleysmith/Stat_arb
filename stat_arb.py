import yfinance as yf
import pandas as pd
import numpy as np
import math
from sklearn import linear_model
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.tools

from warnings import simplefilter
#simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
simplefilter(action="ignore", category=statsmodels.tools.sm_exceptions.ValueWarning)


def gen_stat_arb_position(prices, eigen_cnt=15, param_days=60, k_min=252/30,
                          long_op=-1.25, long_cl=-0.5, short_op=1.25, short_cl=0.75, long_arbs=[], short_arbs=[]):
    '''
    prices: Pandas DF of Adjusted Closing price data. Function will use entire df to estimate eigenportfolio factors
    eigen_cnt: INT number of eigen factors to use in regression (default 15)
    param_days: INT number of days to use in OU process for trading signal generation (default 60)
    k_min: minimum mean reversion factor in OU process for stock to be considered (def = 252/30)

    long_op: s_score to open a long stock / short eigen trade (def < -1.25)
    long_cl: s_score to close a long stock / short eigen trade (def > -0.5)
    short_op: s_score to open a short stock / long eigen trade (def > 1.25)
    short_cl: s_score to close a short stock / long eigen trade (def < 0.75)

    long_arbs: list of stocks with a long stat arb position already (def = [])
    short_arbs: list of stocks with a short stat arb position already (def = [])
    ^position only generated if trade is not already in open long/short arbs
    '''

    # Obtain Returns, normalized returns, and eigenvectors of correlation matrix for stock universe
    if long_arbs is None:
        long_arbs = []
    rets = prices.pct_change()
    norm_rets = (rets - rets.mean()) / rets.std()

    rets = rets.dropna()
    norm_rets = norm_rets.dropna()

    corr_mtrx = norm_rets.corr()
    eigenvalues, eigenvectors = np.linalg.eig(corr_mtrx)

    #e_vals = [np.real(x) for x, _ in sorted(zip(eigenvalues, eigenvectors), reverse=True)]
    e_vecs = pd.DataFrame([np.real(x) for _, x in sorted(zip(eigenvalues, eigenvectors), reverse=True)])
    e_vecs.columns = rets.columns

    # Keep only selected eigen vectors. Add ability to optimize
    e_vecs = e_vecs.iloc[0:eigen_cnt]

    # Construct EigenPortfolios and EigenPortfolio Returns
    e_port = e_vecs / rets.std()
    e_port = e_port.div(e_port.sum(axis=1), axis=0)
    e_port_rets = rets.dot(e_port.transpose())

    # Regress last X-days of Asset Returns on X day PCA Factors (eigenreturns)
    rets_sub = rets.iloc[-param_days:]
    eigen_sub = e_port_rets.iloc[-param_days:]

    residual_df = pd.DataFrame()
    betas_df = pd.DataFrame()
    alphas = []
    for stock in rets_sub.columns:
        regr = linear_model.LinearRegression()
        regr.fit(eigen_sub, rets_sub[stock])
        predict = pd.DataFrame(regr.predict(eigen_sub), index=eigen_sub.index, columns=[stock])
        residual_df[stock] = predict[stock] - rets_sub[stock]
        betas_df[stock] = regr.coef_
        alphas.append(regr.intercept_ * 252)

    # alphas_df = pd.DataFrame(alphas, index=betas_df.columns)

    # Model cumulative residuals of regressed returns using eigenportfolio factors as an Ornstein-Uhlembeck Process
    # Can be fit to an AR(1) model using analytic solution to OU SDE.
    residual_df = residual_df.cumsum()
    ou_df = pd.DataFrame(index=['k', 'm', 'sigma', 'sigma_eq'])
    for stock in residual_df.columns:
        model = AutoReg(residual_df[stock], lags=1).fit()
        a = model.params['const']
        b = model.params[stock + '.L1']

        k = math.log(b) * -252
        m = a / (1 - b)
        sigma = (model.resid.var() * 2 * k / (1 - b ** 2)) ** (1 / 2)
        sigma_eq = (model.resid.var() / (1 - b ** 2)) ** (1 / 2)

        ou_df[stock] = [k, m, sigma, sigma_eq]

    ou_df.loc['m_bar'] = ou_df.loc['m'] - ou_df.mean(axis=1)['m']
    ou_df.loc['s_score'] = -ou_df.loc['m_bar'] / ou_df.loc['sigma_eq']
    #print('Ornstein-Uhlembeck Fitting:')
    #print(ou_df.loc['s_score'])
    #print('')

    holdings = pd.DataFrame(columns=ou_df.columns)
    # Generate positions for new long arbitrage trades
    potential_longs = ou_df[ou_df.columns[(ou_df.loc['s_score'] <= long_op) & (ou_df.loc['k'] >= k_min)]]
    for trade in potential_longs.columns:
        if trade not in long_arbs+short_arbs:
            long_arbs.append(trade)
            temp_df = pd.DataFrame(columns=ou_df.columns)
            for eigen in range(eigen_cnt):
                port_cost = prices.iloc[-1].dot(e_port.iloc[eigen].transpose())
                temp_df.loc['oplng_' + trade + str(eigen)] = -1 * prices.iloc[-1][trade] * \
                                                              betas_df[trade][eigen] / port_cost * e_port.iloc[eigen]

            temp_df.loc['oplng_' + trade + str(eigen_cnt)] = np.zeros(len(ou_df.columns))
            temp_df[trade]['oplng_' + trade + str(eigen_cnt)] = 1
            holdings.loc[trade] = temp_df.sum()

    # Generate positions for new short arbitrage trades
    potential_shorts = ou_df[ou_df.columns[(ou_df.loc['s_score'] >= short_op) & (ou_df.loc['k'] >= k_min)]]
    for trade in potential_shorts.columns:
        if trade not in short_arbs+long_arbs:
            short_arbs.append(trade)
            temp_df = pd.DataFrame(columns=ou_df.columns)
            for eigen in range(eigen_cnt):
                port_cost = prices.iloc[-1].dot(e_port.iloc[eigen].transpose())
                temp_df.loc['opsht_' + trade + str(eigen)] = 1 * prices.iloc[-1][trade] * \
                                                              betas_df[trade][eigen] / port_cost * e_port.iloc[eigen]

            temp_df.loc['opsht_' + trade + str(eigen_cnt)] = np.zeros(len(ou_df.columns))
            temp_df[trade]['opsht_' + trade + str(eigen_cnt)] = -1
            holdings.loc[trade] = temp_df.sum()

    close_positions = []
    # Check for open long/short positions that can be closed
    potential_shorts = ou_df[ou_df.columns[(ou_df.loc['s_score'] >= long_cl) & (ou_df.loc['k'] >= k_min)]]
    for trade in potential_shorts.columns:
        if trade in long_arbs:
            close_positions.append(trade)

    potential_longs = ou_df[ou_df.columns[(ou_df.loc['s_score'] <= short_cl) & (ou_df.loc['k'] >= k_min)]]
    for trade in potential_longs.columns:
        if trade in short_arbs:
            close_positions.append(trade)

    '''print('New Holdings:')
    print(holdings)
    print('')

    print('Positions to Close')
    print(close_positions)
    print('')

    print('Updated Long Arbs')
    print(long_arbs)
    print('')

    print('Updated Short Arbs')
    print(short_arbs)
    print('')
    '''

    output = {'new_trades': holdings, 'close_trades': close_positions,
              'long_pos': long_arbs, 'short_pos': short_arbs}

    return output


if __name__ == '__main__':

    data = yf.download(['AAPL', 'MSFT', 'AMZN', 'TSLA'], start="2022-02-01", end="2023-02-01")['Adj Close']

    # prices = yf.download(djia, start="2022-02-01", end="2023-02-01")['Adj Close']

    test = gen_stat_arb_position(data, eigen_cnt=2, param_days=60, k_min=252/30,
                                 long_op=-1.25, long_cl=-0.5, short_op=1.25, short_cl=0.75)

