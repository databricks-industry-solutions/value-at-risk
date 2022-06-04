def download_market_data(tick, min_date, max_date):
    import pandas as pd
    import yfinance as yf
    msft = yf.Ticker(tick)
    raw = msft.history(start=min_date, end=max_date)[['Open', 'High', 'Low', 'Close', 'Volume']]
    # fill in missing business days
    idx = pd.date_range(min_date, max_date, freq='B')
    # use last observation carried forward for missing value
    output_df = raw.reindex(idx, method='pad')
    # Pandas does not keep index (date) when converted into spark dataframe
    output_df['date'] = output_df.index
    output_df['ticker'] = tick
    output_df = output_df.rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Volume": "volume", "Close": "close"})
    return output_df


def generate_prices(start_price, mu, sigma, days):
    import numpy as np
    shock = np.zeros(days)
    price = np.zeros(days)
    sample_rate = 1 / float(days)
    price[0] = start_price
    for i in range(1, days):
        shock[i] = np.random.normal(loc=mu * sample_rate, scale=sigma * np.sqrt(sample_rate))
        price[i] = max(0, price[i - 1] + shock[i] * price[i - 1])
    return price


def create_seed_df(runs):
    import pandas as pd
    import numpy as np
    return pd.DataFrame(list(np.arange(0, runs)), columns=['trial_id'])


def get_shortfall(simulations, var):
    import numpy as np
    var = get_var(simulations, var)
    return float(np.mean([s for s in simulations if s <= var]))


def get_var(simulations, var):
    import numpy as np
    return float(np.percentile(simulations, 100 - var))


def non_linear_features(xs):
    import numpy as np
    fs = []
    for x in xs:
        fs.append(x)
        fs.append(np.sign(x) * x ** 2)
        fs.append(x ** 3)
        fs.append(np.sign(x) * np.sqrt(abs(x)))
    return fs


def predict_non_linears(ps, fs):
    s = ps[0]
    for i, f in enumerate(fs):
        s = s + ps[i + 1] * f
    return float(s)
