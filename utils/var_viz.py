def plot_candlesticks(stock_df):
  import plotly.graph_objects as go
  layout = go.Layout(
    autosize=False,
    width=1200,
    height=800,
  )
  fig = go.Figure(
    data=[go.Candlestick(
      x=stock_df['date'], 
      open=stock_df['open'], 
      high=stock_df['high'], 
      low=stock_df['low'], 
      close=stock_df['close']
    )],
    layout=layout
  )
  fig.show()
  
  
def plot_var(simulations, var):
  
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  from scipy import stats
  from utils.var_utils import get_var
  
  mean = np.mean(simulations)
  z = stats.norm.ppf(1-var)
  m1 = np.min(simulations)
  m2 = np.max(simulations)
  std = np.std(simulations)
  q1 = get_var(simulations, var)

  x1 = np.arange(np.min(simulations), np.max(simulations), 0.001)
  y1 = stats.norm.pdf(x1, loc=mean, scale=std)
  x2 = np.arange(x1.min(), q1, 0.001)
  y2 = stats.norm.pdf(x2, loc=mean, scale=std)

  mc_df = pd.DataFrame(data = simulations, columns=['return'])
  ax = mc_df.hist(column='return', bins=50, density=True, grid=False, figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
  ax = ax[0]

  for x in ax:
      x.spines['right'].set_visible(False)
      x.spines['top'].set_visible(False)
      x.spines['left'].set_visible(False)
      x.axvline(x=q1, color='r', linestyle='dashed', linewidth=1)
      x.fill_between(x2, y2, zorder=3, alpha=0.4)
      x.plot(x1, y1, zorder=3)
      x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
      vals = x.get_yticks()
      for tick in vals:
          x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

      x.set_title("VAR{} = {:.3f}".format(var, q1), weight='bold', size=15)
      x.set_xlabel("Returns", labelpad=20, weight='bold', size=12)
      x.set_ylabel("Density", labelpad=20, weight='bold', size=12)