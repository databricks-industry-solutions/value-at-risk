# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/value-at-risk on the `web-sync` branch. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/market-risk.

# COMMAND ----------

# MAGIC %md
# MAGIC # Create portfolio
# MAGIC In this notebook, we will use `yfinance` to download stock data for 40 equities in an equal weighted hypothetical Latin America portfolio. We show how to use pandas UDFs to better distribute this process efficiently and store all of our output data as a Delta table. 

# COMMAND ----------

# MAGIC %run ./config/var_config

# COMMAND ----------

portfolio = """
country,company,ticker,industry
CHILE,Banco de Chile,BCH,Banks
CHILE,Banco Santander-Chile,BSAC,Banks
CHILE,Compañía Cervecerías Unidas S.A.,CCU,Beverages
CHILE,Itaú CorpBanca,ITCB,Banks
CHILE,Enersis Chile SA Sponsored ADR,ENIC,Electricity
CHILE,"SQM-Sociedad Química y Minera de Chile, S.A.",SQM,Chemicals
COLOMBIA,BanColombia S.A.,CIB,Banks
COLOMBIA,Ecopetrol S.A.,EC,Oil & Gas Producers
COLOMBIA,Grupo Aval Acciones y Valores S.A,AVAL,Financial Services
MEXICO,"América Móvil, S.A.B. de C.V.",AMX,Mobile Telecommunications
MEXICO,América Móvil SAB de CV Sponsored ADR Class A,AMOV,Mobile Telecommunications
MEXICO,CEMEX S.A.B. de C.V. (CEMEX),CX,Construction & Materials
MEXICO,"Coca-Cola FEMSA, S.A.B. de C.V.",KOF,Beverages
MEXICO,"Controladora Vuela Compañía de Aviación, S.A.B. de C.V",VLRS,Travel & Leisure
MEXICO,"Fomento Económico Mexicano, S.A.B. de C.V. (FEMSA)",FMX,Beverages
MEXICO,"Grupo Aeroportuario del Pacífico, S.A.B. de C.V. (GAP)",PAC,Industrial Transportation
MEXICO,"Grupo Aeroportuario del Sureste, S.A. de C.V. (ASUR)",ASR,Industrial Transportation
MEXICO,"Grupo Financiero Santander México, S.A.B. de C.V",BSMX,Banks
MEXICO,"Grupo Simec, S.A. De CV. (ADS)",SIM,Industrial Metals & Mining
MEXICO,"Grupo Televisa, S.A.",TV,Media
MEXICO,"Industrias Bachoco, S.A.B. de C.V. (Bachoco)",IBA,Food Producers
PANAMA,"Banco Latinoamericano de Comercio Exterior, S.A.",BLX,Banks
PANAMA,"Copa Holdings, S.A.",CPA,Travel & Leisure
PERU,Cementos Pacasmayo S.A.A.,CPAC,Construction & Materials
PERU,Southern Copper Corporation,SCCO,Industrial Metals & Mining
PERU,Fortuna Silver Mines Inc.,FSM,Mining
PERU,Compañía de Minas Buenaventura S.A.,BVN,Mining
PERU,Credicorp Ltd.,BAP,Banks
"""

import pandas as pd
from io import StringIO

# create equally weighted index
portfolio_df = pd.read_csv(StringIO(portfolio))
portfolio_df['weight'] = 1.0
portfolio_df['weight'] = portfolio_df['weight'] / portfolio_df.shape[0]
portfolio_df

# COMMAND ----------

_ = (
  spark
    .createDataFrame(portfolio_df)
    .write
    .format('delta')
    .mode('overwrite')
    .saveAsTable(config['portfolio_table'])
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download stock data
# MAGIC We download stock market data (end of day) from yahoo finance, ensure time series are properly indexed and complete.

# COMMAND ----------

import datetime as dt
startdate = dt.datetime.strptime(config['yfinance_start'], "%Y-%m-%d").date()
enddate = dt.datetime.strptime(config['yfinance_stop'], "%Y-%m-%d").date()

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType
import yfinance as yf

schema = StructType(
  [
    StructField('ticker', StringType(), True), 
    StructField('date', TimestampType(), True),
    StructField('open', DoubleType(), True),
    StructField('high', DoubleType(), True),
    StructField('low', DoubleType(), True),
    StructField('close', DoubleType(), True),
    StructField('volume', DoubleType(), True),
  ]
)

@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def fetch_tick(group, pdf):
  tick = group[0]
  msft = yf.Ticker(tick)
  raw = msft.history(start=startdate, end=enddate)[['Open', 'High', 'Low', 'Close', 'Volume']]
  # fill in missing business days
  idx = pd.date_range(startdate, enddate, freq='B')
  # use last observation carried forward for missing value
  output_df = raw.reindex(idx, method='pad')
  # Pandas does not keep index (date) when converted into spark dataframe
  output_df['date'] = output_df.index
  output_df['ticker'] = tick    
  output_df = output_df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Volume": "volume", "Close": "close"})
  return output_df

# COMMAND ----------

_ = (
  spark
    .read
    .table(config['portfolio_table'])
    .groupBy('ticker')
    .apply(fetch_tick)
    .write
    .format('delta')
    .mode('overwrite')
    .saveAsTable(config['stock_table'])
)

display(spark.read.table(config['stock_table']))

# COMMAND ----------

# MAGIC %md
# MAGIC Databricks runtime come prepackaged with many python libraries such as plotly. We can represent a given instrument through a candlestick visualization

# COMMAND ----------

import plotly.graph_objects as go
from pyspark.sql import functions as F

stock_df = (
  spark
    .read
    .table(config['stock_table'])
    .filter(F.col('ticker') == portfolio_df.iloc[0].ticker)
    .orderBy(F.asc('date'))
    .toPandas()
)

layout = go.Layout(
  autosize=False,
  width=1600,
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download market factors
# MAGIC We assume that our various assets can be better described by market indicators and various indices, such as the S&P500, crude oil, treasury, or the dow. These indicators will be used later to create input features for our risk models

# COMMAND ----------

factors = {
  '^GSPC':'SP500',
  '^NYA':'NYSE',
  '^XOI':'OIL',
  '^TNX':'TREASURY',
  '^DJI':'DOWJONES'
}

# Create a pandas dataframe where each column contain close index
factors_df = pd.DataFrame()
for tick in factors.keys():    
    msft = yf.Ticker(tick)
    raw = msft.history(start=startdate, end=enddate)
    # fill in missing business days
    idx = pd.date_range(raw.index.min(), raw.index.max(), freq='B')
    # use last observation carried forward for missing value
    pdf = raw.reindex(idx, method='pad')
    factors_df[factors[tick]] = pdf['Close'].copy()
        
# Pandas does not keep index (date) when converted into spark dataframe
factors_df['date'] = idx

# COMMAND ----------

_ = (
  spark
    .createDataFrame(factors_df)
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(config['market_table'])
)

display(spark.read.table(config['market_table']))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute market volatility
# MAGIC As mentioned in the introduction, the whole concept of parametric VaR is to learn from past volatility. Instead of processing each day against its closest history sequentially, we can apply a simple window function to compute last X days' worth of market volatility at every single point in time, learning statistics behind those multi variate distributions

# COMMAND ----------

import numpy as np

def get_market_returns():
  
  f_ret_pdf = spark.table(config['market_table']).orderBy('date').toPandas()

  # add date column as pandas index for sliding window
  f_ret_pdf.index = f_ret_pdf['date']
  f_ret_pdf = f_ret_pdf.drop(columns = ['date'])

  # compute daily log returns
  f_ret_pdf = np.log(f_ret_pdf.shift(1)/f_ret_pdf)

  # add date columns
  f_ret_pdf['date'] = f_ret_pdf.index
  f_ret_pdf = f_ret_pdf.dropna()
  
  return (
    spark
      .createDataFrame(f_ret_pdf)
      .select(F.array(config['feature_names']).alias('features'), F.col('date'))
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Instead of recursively querying our data, we can apply a window function so that each insert of our table is "joined" with last X days worth of observations. We can compute statistics of market volatility for each window using simple UDFs

# COMMAND ----------

from pyspark.sql.functions import udf

@udf('array<double>')
def compute_avg(xs):
  import numpy as np
  mean = np.array(xs).mean(axis=0)
  return mean.tolist()
  
@udf('array<array<double>>')
def compute_cov(xs):
  import pandas as pd
  return pd.DataFrame(xs).cov().values.tolist()

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

days = lambda i: i * 86400 
volatility_window = Window.orderBy(F.col('date').cast('long')).rangeBetween(-days(config['past_volatility']), 0)

volatility_df = (
  get_market_returns()
    .select(
      F.col('date'),
      F.col('features'),
      F.collect_list('features').over(volatility_window).alias('volatility')
    )
    .filter(F.size('volatility') > 1)
    .select(
      F.col('date'),
      F.col('features'),
      compute_avg(F.col('volatility')).alias('vol_avg'),
      compute_cov(F.col('volatility')).alias('vol_cov')
    )
)

# COMMAND ----------

volatility_df.write.format('delta').mode('overwrite').saveAsTable(config['volatility_table'])

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we now have access to up to date indicators at every single point in time. For each day, we know the average of returns and our covariance matrix. These statistics will be used to generate random market conditions in our next notebook.

# COMMAND ----------

display(spark.read.table(config['volatility_table']))
