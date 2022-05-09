# Databricks notebook source
# MAGIC %md ## Scaling Financial Time Series with Apache Spark
# MAGIC 
# MAGIC #### Prerequisites: 
# MAGIC 
# MAGIC * Use an ML Cluster with DBR version `6.0`
# MAGIC * Install `plotly` and `koalas=0.18.0` or later
# MAGIC 
# MAGIC <!-- #Koalas #FinServ #timeseries -->
# MAGIC 
# MAGIC ## Business Value
# MAGIC 
# MAGIC One of the biggest technical challenges underlying problems in financial services is manipulating time series at scale.  Another major challenge is centralizing the wide variety of time series data sources, effectively unlocking potential value. Tick data, alternative data sets such as geospatial or transactional data, and fundamental economic data are examples of the rich data sources available to financial institutions, all of which are naturally indexed by timestamp. Solving business problems in finance such as risk, fraud, and compliance ultimately rests on being able to aggregate and analyze thousands of time series in parallel.
# MAGIC 
# MAGIC Below we'll show how to implement *as-of* joins for trading analysis. Then, we'll focus on data science on financial NBBO data using *Koalas* and native Python visualizations within Databricks to detect financial fraud in the form of market manipulation, particularly front running.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 1. Merging and Scaling Analyses with Delta Lake and Apache Spark

# COMMAND ----------

# MAGIC %sh 
# MAGIC 
# MAGIC wget https://pages.databricks.com/rs/094-YMS-629/images/ASOF_Quotes.csv ; wget https://pages.databricks.com/rs/094-YMS-629/images/ASOF_Trades.csv ; 

# COMMAND ----------

# MAGIC %fs cp file:/databricks/driver/ASOF_Quotes.csv /tmp/finserv/ASOF_Quotes.csv

# COMMAND ----------

# MAGIC %fs cp file:/databricks/driver/ASOF_Trades.csv /tmp/finserv/ASOF_Trades.csv

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### 1a) Ingest Trades and Quotes (raw csv), Convert to Delta Lake
# MAGIC 
# MAGIC ##### Why Delta Lake? 
# MAGIC 
# MAGIC * Delta Lake allows us to optimize formats from data providers (often large compressed flat files). This prevents value from getting locked away in silos.
# MAGIC * Apache Spark with Delta as our underlying performance engine gives us the scale needed to process thousands of tickers in parallel.

# COMMAND ----------

from pyspark.sql.types import *

trade_schema = StructType([
    StructField("symbol", StringType()),
    StructField("event_ts", TimestampType()),
    StructField("trade_dt", StringType()),
    StructField("trade_pr", DoubleType())
])

quote_schema = StructType([
    StructField("symbol", StringType()),
    StructField("event_ts", TimestampType()),
    StructField("trade_dt", StringType()),
    StructField("bid_pr", DoubleType()),
    StructField("ask_pr", DoubleType())
])

spark.read.format("csv").schema(trade_schema).option("header", "true").option("delimiter", ",").load("/tmp/finserv/ASOF_Trades.csv").write.mode('overwrite').format("delta").save('/tmp/finserv/delta/trades')

spark.read.format("csv").schema(quote_schema).option("header", "true").option("delimiter", ",").load("/tmp/finserv/ASOF_Quotes.csv").write.mode('overwrite').format("delta").save('/tmp/finserv/delta/quotes')

# COMMAND ----------

display(spark.read.format("delta").load("/tmp/finserv/delta/trades"))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### 1b) Define Helper Class for As-of Joins and VWAP Aggregation

# COMMAND ----------

from pyspark.sql import DataFrame
from pyspark.sql.functions import *
import pyspark.sql.functions as fn
from pyspark.sql.window import Window

import pandas as pd

class base_ts:
  
    def __init__(self, df):
      self.df = df
        
    # define custom data frame join which can scale to billions of quotes - this does not need to perform a full inner join but rather a UNION/SORT
    def join(self, other):
        """Returns the latest right value effective at the time of the left timestamp
        :param other: Right side of the dataset being evaluted
        """
        # select common fields to merge in single format
        left = self.df.select(col('event_ts'), col('price'), col('symbol'), col('bid'), col('offer'), col('ind_cd'))
        right = other.select('event_ts', 'price', 'symbol', 'bid', 'offer', 'ind_cd')
        un = left.union(right)
        
        # define partitioning keys for window
        partition_spec = Window.partitionBy('symbol')
        
        # define sort - the ind_cd is the indicator of whether the record type is a trade or quote (or whatever needs to be sorted first)
        join_spec = partition_spec.orderBy('event_ts', 'ind_cd').rowsBetween(Window.unboundedPreceding, Window.currentRow)
        
        # use the last_value functionality to get the latest effective record (quote) and attach to the trade proceeding the quote
        last_val_un=un.select(col('event_ts'), col('price'), col('symbol'), col('ind_cd'), fn.last("bid", True).over(join_spec).alias("latest_bid"), fn.last("offer", True).over(join_spec).alias("latest_offer"))
        return last_val_un.filter(col('ind_cd') ==1)
      
    def vwap(self, frequency='m'):
      
      # set pre_vwap as self or enrich with the frequency
      pre_vwap = self.df
      if frequency == 'm':
          pre_vwap = self.df.withColumn("time_group", concat(lpad(hour(col("event_ts")), 2, '0'), lit(':'), lpad(minute(col('event_ts')), 2, '0'))) 
      elif frequency == 'H':
          pre_vwap = self.df.withColumn("time_group", concat(lpad(hour(col("event_ts")), 2, '0')))
      elif frequency == 'D':
          pre_vwap = self.df.withColumn("time_group", concat(lpad(day(col("event_ts")), 2, '0')))
        
      vwapped = pre_vwap.withColumn("dllr_value", col("price")*col("volume")).groupby('symbol', 'time_group').agg(sum('dllr_value').alias("dllr_value"), sum('volume').alias("volume"), max('price').alias("max_price")).withColumn("vwap", col("dllr_value")/col("volume"))
      return vwapped

# COMMAND ----------

from pyspark.sql.functions import *

trades = spark.read.format("delta").load("/tmp/finserv/delta/trades") \
                                   .withColumnRenamed("trade_pr", "price") \
                                   .withColumn("bid", lit(None).cast("double")) \
                                   .withColumn("offer", lit(None).cast("double")) \
                                   .withColumn("ind_cd", lit(1)) 

quotes = spark.read.format("delta").load("/tmp/finserv/delta/quotes") \
                                   .withColumn("price", lit("")) \
                                   .withColumnRenamed("bid_pr", "bid") \
                                   .withColumnRenamed("ask_pr", "offer") \
                                   .withColumn("ind_cd", lit(-1)) 

# COMMAND ----------

display(spark.read.format("delta").load("/tmp/finserv/delta/trades"))

# COMMAND ----------

display(spark.read.format("delta").load("/tmp/finserv/delta/quotes"))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Demonstrate Merged Format

# COMMAND ----------

un= trades.filter(col("symbol") == "K").select('event_ts', 'price', 'symbol', 'bid', 'offer', 'ind_cd').union(quotes.filter(col("symbol") == "K").select('event_ts', 'price', 'symbol', 'bid', 'offer', 'ind_cd'))

display(un)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### 1c) Show Attached Bid/Offer For Each Trade

# COMMAND ----------

mkt_hrs_trades = trades.filter(col("symbol") == "K"). \
                        filter(col("event_ts") >= "2017-08-31 06:29"). \
                        filter(col("event_ts") <= "2017-08-31 16:00:00")

mkt_hrs_trades_ts = base_ts(mkt_hrs_trades)
quotes_ts = quotes.filter(col("symbol") == "K")

display(mkt_hrs_trades_ts.join(quotes_ts))

# COMMAND ----------

display(mkt_hrs_trades_ts.join(quotes_ts))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### 1d) Use Pre-Defined VWAP Function to Understand Trends 

# COMMAND ----------

trade_ts = base_ts(trades.select('event_ts', 'symbol', 'price', lit(100).alias("volume")))

vwap_df = trade_ts.vwap(frequency = 'm')

display(vwap_df.filter(col('symbol') == "K").filter(col('time_group').between('09:30', '16:00')).orderBy('time_group'))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 2. Detecting market manipulation with Koalas
# MAGIC 
# MAGIC 
# MAGIC <p></p>
# MAGIC 
# MAGIC * Note: The data obtained for the analysis below was obtained from the following source: https://www.tickdata.com/product/nbbo/

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### 2a) Download NBBO data

# COMMAND ----------

# MAGIC %sh wget https://s3-us-west-2.amazonaws.com/tick-data-s3/downloads/SampleEquityData_US.zip ; unzip SampleEquityData_US.zip

# COMMAND ----------

# MAGIC %fs cp file:/databricks/driver/NBBO/23444.csv /FileStore/tables/nbbo.csv

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import col, concat, to_date, unix_timestamp, lit

quote_schema = StructType([
    StructField("symbol", StringType()),
    StructField("event_ts", TimestampType()),
    StructField("trade_dt", StringType()),
    StructField("bid_pr", DoubleType()),
    StructField("ask_pr", DoubleType()),
    StructField("bid_shrs_qt", IntegerType()),
    StructField("ask_shrs_qt", IntegerType())
])

spark.read.format("csv").option("header", "true").option("delimiter", ",").load("/FileStore/tables/nbbo.csv"). \
                         withColumn("bid_shrs_qt", 100*col("Bid Size")). \
                         withColumn("ask_shrs_qt", 100*col("Ask Size")). \
                         withColumnRenamed("Bid Price", "bid_pr"). \
                         withColumnRenamed("Ask Price", "ask_pr"). \
                         withColumn("event_ts", concat(to_date((unix_timestamp(col("Date"), 'MM/dd/yyyy')).cast("timestamp")), lit(' '), col("Time")).cast("timestamp")). \
                         drop("Bid Size"). \
                         drop("Ask Size"). \
                         drop("Bid Exchange"). \
                         drop("Ask Exchange"). \
                         write.mode('overwrite').option("mergeSchema", "true").format("delta").save('/tmp/finserv/delta/ofi_quotes2')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### 2b) Understand depth at different best bid prices

# COMMAND ----------

from pyspark.sql.functions import *

# COMMAND ----------

import databricks.koalas as ks 

kdf = ks.read_delta("/tmp/finserv/delta/ofi_quotes2")
kdf_src = kdf.loc[kdf.Date == '03/05/2014']
kdf_src.head()

# COMMAND ----------

grouped_kdf = kdf_src.groupby(['event_ts'], as_index=False).max()
grouped_kdf.sort_values(by=['event_ts'])
grouped_kdf.head()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### 2c) Perform windowing, merging, and aggregation with `Koalas`

# COMMAND ----------

grouped_kdf.set_index('event_ts', inplace=True, drop=True)
lag_grouped_kdf = grouped_kdf.shift(periods=1, fill_value=0)

lag_grouped_kdf.head()

# COMMAND ----------

lagged = grouped_kdf.merge(lag_grouped_kdf, left_index=True, right_index=True, suffixes=['', '_lag'])
lagged.head()

# COMMAND ----------

type(lagged)

# COMMAND ----------

q = lagged

# compute supply and demand from the merged data frame
lagged['incr_demand'] = 1 if lagged.bid_pr >= lagged.bid_pr_lag else 0
lagged['decr_demand'] = 1 if lagged.bid_pr <= lagged.bid_pr_lag else 0
lagged['incr_supply'] = 1 if lagged.ask_pr <= lagged.ask_pr_lag else 0
lagged['decr_supply'] = 1 if lagged.ask_pr >= lagged.ask_pr_lag else 0

# perform arithmetic using koalas, avoid Spark syntax
lagged['imblnc_contrib'] = lagged['bid_shrs_qt']*lagged['incr_demand'] - lagged['bid_shrs_qt_lag']*lagged['decr_demand'] - lagged['ask_shrs_qt']*lagged['incr_supply'] + lagged['ask_shrs_qt_lag']*lagged['decr_supply']

lagged[['Symbol', 'Time', 'bid_pr', 'ask_pr', 'bid_shrs_qt', 'ask_shrs_qt', 'bid_shrs_qt_lag', 'ask_shrs_qt_lag', 'imblnc_contrib']].head()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### 2d) Compute Order Flow Imbalance as a proxy for market impact

# COMMAND ----------

from scipy.stats import t
import scipy.stats as st
import numpy as np

q_ofi_values = lagged['imblnc_contrib'].to_numpy()

# COMMAND ----------

bins=200
data = q_ofi_values[1:]


# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [        
        st.laplace, st.dgamma, st.powerlaw
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf
    
    other_distribution = st.norm
    other_params = (0.0, 1.0)
    other_sse = np.info

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:
        # Ignore warnings from data that can't be fit
            # fit dist to data
        params = distribution.fit(data)

        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1] 

        # Calculate fitted PDF and error with fit in distribution
        pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
        sse = np.sum(np.power(y - pdf, 2.0))

        # if axis pass in add to plot
        try:
            if ax:
                pd.Series(pdf, x).plot(ax=ax)
            end
        except Exception:
            'FAILURE FITTING!'

        # identify if this distribution is better
        if best_sse > sse > 0:
            best_distribution = distribution
            best_params = params
            best_sse = sse

    return (best_distribution.name, best_params)

# COMMAND ----------

def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

# COMMAND ----------

import pandas as pd
import statsmodels as sm
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

# Plot for comparison
data_flattened = pd.Series(data.flatten())

fig = plt.figure()
ax = fig.add_subplot(111)

# Load data from statsmodels datasets
data = q_ofi_values[1:]

# Find best fit distribution
best_fit_name, best_fit_params = best_fit_distribution(data, 200, ax)
best_dist = getattr(st, best_fit_name)

# Make PDF with best params 
pdf = make_pdf(best_dist, best_fit_params)

# Display
plt.figure(figsize=(10,5))
ax = pdf.plot(lw=2, label='best_pdf', legend=True)
data_flattened.plot(kind='hist', bins=1000, alpha=0.5, normed=True, label='OFI Bars', legend=True, ax=ax, color='b')
plt.legend(fontsize='small')

param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
dist_str = '{}({})'.format(best_fit_name, param_str)

ax.set_title(u'Order Flow Imbalance with Best Fit distribution \n' + dist_str)
ax.set_xlabel(u'Order Imbalance Using Supply/Demand')
ax.set_ylabel('Frequency')

# define endpoints for confidence intervals for best fit distribution
lb, ub = st.dgamma.interval(alpha = 0.90, a=best_fit_params[0], loc=best_fit_params[1], scale=best_fit_params[2])

plt.axvline(x=lb, color='red', linestyle='--')
plt.axvline(x=ub, color='red', linestyle='--')

# Update plots
ax.set_xlim([-4000, 4000])
ax.set_ylim([0, 0.0015])


fig = plt.show()
display(fig)

plt.gcf().clear()

# COMMAND ----------

# Start and end of the date range to extract
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

start, end = '2014-03-05 10:49:00', '2014-03-05 10:59:00'
lagged_pdf = lagged.toPandas()[['imblnc_contrib']]
# Plot daily, weekly resampled, and 7-day rolling mean time series together
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(lagged_pdf['imblnc_contrib'][start:end], marker='o', markersize='3', linestyle='-', linewidth=0.7, label='Intra-day Imbalance')
ax.scatter(x = lagged_pdf[(lagged_pdf['imblnc_contrib'] > ub) | (lagged_pdf['imblnc_contrib'] < lb)]['imblnc_contrib'][start:end].index, y = lagged_pdf[(lagged_pdf['imblnc_contrib'] > ub) | (lagged_pdf['imblnc_contrib'] < lb)]['imblnc_contrib'][start:end], color = 'orange', label='Anomaly')

ax.set_ylabel('Order Imbalance Value')
ax.legend();

fig = plt.show()
display(fig)

plt.gcf().clear()


# COMMAND ----------

import plotly.graph_objects as go
import datetime
import numpy as np
np.random.seed(1)

programmers = ['Anomalies (10s)']

lagged_pdf['is_anmly'] = ((lagged_pdf['imblnc_contrib'] > ub) | (lagged_pdf['imblnc_contrib'] < lb))
z = lagged_pdf['is_anmly'][start:end].resample('10S').sum().values.reshape(1, 60)

base = datetime.datetime(2014, 3, 15, 10, 59, 0)
dates = base - np.arange(600) * datetime.timedelta(seconds=10)

fig = go.Figure(data=go.Heatmap(
        x=dates,
        y=programmers,
        z = z,
        colorscale='Viridis'))

fig.update_layout(
    title='1D Order Imbalance Heat Map',
    xaxis_nticks=60)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Conclusion
# MAGIC 
# MAGIC #### 
# MAGIC 
# MAGIC * Koalas allowed us to easily manipulate large datasets and summarize
# MAGIC * Based on our Koalas feature engineering, we can then use rich statistical models in SciPy to find amomalies in our tick data
# MAGIC * Native Python visualizations in Databricks allow us to pinpoint time windows where we might find market manipulation and protect investors from trillion-dollar impacts

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC ls /dbfs/Users/srijith.rajamohan@databricks.com/
