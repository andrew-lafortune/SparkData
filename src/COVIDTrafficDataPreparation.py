# Databricks notebook source
# MAGIC %md
# MAGIC # Predicting the Effects of COVID-19 Restrictions Using Historical Traffic Data
# MAGIC ## By: Andrew LaFortune, May 6th 2022
# MAGIC 
# MAGIC ### Scenario
# MAGIC Two years into the COVID-19 pandemic and starting to see the light at the end of the tunnel, the Minnesota Governor's Office has started to reflect on the actions they took early on in the pandemic. They have asked LFDS (LaFortune Data Solutions) to find a way to numerically evaluate the effectiveness of stay-at-home and reopening policies not just in terms of the number of COVID cases, but in other less obvious numbers that might be found in other datasets online. These other data points are up to LFDS to find and examine. The final product to be delivered is a model that might be used in future pandemic situations to determine which restrictions and relaxations on businesses to deploy and in what time frame to minimize the spread of disease. 
# MAGIC 
# MAGIC ### Approach
# MAGIC The primary dataset that will bridge the gap between the Governor's policies and COVID-19 cases is the Minnesota Department of Transportation's (MNDOT) Automatic Traffic Recorder (ATR) and Weigh in Motion (WIM) [Hourly Volume Data](http://www.dot.state.mn.us/traffic/data/data-products.html). Some work has already been done with this data to visualize the percent difference in traffic volume from the average daily traffic volume across all days 2016-2019 ([PDF Download](https://edocs-public.dot.state.mn.us/edocs_public/DMResultSet/download?docId=12227832)). LFDS believes this is a good first step, but would like to produce a model that adjusts for nuances in typical daily or seasonal traffic patterns. This goal is summarized by Deliverable 1:
# MAGIC > Deliverable 1: Graph Daily Volume Change for each MNDOT District and statewide compared to daily baseline.
# MAGIC __Note:__ The 2016 volume data is no longer available, so the baseline to compare will be from 2017-2019
# MAGIC 
# MAGIC The goal of this report is to see how key policy decisions affected COVID cases. To see this, LFDS gathered several COVID case count datasets from the [New York Times COVID-19 Data GitHub](https://github.com/nytimes/covid-19-data). These datasets provide counts from January 21st, 2020 to the present day (May 3rd at the latest update of this notebook) at country-wide, state-wide, and county-wide levels. Compiling these into graphs for comparison with the traffic volume data will be the goal of Deliverable 2:
# MAGIC > Deliverable 2: Graph Daily COVID-19 case and death counts in Minnesota.
# MAGIC 
# MAGIC These graphs will likely show some clear trends in traffic and COVID cases that may have some visual relation and indicate when major policy changes occured. To verify these instances and work towards a model for gauging effectiveness of policy changes, the events from BallotPedia's [Documenting Minnesota's path to recovery from the coronavirus (COVID-19) pandemic, 2020-2021"](https://ballotpedia.org/Documenting_Minnesota%27s_path_to_recovery_from_the_coronavirus_(COVID-19)_pandemic,_2020-2021) article have been compiled into a spreadsheet with dates and classifications for each notable event. The events can be added to the plots of traffic volume and COVID-19 cases to paint a clearer picture of what happened immediately following policy changes for Deliverable 3:
# MAGIC > Deliverable 3: Deliverable 1 and 2 graphs with vertical lines indicating when policy changes went into effect.
# MAGIC 
# MAGIC Deliverables 1, 2, and 3 will give some idea of __what happened__. The last piece of this project is to condense that data into a useful model that can be used to predict __what will happen__ if similar restriction/relaxation measures are taken in a future pandemic situation. LFDS has decided that the best model for the job is a Time Series Forecasing model that takes as inputs the proportion of expected traffic volume for 30 days prior to an event predict the target variable of daily traffic volume percent difference from baseline for next 10 days.
# MAGIC 
# MAGIC Deliverable 4 will consist of testing the Facebook Prophet time series prediction model for this task with the goal of minimizing the error between model prediction and real value for traffic volume percent difference. This will include the test scenarios:
# MAGIC - a baseline prediction on a non-COVID year
# MAGIC - a prediction immediately following each type of COVID policy change
# MAGIC 
# MAGIC Based on the results of these scenarios, a recommendation to use Facebook's Prophet model or to look for other options:
# MAGIC > Deliverable 4: A recommendation of Time Series Forecasting model, or a recommendation to look for other prediction methods.

# COMMAND ----------

# imports
from pyspark.sql.types import *
from pyspark.sql.functions import mean, lit, min as colmin, max as colmax, sum as colsum, col, when, count, datediff, lag, countDistinct, to_date, regexp_replace, udf, date_format,desc
from pyspark.sql.window import Window
from operator import add
from functools import reduce

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Loading

# COMMAND ----------

# MAGIC %md
# MAGIC ### Available Data Sources

# COMMAND ----------

# MAGIC %sh
# MAGIC rm -r /dbfs/lafor038;
# MAGIC rm -r lafor038;

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir lafor038;
# MAGIC cd lafor038;
# MAGIC pwd;
# MAGIC mkdir wim_volume;
# MAGIC cd wim_volume;
# MAGIC wget https://github.com/andrew-lafortune/SparkData/raw/main/data/2017_wim_atr_volume.csv;
# MAGIC wget https://github.com/andrew-lafortune/SparkData/raw/main/data/2018_wim_atr_volume.csv;
# MAGIC wget https://github.com/andrew-lafortune/SparkData/raw/main/data/2019_wim_atr_volume.csv;
# MAGIC wget https://github.com/andrew-lafortune/SparkData/raw/main/data/2020_wim_atr_volume.csv;
# MAGIC wget https://github.com/andrew-lafortune/SparkData/raw/main/data/2021_wim_atr_volume.csv;
# MAGIC wget https://github.com/andrew-lafortune/SparkData/raw/main/data/2022_wim_atr_volume.csv;

# COMMAND ----------

# MAGIC %sh
# MAGIC cd lafor038;
# MAGIC mkdir covid_cases;
# MAGIC cd covid_cases;
# MAGIC wget https://github.com/nytimes/covid-19-data/raw/master/us-states.csv;
# MAGIC wget https://github.com/nytimes/covid-19-data/raw/master/us.csv;
# MAGIC mkdir us_counties;
# MAGIC cd us_counties;
# MAGIC wget https://github.com/nytimes/covid-19-data/raw/master/us-counties-2020.csv;
# MAGIC wget https://github.com/nytimes/covid-19-data/raw/master/us-counties-2021.csv;
# MAGIC wget https://github.com/nytimes/covid-19-data/raw/master/us-counties-2022.csv;

# COMMAND ----------

# MAGIC %sh
# MAGIC cd lafor038;
# MAGIC mkdir stations_counties_districts;
# MAGIC cd stations_counties_districts;
# MAGIC 
# MAGIC wget https://github.com/andrew-lafortune/SparkData/raw/main/data/Current_CC_StationList.csv;
# MAGIC wget https://github.com/andrew-lafortune/SparkData/raw/main/data/Retired_CC_StationList.csv;
# MAGIC wget https://github.com/andrew-lafortune/SparkData/raw/main/data/MN_Counties_Districts.csv;

# COMMAND ----------

# MAGIC %sh
# MAGIC cd lafor038;
# MAGIC wget https://github.com/andrew-lafortune/SparkData/raw/main/data/MN_COVID_Timeline.csv;

# COMMAND ----------

# MAGIC %sh ls lafor038

# COMMAND ----------

# MAGIC %sh cp -r lafor038 /dbfs/lafor038

# COMMAND ----------

# MAGIC %fs ls file:/dbfs/lafor038

# COMMAND ----------

# MAGIC %md
# MAGIC ### WIM ATR Volume Data

# COMMAND ----------

# MAGIC %sh ls /dbfs/lafor038/wim_volume

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define Schema
# MAGIC Columns identify station, direction, and lane of observation
# MAGIC 
# MAGIC Each Numbered column represents an hour of the day (e.g. 1 = 1AM)

# COMMAND ----------

wim_volume_schema = StructType([StructField('station_id',IntegerType(),True),
                               StructField('dir_of_travel',IntegerType(),True),
                               StructField('lane_of_travel',IntegerType(),True),
                               StructField('date',StringType(),True),
                               StructField('1',IntegerType(),True),
                               StructField('2',IntegerType(),True),
                               StructField('3',IntegerType(),True),
                               StructField('4',IntegerType(),True),
                               StructField('5',IntegerType(),True),
                               StructField('6',IntegerType(),True),
                               StructField('7',IntegerType(),True),
                               StructField('8',IntegerType(),True),
                               StructField('9',IntegerType(),True),
                               StructField('10',IntegerType(),True),
                               StructField('11',IntegerType(),True),
                               StructField('12',IntegerType(),True),
                               StructField('13',IntegerType(),True),
                               StructField('14',IntegerType(),True),
                               StructField('15',IntegerType(),True),
                               StructField('16',IntegerType(),True),
                               StructField('17',IntegerType(),True),
                               StructField('18',IntegerType(),True),
                               StructField('19',IntegerType(),True),
                               StructField('20',IntegerType(),True),
                               StructField('21',IntegerType(),True),
                               StructField('22',IntegerType(),True),
                               StructField('23',IntegerType(),True),
                               StructField('24',IntegerType(),True)])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load Each File and Save as Parquet Table

# COMMAND ----------

# MAGIC %fs ls file:/dbfs/lafor038

# COMMAND ----------

file_dir = "file:/dbfs/lafor038/wim_volume/"
file_type = "csv"
f_name_suffix = "_wim_atr_volume.csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

wim_volume = {}

for year in range(2017,2023):
    file_location = file_dir + str(year) + f_name_suffix
    wim_volume[year] = spark.read.format(file_type) \
      .schema(wim_volume_schema) \
      .option("header", first_row_is_header) \
      .option("sep", delimiter) \
      .load(file_location)
    
    table_name = str(year)+"_wim_atr_volume"

    wim_volume[year].createOrReplaceTempView(table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### COVID Case Data

# COMMAND ----------

# MAGIC %sh ls lafor038/covid_cases

# COMMAND ----------

# MAGIC %sh ls lafor038/covid_cases/us_counties

# COMMAND ----------

#define schemas for each level of case and death data
us_covid_schema = StructType([StructField('date',StringType(),True),
                             StructField('cases',IntegerType(),True),
                             StructField('deaths',IntegerType(),True)])

states_covid_schema = StructType([StructField('date',StringType(),True),
                                  StructField('state',StringType(),True),
                                  StructField('fips',IntegerType(),True),
                                  StructField('cases',IntegerType(),True),
                                  StructField('deaths',IntegerType(),True)])

counties_covid_schema = StructType([StructField('date',StringType(),True),
                                    StructField('county',StringType(),True),
                                    StructField('state',StringType(),True),
                                    StructField('fips',IntegerType(),True),
                                    StructField('cases',IntegerType(),True),
                                    StructField('deaths',IntegerType(),True)])

#file location and type
covid_dir = "file:/dbfs/lafor038/covid_cases/"
file_type = "csv"

# CSV options
first_row_is_header = "true"
delimiter = ","

# read files for country and state wide case counts and save to Parquet
us_cases = spark.read.format(file_type).schema(us_covid_schema).option("header",first_row_is_header).option("sep",delimiter).load(covid_dir+"us.csv")
state_cases = spark.read.format(file_type).schema(states_covid_schema).option("header",first_row_is_header).option("sep",delimiter).load(covid_dir+"us-states.csv")

us_cases.createOrReplaceTempView("us_covid_cases")
state_cases.createOrReplaceTempView("state_covid_cases")                                                             
                                                             
# read and write for each year of county data                                                             
counties = {}                                                             
for year in range(2020,2023):    
    file_location = covid_dir+"us_counties/us-counties-"+str(year)+".csv"                                                            
    counties[year] = spark.read.format(file_type) \
      .schema(counties_covid_schema) \
      .option("header", first_row_is_header) \
      .option("sep", delimiter) \
      .load(file_location)
    
    table_name = "county_cases_"+str(year)                                                             
    counties[year].createOrReplaceTempView(table_name)   

# COMMAND ----------

# MAGIC %md
# MAGIC ### Station Info
# MAGIC Each station has columns indicating location, type, number of lanes, etc. Most important is the "County Name" column which can be used to connect WIM recordings to specific counties for comparison with COVID case counts.
# MAGIC 
# MAGIC The MN_Counties_Districts data connects county names to district numbers to mimic the graphs from MNDOT which are being recreated for Deliverable 1

# COMMAND ----------

# MAGIC %sh ls lafor038/stations_counties_districts

# COMMAND ----------

current_station_schema = StructType([StructField("Continuous Number",IntegerType(),True),
                             StructField("Sequence Number",IntegerType(),True),
                             StructField("Collection Type",StringType(),True),
                             StructField("Route",StringType(),True),
                             StructField("Pos Dir Dir",StringType(),True),
                             StructField("Pos Lanes",IntegerType(),True),
                             StructField("Neg Lanes",IntegerType(),True),
                             StructField("Urban/Rural",StringType(),True),
                             StructField("Functional Class",StringType(),True),
                             StructField("County Name",StringType(),True),
                             StructField("Location Text",StringType(),True)])

retired_station_schema = StructType([StructField("Continuous Number",IntegerType(),True),
                                     StructField("Collection Type",StringType(),True),
                                     StructField("Measurement",StringType(),True),
                                     StructField("Route",StringType(),True),
                                     StructField("Pos Lanes",IntegerType(),True),
                                     StructField("Pos Dir Dir",StringType(),True),
                                     StructField("Ref Post",StringType(),True),
                                     StructField("True Mile",FloatType(),True),
                                     StructField("Location",StringType(),True),
                                     StructField("County",StringType(),True),
                                     StructField("Closest City",StringType(),True),
                                     StructField("Status",StringType(),True)])

county_district_schema = StructType([StructField("County",StringType(),True),
                                    StructField("MnDOT district(s)",StringType(),True)])

# COMMAND ----------

file_location = "file:/dbfs/lafor038/stations_counties_districts/"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

current_stations = spark.read.format(file_type).schema(current_station_schema).option("header",first_row_is_header).option("sep",delimiter).load(file_location+"Current_CC_StationList.csv")
retired_stations = spark.read.format(file_type).schema(retired_station_schema).option("header",first_row_is_header).option("sep",delimiter).load(file_location+"Retired_CC_StationList.csv")
county_district = spark.read.format(file_type).schema(county_district_schema).option("header",first_row_is_header).option("sep",delimiter).load(file_location+"MN_Counties_Districts.csv")

current_stations.createOrReplaceTempView("current_stations")
retired_stations.createOrReplaceTempView("retired_stations")
county_district.createOrReplaceTempView("county_district")

# COMMAND ----------

# MAGIC %md
# MAGIC ## COVID Timeline
# MAGIC This data was made by hand based on data in an article from [BallotPedia](https://ballotpedia.org/Documenting_Minnesota%27s_path_to_recovery_from_the_coronavirus_%28COVID-19%29_pandemic,_2020-2021). It classifies each significant policy event in Minnesota throughout the pandemic by date, type, and category of society affected to be aligned with traffic volume and COVID data later on.

# COMMAND ----------

timeline_schema = StructType([StructField("Date",StringType(),True),
                              StructField("Category",StringType(),True),
                              StructField("Order Type",StringType(),True),
                              StructField("Event Type",StringType(),True),
                              StructField("Description",StringType(),True)])

file_location = "file:/dbfs/lafor038/"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

timeline = spark.read.format(file_type).schema(timeline_schema).option("header",first_row_is_header).option("sep",delimiter).load(file_location+"MN_COVID_Timeline.csv")
timeline.createOrReplaceTempView("MN_COVID_Timeline")

# COMMAND ----------

# MAGIC %sql
# MAGIC show views;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Analysis (exploration)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Timeline
# MAGIC This data was made by hand for this project, so the contents shouldn't have anything unexpected. Exploration will consist of checking the different categories available.

# COMMAND ----------

timeline.groupBy("Category", "Order Type", "Event Type").count().orderBy(desc("count")).display()

# COMMAND ----------

# MAGIC %md
# MAGIC There are 23 different groupings of Category, Order Type, and Event Type with Business regulations being the most common by far. The most useful points in the policy timeline will probably be when Business Relaxations or Restrictions go into effect.

# COMMAND ----------

# MAGIC %md
# MAGIC #### COVID Data
# MAGIC Test for: 
# MAGIC - Completeness: no null values, complete time series
# MAGIC - Uniqueness: how many unique values for each dataset?
# MAGIC - Validity: does the data make sense? (e.g. no negative case counts)

# COMMAND ----------

us_cases.limit(100).display()

# COMMAND ----------

# MAGIC %md
# MAGIC Looks like the counts per date are a running total.

# COMMAND ----------

us_cases.select([count(when(col(c).isNotNull() , c)).alias("notNull"+c.capitalize()) for c in us_cases.columns]
   ).withColumn("expected",lit(us_cases.count())).show()

us_cases.select([count(when(col(c) >= 0 , c)).alias("pos"+c.capitalize()) for c in ["cases","deaths"]]
   ).withColumn("expected",lit(us_cases.count())).show()

us_cases.select(mean((us_cases["deaths"] <= us_cases["cases"]).cast("int")).alias("isValidDeathCount")).show()

windowSpec = Window.orderBy("date")
us_cases.orderBy("date") \
        .withColumn("date_diff",datediff(us_cases.date,lag("date",1).over(windowSpec))) \
        .select(mean("date_diff").alias("isValidTimeStep")).withColumn("expected",lit(1.0)).show()

us_cases.select("date").distinct().count() == us_cases.count()

# COMMAND ----------

state_cases.limit(100).display()
state_cases.filter(state_cases["state"] == "Washington").limit(100).display()

# COMMAND ----------

# MAGIC %md
# MAGIC Again, running total of cases and deaths.

# COMMAND ----------

state_cases.select([count(when(col(c).isNotNull() , c)).alias("notNull"+c.capitalize()) for c in state_cases.columns]
   ).withColumn("expected",lit(state_cases.count())).show()

state_cases.select([count(when(col(c) >= 0 , c)).alias("pos"+c.capitalize()) for c in ["cases","deaths"]]
   ).withColumn("expected",lit(state_cases.count())).show()

state_cases.select(mean((state_cases["deaths"] <= state_cases["cases"]).cast("int")).alias("isValidDeathCount")).show()

windowSpec = Window.partitionBy("state").orderBy("date")
state_cases.orderBy("date") \
        .withColumn("date_diff",datediff(state_cases.date,lag("date",1).over(windowSpec))) \
        .select(mean("date_diff").alias("isValidTimeStep")).withColumn("expected",lit(1.0)).show()

state_cases.groupBy("state").agg(countDistinct("date").alias("distinct"),count(lit(1)).alias("expected")) \
           .withColumn("isUniqueDates",(col("distinct") == col("expected")).cast("int")).select(mean("isUniqueDates").alias("pctUnique")).withColumn("expected",lit(1.0)).show()

# COMMAND ----------

# MAGIC %md
# MAGIC For states, check how many are accounted.

# COMMAND ----------

state_cases.select("state").distinct().orderBy("state").display()

# COMMAND ----------

# MAGIC %md
# MAGIC Looks like all 50 states are there (verified by singing the "50 Nifty" song from elementary school) plus 6 territories like American Samoa and Puerto Rico. Leave those in for now.

# COMMAND ----------

counties[2020].filter(col("county") == lit("Hennepin")).limit(20).show()

# COMMAND ----------

for year in counties:
    df = counties[year]
    print(year)
    df.select([count(when(col(c).isNotNull() , c)).alias("notNull"+c.capitalize()) for c in df.columns]
       ).withColumn("expected",lit(df.count())).show()

    df.select([count(when(col(c) >= 0 , c)).alias("pos"+c.capitalize()) for c in ["cases","deaths"]]
       ).withColumn("expected",lit(df.count())).show()

    df.select(mean((df["deaths"] <= df["cases"]).cast("int")).alias("isValidDeathCount")).show()

    windowSpec = Window.partitionBy("county").orderBy("date")
    df.orderBy("date") \
            .withColumn("date_diff",datediff(df.date,lag("date",1).over(windowSpec))) \
            .select(mean((col("date_diff") == lit(1)).cast("int")).alias("isValidTimeStep")).withColumn("expected",lit(1.0)).show()
    
    df.groupBy("state","county").agg(countDistinct("date").alias("distinct"),count(lit(1)).alias("expected")) \
           .withColumn("isUniqueDates",(col("distinct") == col("expected")).cast("int")).select(mean("isUniqueDates").alias("pctUnique")).withColumn("expected",lit(1.0)).show()
    break

# COMMAND ----------

# MAGIC %md
# MAGIC Null counts in fips and deaths don't match the expected. Missing fips is ok since that column won't be used, but the missing death counts need to be identified.

# COMMAND ----------

for year in counties:
    print(year)
    df = counties[year]
    df.filter(df["deaths"].isNull()).limit(10).show()
    df.filter(df["deaths"].isNull()).select(df["state"]).distinct().show()
    

# COMMAND ----------

# MAGIC %md
# MAGIC Null death counts are only in Puerto Rico. That's a US territory, not a state, and the county-level data isn't needed, so those values get dropped.

# COMMAND ----------

for year in counties:
    df = counties[year]
    df = df.filter(df["state"] != "Puerto Rico")
    print(year)
    df.select([count(when(col(c).isNotNull() , c)).alias("notNull"+c.capitalize()) for c in df.columns]
       ).withColumn("expected",lit(df.count())).show()

    df.select([count(when(col(c) >= 0 , c)).alias("pos"+c.capitalize()) for c in ["cases","deaths"]]
       ).withColumn("expected",lit(df.count())).show()

    df.select(mean((df["deaths"] <= df["cases"]).cast("int")).alias("isValidDeathCount")).show()

    windowSpec = Window.partitionBy("county","state").orderBy("date")
    df.orderBy("date") \
            .withColumn("date_diff",datediff(df.date,lag("date",1).over(windowSpec))) \
            .select(mean((col("date_diff") == lit(1)).cast("int")).alias("isValidTimeStep")).withColumn("expected",lit(1.0)).show()
    
    df.groupBy("state","county").agg(countDistinct("date").alias("distinct"),count(lit(1)).alias("expected")) \
           .withColumn("isUniqueDates",(col("distinct") == col("expected")).cast("int")).select(mean("isUniqueDates").alias("pctUnique")).withColumn("expected",lit(1.0)).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Some rows have more deaths than cases. Filter to see what's going on.

# COMMAND ----------

df.filter(col("deaths") > col("cases")).limit(20).show()
df.filter((col("deaths") > col("cases")) & (col("county") != lit("Unknown"))).show()

# COMMAND ----------

# MAGIC %md
# MAGIC This is only a problem in Unknown counties, wich can be removed later.
# MAGIC 
# MAGIC Next, it seems like some time steps might be missing. Filter by date difference to see what's going on.

# COMMAND ----------

windowSpec = Window.partitionBy("county","state").orderBy("date")
withDiff = df.orderBy("date") \
        .withColumn("date_diff",datediff(df.date,lag("date",1).over(windowSpec)))
withDiff.filter(withDiff["date_diff"] != 1).show()
withDiff.filter(withDiff["date_diff"] != 1).filter(col("county") != lit("Unknown")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC The missing dates is only happening in Unknown county reports. These can be filtered out when finding average cases by county. 

# COMMAND ----------

df.filter(df["county"] != "Unknown").display()

# COMMAND ----------

df.filter((df["county"] != "Unknown") & (df["state"] == "Minnesota")).display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Results
# MAGIC - Completeness: 
# MAGIC   - State and Countrywide Data is fully complete
# MAGIC   - County data was missing death counts in Puerto Rico, so that "state" will be excluded from further analysis. "fips" values were missing for all unknown counties, but those aren't used for analysis or joining so that column can be dropped.
# MAGIC - Uniqueness: 
# MAGIC   - There are 56 unique "state" identifiers including all 50 US states and 6 territories.
# MAGIC   - No date is repeated for any unique state/county combination or in the countrywide DataFrame
# MAGIC - Validity: 
# MAGIC   - With each row being a running total of COVID cases and deaths, there should never be more deaths than cases. This does occur in many of the "Unknown" district entries, which will be filtered out.
# MAGIC   - All counts of cases and deaths are greater than or equal to 0.
# MAGIC   - When sorted by date, each row in the countrywide data is one day apart indicating a continuous time series.
# MAGIC   - When sorted by date and state, each row in the statewide data is one data apart indicating continuous time series.
# MAGIC   - County data has some gaps. These will be filled in with repeated values in the Data Preparation step.
# MAGIC   

# COMMAND ----------

# MAGIC %md
# MAGIC ### Station Data
# MAGIC Test for: 
# MAGIC - Completeness: no null values, coverage of all districts
# MAGIC - Uniqueness: how many unique values for each dataset?
# MAGIC   - Unique station #'s
# MAGIC   - Urban/Rural split
# MAGIC   - Number of counties
# MAGIC - Validity: does the data make sense?
# MAGIC   - Does each station belong to a county?
# MAGIC   - Does each county have a corresponding district?

# COMMAND ----------

current_stations.limit(100).display()
retired_stations.limit(100).display()

# COMMAND ----------

case_counties = None

# COMMAND ----------

print("Non Null counts")
current_stations.select([count(when(col(c).isNotNull() , c)).alias(c) for c in current_stations.columns]).withColumn("rowCount",lit(current_stations.count())).show()
retired_stations.select([count(when(col(c).isNotNull() , c)).alias(c) for c in retired_stations.columns]).withColumn("rowCount",lit(retired_stations.count())).show()
print("Unique value counts")
current_stations.select([countDistinct(c).alias(c) for c in current_stations.columns]).withColumn("rowCount",lit(current_stations.count())).show()
retired_stations.select([countDistinct(c).alias(c) for c in retired_stations.columns]).withColumn("rowCount",lit(retired_stations.count())).show()
print("Number of unique stations retired and active")
print(current_stations.select("Continuous Number").unionByName(retired_stations.select("Continuous Number")).distinct().count())
print("Urban / Rural Split")
current_stations.groupBy().pivot("Urban/Rural").count().show()

# all counties that have or have had a measurement station in them
station_counties = current_stations.select(col("County Name").alias("stationCounties")) \
                    .unionByName(retired_stations.select(col("COUNTY").alias("stationCounties"))).distinct()

for year in counties:
    if case_counties is None:
        case_counties = counties[year].filter(col("state") == "Minesota").select("county").distinct()
    
    case_counties = case_counties.union(counties[year].filter(col("state") == "Minnesota").select("county")).distinct()
    
print("Station List Counties Matched with COVID Case Counties")
station_counties = station_counties.join(case_counties,case_counties.county == station_counties.stationCounties,"outer").distinct().orderBy("stationCounties")
station_counties.display()

# COMMAND ----------

# MAGIC %md
# MAGIC There are 28 counties that reported COVID cases, but don't have any WIM stations. That should be ok as long as each county can be mapped to a district to aggregate and predict at the district level.

# COMMAND ----------

station_counties.columns

# COMMAND ----------

county_district.select([countDistinct(c).alias(c) for c in county_district.columns]).withColumn("rowCount",lit(county_district.count())).show()

county_district.select("MnDOT district(s)").distinct().show()

station_counties.join(county_district,"county","outer").distinct().orderBy("MnDOT district(s)").display()

# COMMAND ----------

# MAGIC %md
# MAGIC There are a few cases that can't be aligned to MnDOT districts here:
# MAGIC - Unknown: These will be dropped for per-district predictions.
# MAGIC - "St Louis"/"Saint Louis": There is a "St. Louis" assigned to District 1. "Saint Louis" is the closest match to other county names, so all instances of "St Louis" and "St. Louis" will be replaced and matched to district 1.
# MAGIC - Olmsted: a quick Google search reveals that this is in district 6
# MAGIC - Aitkin: There is an "Aikin" entry in the county_districts table. No "Aikin" county exists in Minnesota, and "Aikin" is assigned to District 1 which is Aitkin's actual district. "Aikin" can be replaced with "Aitkin" and connected to district 1.
# MAGIC - Pennington: There is a "Penningham" entry in the country_districts table, but again that is not a real county. The real Pennington county is ni District 2, which "Penningham" was assigned to. Like "Aikin", replace "Penningham" with "Pennington" and associate with district 2.
# MAGIC - Carlton: "Carlson" is entered in the county_districts table. This is another mispelling, and entries labeled "Carlson" should be replaced with Carlton and assigned to district 1. 
# MAGIC - Cass: This is assigned to district 2 and 3. Looking at the district map it is mostly in district 3, so it can be relabeled accordingly.
# MAGIC - Koochiching: This is assigned to district 1 and 2. The district map shows a majority in district 1, so it can be relabeled to district 1.
# MAGIC - Itasca: This is the same as Koochiching. Relabel to district 1.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Results
# MAGIC - Completeness:
# MAGIC   - All districts are covered and there are no null values in columns
# MAGIC   - Some counties that reported COVID cases do not have WIM stations, but those case counts will be aggregated to district counts later on.
# MAGIC - Uniqueness:
# MAGIC   - 62 different counties have WIM stations, covering all 8 MNDOT Districts.
# MAGIC   - There are 104 active stations and 119 retired stations for a total of 223 potential stations to view WIM data from.
# MAGIC   - The current stations show Urban / Rural classification, which is split 61 / 43. This likely means that a large portion of rural traffic is not measured, but relative changes in activity should still be discernable.
# MAGIC - Validity:
# MAGIC   - Some counties were assigned to multiple districts. There are few enough that they can be reassigned manually to the district covering a majority of the district.
# MAGIC   - Some county names were not immediately matched to a MNDOT District because of spelling variations or mispellings. Reviewing each case showed that all counties that are not "Unknown" can be matched to a real county in Minnesota. 
# MAGIC   - There are no null values for counties in the station tables, so each station is assigned to a county, and the county name validation confirms that the county for each station is a real county.

# COMMAND ----------

# MAGIC %md
# MAGIC ### WIM Volume Data
# MAGIC Test for:
# MAGIC - Completeness:
# MAGIC   - No null values
# MAGIC   - Full time series for each station
# MAGIC - Uniqueness:
# MAGIC   - No repeat entries for station, dir, lane, date groupings
# MAGIC - Validity:
# MAGIC   - No negative values in columns
# MAGIC   - Each station has a corresponding value in either current or retired station tables

# COMMAND ----------

for year in wim_volume:
    wim_volume[year].limit(20).show()
    break

# COMMAND ----------

print("Null value check")
for year in wim_volume:
    tmp = wim_volume[year]
    print(year, ":", tmp.count(), "Rows")
    tmp.select([count(when(col(c).isNull() , c)).alias(c) for c in tmp.columns]).withColumn("rowCount",lit(tmp.count())).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Note that the number of rows for yearly WIM data doubled from 2020 to 2021. Did the number of stations increase?

# COMMAND ----------

for year in wim_volume:
    tmp = wim_volume[year]
    print(year, "unique stations:",tmp.select("station_id").distinct().count())

# COMMAND ----------

# MAGIC %md
# MAGIC Yes, there are more than twice as many stations for 2021 than 2020. 

# COMMAND ----------

# check if each station across all sheets is in the active or retired list
all_stations = current_stations.select("Continuous Number").unionByName(retired_stations.select("Continuous Number")).distinct()

for year in wim_volume:
    tmp = wim_volume[year]
    print(year)
    tmp.select("station_id").distinct().join(all_stations,col("station_id") == col("Continuous Number"),"left").select(count(when(col("station_id").isNull(),"station_id")).alias("unmatched")).show()

# COMMAND ----------

print("Unique station, direction, lane, date entries")
for year in wim_volume:
    tmp = wim_volume[year]
    print(year)
    print("Unique:",tmp.select("station_id","dir_of_travel","lane_of_travel","date").distinct().count(),"| Expected:",tmp.count())

# COMMAND ----------

print("No Negative values check")
for year in wim_volume:
    tmp = wim_volume[year].drop("date")
    print(year, ":", tmp.count(), "Rows")
    tmp.select([count(when(col(c) < 0 , c)).alias(c) for c in tmp.columns]).withColumn("rowCount",lit(tmp.count())).show()

# COMMAND ----------

print("Time series check")
windowSpec = Window.partitionBy("station_id","dir_of_travel","lane_of_travel").orderBy("date")
for year in wim_volume:
    tmp = wim_volume[year]
    print(year)
    tmp.orderBy("date") \
        .withColumn("date_diff",datediff(tmp.date,lag("date",1).over(windowSpec))) \
        .select(mean((col("date_diff") == lit(1)).cast("int")).alias("isValidTimeStep")).withColumn("expected",lit(1.0)).show()


# COMMAND ----------

# MAGIC %md
# MAGIC Some stations like 29 seem to have gaps in recordings. Those can be filled in with repeat values after aggregating by date. 

# COMMAND ----------

wim_volume[2021].limit(10).show()

# COMMAND ----------

# MAGIC %md
# MAGIC The date format changed from 2020 to 2021. That can be transformed quickly and the time series check re-run.

# COMMAND ----------

windowSpec = Window.partitionBy("station_id","dir_of_travel","lane_of_travel").orderBy("new_date")
for year in [2021,2022]:
    tmp = wim_volume[year]
    print(year)
    tmp.orderBy("date") \
        .withColumn("new_date",to_date(col("date"),"M/D/yyyy")) \
        .withColumn("date_diff",datediff(col("new_date"),lag("new_date",1).over(windowSpec))) \
        .select(mean((col("date_diff") == lit(1)).cast("int")).alias("isValidTimeStep")).withColumn("expected",lit(1.0)).show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Results
# MAGIC - Completeness:
# MAGIC   - There are no null values in the volume data.
# MAGIC   - Some stations have gaps in their time series, which will be filled in with an average value of several days before and after the gap.
# MAGIC - Uniqueness: 
# MAGIC   - There are no repeated combinations of station, direction, lane, and date.
# MAGIC - Validity:
# MAGIC   - The date format changed from 2020 to 2021 and 2022. That will need to be fixed when aggregating.
# MAGIC   - No numeric values were negative.
# MAGIC   - Each station matches with one listed in the current or retired station list tables.

# COMMAND ----------

county_district.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Conditioning
# MAGIC The available data for ATR WIM stations (current and retired), COVID cases (county, state, and country), and traffic volume has been examined and several necessary changes identified. In order of discovery those are:
# MAGIC 1. Remove Puerto Rico from county level COVID case reports because of null death count values
# MAGIC 2. Remove Unknown counties from county level COVID case reports because of non-increasing case/death counts and incomplete time series
# MAGIC 3. Minnesota Counties:
# MAGIC     - Rename all instances of "St Louis"/"Saint Louis"/"St. Louis" to "Saint Louis" in COVID case data and rename "St. Louis" to "Saint Louis" in county_district view. (Use again after combining retired and current stations)
# MAGIC     - Remove mysterious whitespace character from Olmsted
# MAGIC     - Replace "Aikin" in county_districts with "Aitkin"
# MAGIC     - Replace "Penningham" in county_districts with "Pennington"
# MAGIC     - Replace "Carlson" with "Carlton" in county_districts
# MAGIC     - Assign Cass county to district 3 only.
# MAGIC     - Assign Koochiching county to district 1 only.
# MAGIC     - Assign Itasca county to district 1 only.
# MAGIC 4. Reformat dates for wim_volume data in 2021 and 2022

# COMMAND ----------

# Remove Puerto Rico and Unknown counties from county level COVID reports
for year in counties:
    tmp = counties[year]
    tmp = tmp.filter(col("state") != "Puerto Rico")
    #verify no more Null death counts
    tmp.select(count(when(col("deaths").isNull(), "deaths")).alias("nullDeathCount")).show()
    tmp = tmp.filter(col("county") != "Unknown")
    
    counties[year] = tmp

# COMMAND ----------

mn_counties = spark.createDataFrame(spark.sparkContext.emptyRDD(),schema=counties_covid_schema)

for year in counties:
    mn_counties = mn_counties.unionByName(counties[year].filter(col("state") == "Minnesota"))
mn_counties.orderBy("county","date").limit(10).show()

# COMMAND ----------

# replace St. Louis with Saint Louis
mn_counties = mn_counties.withColumn('county', regexp_replace('county', 'St. Louis', 'Saint Louis'))

# COMMAND ----------

def fix_spelling(str):
    corrections = {'Aikin':'Aitkin',
                   'Carlson':'Carlton',
                  'Penningham':'Pennington',
                  'St Louis':'Saint Louis',
                  'St. Louis':'Saint Louis'
                  }
    if str.startswith('Olmsted'):
        return 'Olmsted'
    if str in corrections:
        return corrections[str]
    else:
        return str
    
def remap_district(str):
    corrections = {'Cass':'3',
                   'Koochiching':'1',
                   'Itasca':'1'}
    if str in corrections:
        return corrections[str]
    else:
        return '0'
    
fix_spellingUDF = udf(lambda s: fix_spelling(s),StringType())
remap_districtUDF = udf(lambda s: remap_district(s),StringType())
    
county_district = county_district.withColumn("county", fix_spellingUDF(col("county"))) \
                                 .withColumn("MnDOT district(s)", when(remap_districtUDF(col("county")) != "0", remap_districtUDF(col("county"))) \
                                                                    .otherwise(col("MnDOT district(s)")))
county_district.display()

# COMMAND ----------

# reformat dates for wim_volume in 2021 and 2022
windowSpec = Window.partitionBy("station_id","dir_of_travel","lane_of_travel").orderBy("new_date")
test = {}
for year in [2021,2022]:
    tmp = wim_volume[year]
    tmp = tmp.orderBy("date") \
        .withColumn("date", to_date(col("date"),"M/d/yyyy"))
    wim_volume[year] = tmp

# COMMAND ----------

# MAGIC %md
# MAGIC ### Aggregation
# MAGIC Now that data has been prepped, it needs to be aggregated so that there are singular datasets for WIM volumes by date, county, and district. The steps to do that are:
# MAGIC 1. Combine all years of WIM volume into a single DataFrame
# MAGIC 2. Combine all lanes and directions for each station and date so there is one row per station per date
# MAGIC 3. Sum across all hours of each day for each station and date
# MAGIC 
# MAGIC ### Joining
# MAGIC 1. Join current and retired stations and apply fix_spelling function to make joining with county_district frame possible
# MAGIC 2. Join each station / date row to the appropriate county in the joined station frame
# MAGIC 3. Join each station / date / county row to the appropriate district in the county_distric frame

# COMMAND ----------

# combine all years of WIM volume
all_wim = spark.createDataFrame(spark.sparkContext.emptyRDD(),schema=wim_volume_schema)

for year in wim_volume:
    all_wim = all_wim.unionByName(wim_volume[year])

# COMMAND ----------

all_wim.limit(10).show()

# COMMAND ----------

# Aggregate by station
exprs = {str(i): "sum" for i in range(1,25)}
all_wim = all_wim.groupBy("station_id","date").agg(exprs)
all_wim.filter(col("station_id") == 51).limit(10).show()

# COMMAND ----------

# Aggregate days
aggregated_wim = all_wim.withColumn("totalVolume", reduce(add, [col("sum("+str(i)+")") for i in range(1,25)])).select("station_id","date","totalVolume")
aggregated_wim.limit(10).show()

# COMMAND ----------

aggregated_wim.filter(col("station_id").isNull()).display()

# COMMAND ----------

# Join current and retired stations only keeping one row for each id
# To keep one row, order by Status on retired stations, which will filter by retired date
# Keep only the retired stations with ids not in use by current stations, and union the two
filtered_retired_stations = retired_stations.orderBy("Status").dropDuplicates(["Continuous Number"]) \
                            .join(retired_stations.select("Continuous Number").subtract(current_stations.select("Continuous Number")),"Continuous Number","leftouter")
all_stations = current_stations.select("Continuous Number",col("County Name").alias("County")) \
                .unionByName(filtered_retired_stations.select("Continuous Number","County")) \
                .withColumn("County", fix_spellingUDF(col("County")))
all_stations.display()

# COMMAND ----------

county_district.display()

# COMMAND ----------

# join station number and county to district
station_county_district = all_stations.join(county_district,all_stations["County"] == county_district["county"],"left").select("Continuous Number",county_district["County"],"MnDOT district(s)")
station_county_district.limit(10).show()

# COMMAND ----------

# MAGIC %md
# MAGIC At this point there are some stations that have volume data that are not listed among the current or retired stations. Some of the stations listed as active also do not show up in any of the traffic volume reports. Unfortunately the missing data is not readily available, so an inner join will be used keeping only rows that have a station id, county, district, date, and volume data.

# COMMAND ----------

# join aggregated WIM to county
wim_joined = station_county_district.join(aggregated_wim,col("Continuous Number") == col("station_id"),"inner")
wim_joined.limit(10).show()

# COMMAND ----------

# confirm there are no null values at this point
wim_joined.select([count(when(col(c).isNull() , c)).alias(c) for c in wim_joined.columns]).show()

# COMMAND ----------

# rename and drop unnecessary column
wim_joined = wim_joined.drop("Continuous Number").withColumnRenamed("MnDOT district(s)","district").withColumnRenamed("County","county")
print(wim_joined.columns)

# COMMAND ----------

print("Total rows of Traffic Volume data:",wim_joined.count())

# COMMAND ----------

# MAGIC %md
# MAGIC #### State Volume Data
# MAGIC A much simpler task is to aggregate the county data per date into a statewide Traffic Volume total.

# COMMAND ----------

state_wim = wim_joined.groupBy("date").agg(colsum(col("totalVolume")).alias("totalVolume")).select("date", "totalVolume")
state_wim.limit(20).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### MN County COVID Data
# MAGIC Back to the `mn_counties` DataFrame: now that all of the county and district mappings are complete that can be joined and written out to a table.

# COMMAND ----------

mn_counties_districts_covid = mn_counties.join(county_district,mn_counties["county"] == county_district["County"],"left").drop(county_district["county"]).drop("fips").withColumnRenamed("MnDOT district(s)","district")
print("Check Null Counts")
mn_counties_districts_covid.select([count(when(col(c).isNull() , c)).alias(c) for c in mn_counties_districts_covid.columns]).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## All County COVID Data
# MAGIC Lastly, the COVID case and death totals by county that were cleaned up earlier can be aggregated into average counts by county for each day. There is too much data to look at each county individually, but it may be useful to compare counties in Minnesota to the average of all counties in the US.

# COMMAND ----------

average_county_covid_schema = StructType([StructField("date",StringType(),True),
                                          StructField("cases",FloatType(),True),
                                          StructField("deaths",FloatType(),True)])

average_county_covid = spark.createDataFrame(spark.sparkContext.emptyRDD(),schema=average_county_covid_schema)

for year in counties:
    tmp = counties[year].drop('fips')
    average_county_covid = average_county_covid.unionByName(tmp.groupBy("date").agg(mean(col("cases")).alias("cases"), mean(col("deaths")).alias("deaths")).select("date","cases","deaths"))

average_county_covid.limit(20).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export
# MAGIC At the end of this notebook, the data sources available for modeling are:
# MAGIC * Minnesota Traffic Volume by Date / WIM Station (wim_joined, columns={Date, County, District, StationID, TotalVolume})
# MAGIC * Minnesota Total Traffic Volume by Date (state_wim, columns={Date,TotalVolume})
# MAGIC * Minnesota COVID cases / deaths by county by date (mn_counties_districts_covid, columns={Date, County, State, District, Cases, Deaths})
# MAGIC * Average COVID cases per county by date (average_county_covid, columns={date,cases,deaths})
# MAGIC * Total US cases by date (unmodified us_cases, columns={date,cases,deaths})
# MAGIC * Total cases per state by date (unmodified state_cases, columns={date,state,fips,cases,deaths})
# MAGIC 
# MAGIC Each will be written to a parquet table for use in the Modeling notebook.

# COMMAND ----------

# check schemas
print(wim_joined.schema)
print(state_wim.schema)
print(mn_counties_districts_covid.schema)
print(average_county_covid.schema)
print(us_cases.schema)
print(state_cases.schema)

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists county_wim_2017_2022;
# MAGIC drop table if exists state_wim_2017_2022;
# MAGIC drop table if exists mn_county_covid_2020_2022;
# MAGIC drop table if exists average_county_covid_2020_2022;
# MAGIC drop table if exists total_us_covid_2020_2022;
# MAGIC drop table if exists total_state_covid_2020_2022;
# MAGIC drop table if exists mn_covid_timeline;

# COMMAND ----------

dbutils.fs.rm("dbfs:/user/hive/warehouse/", True)
wim_joined.write.format("parquet").saveAsTable("county_wim_2017_2022")
state_wim.write.format("parquet").saveAsTable("state_wim_2017_2022")
mn_counties_districts_covid.write.format("parquet").saveAsTable("mn_county_covid_2020_2022")
average_county_covid.write.format("parquet").saveAsTable("average_county_covid_2020_2022")
us_cases.write.format("parquet").saveAsTable("total_us_covid_2020_2022")
state_cases.write.format("parquet").saveAsTable("total_state_covid_2020_2022")
timeline.write.format("parquet").saveAsTable("mn_covid_timeline")

# COMMAND ----------

# MAGIC %sql
# MAGIC show tables;
