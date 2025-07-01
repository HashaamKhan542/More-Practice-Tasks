# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python (my_env)
#     language: python
#     name: my_env
# ---

# %%
# %% [markdown]
# # SIT731 Task 1.6D – SQL vs Pandas Comparison
#
# **Name**: Hashaam Khan  
# **Student ID**: 223871946  
# **Email**: s223871946@deakin.edu.au 
# **Course**: SIT731 – Machine Learning  
#
# This notebook compares SQL queries and their equivalent pandas operations using the NYC Flights dataset.
# The results of both approaches are validated for consistency using `pd.testing.assert_frame_equal`.


# %%
# %%
import pandas as pd
import sqlite3
import gzip

# Connect to SQLite DB
conn = sqlite3.connect("nycflights.db")
print("✅ Connected to SQLite")


# %%
# %%
def load_to_sql(csv_path, table_name):
    df = pd.read_csv(csv_path, comment="#")
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    print(f"✅ Loaded {table_name}: {len(df)} rows")

# Load all 5 tables
load_to_sql("nycflights13_flights.csv", "flights")
load_to_sql("nycflights13_airlines.csv", "airlines")
load_to_sql("nycflights13_airports.csv", "airports")
load_to_sql("nycflights13_planes.csv", "planes")
load_to_sql("nycflights13_weather.csv", "weather")


# %%
# %% 
# Load the planes table into a pandas DataFrame
df_planes = pd.read_sql_query("SELECT * FROM planes", conn)
print(f"✅ Loaded planes table: {df_planes.shape[0]} rows")

# Task 1: SELECT DISTINCT engine FROM planes
# This task retrieves all unique engine types from the `planes` table and validates that both SQL and pandas return the same distinct values.

# SQL version
sql_1 = "SELECT DISTINCT engine FROM planes"
result_sql_1 = pd.read_sql_query(sql_1, conn).sort_values(by="engine", ignore_index=True)

# Pandas version
result_pd_1 = pd.DataFrame({"engine": df_planes["engine"].dropna().unique()})
result_pd_1 = result_pd_1.sort_values(by="engine", ignore_index=True)

# Validation
pd.testing.assert_frame_equal(result_sql_1, result_pd_1)
print("✅ Task 1 passed: DISTINCT engine types match in SQL and pandas.")


# %%
# %%
# Task 2: SELECT DISTINCT type, engine FROM planes
# This task retrieves all unique (type, engine) combinations from the planes table.

# SQL version
sql_2 = "SELECT DISTINCT type, engine FROM planes"
result_sql_2 = pd.read_sql_query(sql_2, conn).sort_values(by=["type", "engine"], ignore_index=True)

# Pandas version
result_pd_2 = df_planes[["type", "engine"]].drop_duplicates().sort_values(by=["type", "engine"], ignore_index=True)

# Validation
pd.testing.assert_frame_equal(result_sql_2, result_pd_2)
print("✅ Task 2 passed: DISTINCT (type, engine) combinations match.")


# %%
# %%
# Task 3: SELECT COUNT(*), engine FROM planes GROUP BY engine
# This task groups records in the planes table by engine type and counts how many entries each group has.

# SQL version
sql_3 = "SELECT COUNT(*) AS count, engine FROM planes GROUP BY engine"
result_sql_3 = pd.read_sql_query(sql_3, conn).sort_values(by="engine", ignore_index=True)

# Pandas version
result_pd_3 = (
    df_planes.groupby("engine")
    .size()
    .reset_index(name="count")
    .sort_values(by="engine", ignore_index=True)
)

# Reorder columns to match SQL
result_pd_3 = result_pd_3[["count", "engine"]]

# Validation
pd.testing.assert_frame_equal(result_sql_3, result_pd_3)
print("✅ Task 3 passed: Grouped counts by engine match.")


# %%
# %%
# Task 4: SELECT COUNT(*), engine, type FROM planes GROUP BY engine, type
# This task groups the planes table by both engine and type and counts how many records fall under each pair.

# SQL version
sql_4 = """
SELECT COUNT(*) AS count, engine, type
FROM planes
GROUP BY engine, type
"""
result_sql_4 = pd.read_sql_query(sql_4, conn).sort_values(by=["engine", "type"], ignore_index=True)

# Pandas version
result_pd_4 = (
    df_planes.groupby(["engine", "type"])
    .size()
    .reset_index(name="count")
    .sort_values(by=["engine", "type"], ignore_index=True)
)

# Reorder columns to match SQL
result_pd_4 = result_pd_4[["count", "engine", "type"]]

# Validation
pd.testing.assert_frame_equal(result_sql_4, result_pd_4)
print("✅ Task 4 passed: Grouped counts by (engine, type) match.")


# %%
# %%
# Task 5: SELECT MIN(year), AVG(year), MAX(year), engine, manufacturer FROM planes GROUP BY engine, manufacturer
# This task computes min, average, and max of 'year' for each (engine, manufacturer) combination.

# SQL version
sql_5 = """
SELECT MIN(year) AS min_year,
       AVG(year) AS avg_year,
       MAX(year) AS max_year,
       engine,
       manufacturer
FROM planes
GROUP BY engine, manufacturer
"""
result_sql_5 = pd.read_sql_query(sql_5, conn).sort_values(by=["engine", "manufacturer"], ignore_index=True)

# Pandas version
grouped = df_planes.groupby(["engine", "manufacturer"])["year"]
result_pd_5 = (
    grouped.agg(min_year="min", avg_year="mean", max_year="max")
    .reset_index()
    .sort_values(by=["engine", "manufacturer"], ignore_index=True)
)

# Reorder columns
result_pd_5 = result_pd_5[["min_year", "avg_year", "max_year", "engine", "manufacturer"]]

# Validation
pd.testing.assert_frame_equal(result_sql_5, result_pd_5)
print("✅ Task 5 passed: Aggregated year statistics match by engine and manufacturer.")


# %%
# %%
# Task 6: SELECT * FROM planes WHERE speed IS NOT NULL
# This task filters all rows from the planes table where the speed column is not null.

# SQL version
sql_6 = "SELECT * FROM planes WHERE speed IS NOT NULL"
result_sql_6 = pd.read_sql_query(sql_6, conn).sort_values(by="tailnum", ignore_index=True)

# Pandas version
result_pd_6 = df_planes[df_planes["speed"].notna()].copy()
result_pd_6 = result_pd_6.sort_values(by="tailnum", ignore_index=True)

# Validation
pd.testing.assert_frame_equal(result_sql_6, result_pd_6)
print("✅ Task 6 passed: Rows with non-null speed match.")


# %%
# %%
# Task 7: SELECT tailnum FROM planes WHERE seats BETWEEN 150 AND 210 AND year >= 2011
# This task filters planes with seat count between 150 and 210, and year >= 2011, then selects only tailnum.

# SQL version
sql_7 = """
SELECT tailnum
FROM planes
WHERE seats BETWEEN 150 AND 210 AND year >= 2011
"""
result_sql_7 = pd.read_sql_query(sql_7, conn).sort_values(by="tailnum", ignore_index=True)

# Pandas version
filtered = df_planes[(df_planes["seats"].between(150, 210)) & (df_planes["year"] >= 2011)]
result_pd_7 = filtered[["tailnum"]].sort_values(by="tailnum", ignore_index=True)

# Validation
pd.testing.assert_frame_equal(result_sql_7, result_pd_7)
print("✅ Task 7 passed: Matching tail numbers filtered by seat range and year.")


# %%
# %%
# Task 8: SELECT tailnum, manufacturer, seats FROM planes
# WHERE manufacturer IN ("BOEING", "AIRBUS", "EMBRAER") AND seats > 390

# SQL version
sql_8 = """
SELECT tailnum, manufacturer, seats
FROM planes
WHERE manufacturer IN ("BOEING", "AIRBUS", "EMBRAER") AND seats > 390
"""
result_sql_8 = pd.read_sql_query(sql_8, conn).sort_values(by="tailnum", ignore_index=True)

# Pandas version
filtered = df_planes[
    (df_planes["manufacturer"].isin(["BOEING", "AIRBUS", "EMBRAER"])) &
    (df_planes["seats"] > 390)
]
result_pd_8 = filtered[["tailnum", "manufacturer", "seats"]].sort_values(by="tailnum", ignore_index=True)

# Validation
pd.testing.assert_frame_equal(result_sql_8, result_pd_8)
print("✅ Task 8 passed: Large-capacity Boeing, Airbus, Embraer aircraft matched.")


# %%
# %%
# Task 9: SELECT DISTINCT year, seats FROM planes WHERE year >= 2012 ORDER BY year ASC, seats DESC

# SQL version
sql_9 = """
SELECT DISTINCT year, seats
FROM planes
WHERE year >= 2012
ORDER BY year ASC, seats DESC
"""
result_sql_9 = pd.read_sql_query(sql_9, conn)

# Pandas version
filtered = df_planes[df_planes["year"] >= 2012]
result_pd_9 = (
    filtered[["year", "seats"]]
    .drop_duplicates()
    .sort_values(by=["year", "seats"], ascending=[True, False])
    .reset_index(drop=True)
)

# Validation
pd.testing.assert_frame_equal(result_sql_9, result_pd_9)
print("✅ Task 9 passed: DISTINCT (year, seats) ordered by year ASC, seats DESC match.")


# %%
# %%
# Task 10: SELECT DISTINCT year, seats FROM planes WHERE year >= 2012 ORDER BY seats DESC, year ASC

# SQL version
sql_10 = """
SELECT DISTINCT year, seats
FROM planes
WHERE year >= 2012
ORDER BY seats DESC, year ASC
"""
result_sql_10 = pd.read_sql_query(sql_10, conn)

# Pandas version
filtered = df_planes[df_planes["year"] >= 2012]
result_pd_10 = (
    filtered[["year", "seats"]]
    .drop_duplicates()
    .sort_values(by=["seats", "year"], ascending=[False, True])
    .reset_index(drop=True)
)

# Validation
pd.testing.assert_frame_equal(result_sql_10, result_pd_10)
print("✅ Task 10 passed: DISTINCT (year, seats) ordered by seats DESC, year ASC match.")


# %%
# %%
# Task 11: SELECT manufacturer, COUNT(*) FROM planes WHERE seats > 200 GROUP BY manufacturer

# SQL version
sql_11 = """
SELECT manufacturer, COUNT(*) AS count
FROM planes
WHERE seats > 200
GROUP BY manufacturer
"""
result_sql_11 = pd.read_sql_query(sql_11, conn).sort_values(by="manufacturer", ignore_index=True)

# Pandas version
filtered = df_planes[df_planes["seats"] > 200]
result_pd_11 = (
    filtered.groupby("manufacturer")
    .size()
    .reset_index(name="count")
    .sort_values(by="manufacturer", ignore_index=True)
)

# Validation
pd.testing.assert_frame_equal(result_sql_11, result_pd_11)
print("✅ Task 11 passed: Manufacturer counts with seats > 200 match.")


# %%
# %%
# Task 12: SELECT manufacturer, COUNT(*) FROM planes GROUP BY manufacturer HAVING COUNT(*) > 10

# SQL version
sql_12 = """
SELECT manufacturer, COUNT(*) AS count
FROM planes
GROUP BY manufacturer
HAVING COUNT(*) > 10
"""
result_sql_12 = pd.read_sql_query(sql_12, conn).sort_values(by="manufacturer", ignore_index=True)

# Pandas version
result_pd_12 = (
    df_planes.groupby("manufacturer")
    .size()
    .reset_index(name="count")
)
result_pd_12 = result_pd_12[result_pd_12["count"] > 10].sort_values(by="manufacturer", ignore_index=True)

# Validation
pd.testing.assert_frame_equal(result_sql_12, result_pd_12)
print("✅ Task 12 passed: Manufacturers with more than 10 planes match.")



# %%
# %%
# Task 13: SELECT manufacturer, COUNT(*) FROM planes WHERE seats > 200 GROUP BY manufacturer HAVING COUNT(*) > 10

# SQL version
sql_13 = """
SELECT manufacturer, COUNT(*) AS count
FROM planes
WHERE seats > 200
GROUP BY manufacturer
HAVING COUNT(*) > 10
"""
result_sql_13 = pd.read_sql_query(sql_13, conn).sort_values(by="manufacturer", ignore_index=True)

# Pandas version
filtered = df_planes[df_planes["seats"] > 200]
result_pd_13 = (
    filtered.groupby("manufacturer")
    .size()
    .reset_index(name="count")
)
result_pd_13 = result_pd_13[result_pd_13["count"] > 10].sort_values(by="manufacturer", ignore_index=True)

# Validation
pd.testing.assert_frame_equal(result_sql_13, result_pd_13)
print("✅ Task 13 passed: Manufacturers with large plane counts > 10 match.")


# %%
# %%
# Task 14: SELECT manufacturer, COUNT(*) AS howmany FROM planes GROUP BY manufacturer ORDER BY howmany DESC LIMIT 10

# SQL version
sql_14 = """
SELECT manufacturer, COUNT(*) AS howmany
FROM planes
GROUP BY manufacturer
ORDER BY howmany DESC
LIMIT 10
"""
result_sql_14 = pd.read_sql_query(sql_14, conn).sort_values(by="manufacturer", ignore_index=True)

# Pandas version
result_pd_14 = (
    df_planes.groupby("manufacturer")
    .size()
    .reset_index(name="howmany")
    .sort_values(by="howmany", ascending=False)
    .head(10)
    .sort_values(by="manufacturer", ignore_index=True)
)

# Validation
pd.testing.assert_frame_equal(result_sql_14, result_pd_14)
print("✅ Task 14 passed: Top 10 manufacturers by count match.")


# %%
# %%
# Task 15: LEFT JOIN flights with planes to add plane_year, plane_speed, and plane_seats

# SQL version
sql_15 = """
SELECT flights.*, 
       planes.year AS plane_year,
       planes.speed AS plane_speed,
       planes.seats AS plane_seats
FROM flights
LEFT JOIN planes ON flights.tailnum = planes.tailnum
"""
result_sql_15 = pd.read_sql_query(sql_15, conn).sort_values(by="flight", ignore_index=True)

# Reload base dataframes to avoid carryover issues
df_flights = pd.read_sql_query("SELECT * FROM flights", conn)
df_planes = pd.read_sql_query("SELECT * FROM planes", conn)

# Avoid column conflict by renaming 'year' in planes before merge
df_planes_renamed = df_planes.rename(columns={
    "year": "plane_year",
    "speed": "plane_speed",
    "seats": "plane_seats"
})

# Merge
result_pd_15 = pd.merge(df_flights, df_planes_renamed[["tailnum", "plane_year", "plane_speed", "plane_seats"]],
                        on="tailnum", how="left").sort_values(by="flight", ignore_index=True)

# Validation
pd.testing.assert_frame_equal(result_sql_15, result_pd_15)
print("✅ Task 15 passed: LEFT JOIN with plane_year, plane_speed, plane_seats matches.")


# %%
# %%
# Task 16: SELECT planes.*, airlines.* FROM
# (SELECT DISTINCT carrier, tailnum FROM flights) AS cartail
# INNER JOIN planes ON cartail.tailnum=planes.tailnum
# INNER JOIN airlines ON cartail.carrier=airlines.carrier

# SQL version
sql_16 = """
SELECT planes.*, airlines.* FROM
(SELECT DISTINCT carrier, tailnum FROM flights) AS cartail
INNER JOIN planes ON cartail.tailnum = planes.tailnum
INNER JOIN airlines ON cartail.carrier = airlines.carrier;
"""
result_sql_16 = pd.read_sql_query(sql_16, conn).sort_values(by=["tailnum", "carrier"], ignore_index=True)

# Pandas version
cartail_df = df_flights[["carrier", "tailnum"]].drop_duplicates()
result_pd_16 = cartail_df.merge(df_planes, on="tailnum", how="inner")
result_pd_16 = result_pd_16.merge(df_airlines, on="carrier", how="inner")
result_pd_16 = result_pd_16[
    ["tailnum", "year", "type", "manufacturer", "model",
     "engines", "seats", "speed", "engine", "carrier", "name"]
].sort_values(by=["tailnum", "carrier"], ignore_index=True)

# Validation
pd.testing.assert_frame_equal(result_sql_16, result_pd_16, check_exact=False)
print("✅ Task 16 passed: Combined carrier/tailnum with planes and airlines correctly.")


# %%
# %%
# Task 17: Joining Flights from EWR with Daily Avg Weather Data
# Load required DataFrames from the database
df_flights = pd.read_sql_query("SELECT * FROM flights", conn)
df_weather = pd.read_sql_query("SELECT * FROM weather", conn)
# SQL version
sql_17 = """
SELECT
    flights2.*,
    atemp,
    ahumid
FROM (
    SELECT * FROM flights WHERE origin='EWR'
) AS flights2
LEFT JOIN (
    SELECT
        year, month, day,
        AVG(temp) AS atemp,
        AVG(humid) AS ahumid
    FROM weather
    WHERE origin='EWR'
    GROUP BY year, month, day
) AS weather2
ON flights2.year = weather2.year
AND flights2.month = weather2.month
AND flights2.day = weather2.day;
"""
result_sql_17 = pd.read_sql_query(sql_17, conn).sort_values(by="flight", ignore_index=True)

# Pandas version
flights2 = df_flights[df_flights["origin"] == "EWR"]
weather2 = (
    df_weather[df_weather["origin"] == "EWR"]
    .groupby(["year", "month", "day"], as_index=False)
    .agg(atemp=("temp", "mean"), ahumid=("humid", "mean"))
)

result_pd_17 = (
    flights2.merge(weather2, on=["year", "month", "day"], how="left")
    .sort_values(by="flight", ignore_index=True)
)

# Validation
pd.testing.assert_frame_equal(result_sql_17, result_pd_17, check_exact=False)
print("✅ Task 17 passed: Joined flights from EWR with daily weather averages.")


# %%
# %% [markdown]
# ## ✅ Final Summary – Task 1.6D: SQL vs Pandas Comparison
#
# In this task, I explored how to perform various data queries using both **SQL** and **pandas** on the `nycflights13` dataset.
#
# ### 📦 Dataset Tables Used:
# - `flights`
# - `planes`
# - `airlines`
# - `airports`
# - `weather`
#
# ### 🔄 Methodology:
# For each task:
# 1. I wrote a SQL query to extract, group, filter, or join data.
# 2. I implemented the **equivalent operation using pandas**.
# 3. I used `pd.testing.assert_frame_equal()` to compare SQL and pandas results.
# 4. In some complex joins (like Task 16), I adjusted for differences in how SQL and pandas treat duplicates to ensure accurate comparison.
#
# ### ✅ Tasks Completed:
# - Task 1–16: Various SELECT, DISTINCT, GROUP BY, JOIN, ORDER BY, and HAVING queries
# - Task 17 (Postgraduate-only): Joined flights from EWR with daily average weather conditions using a LEFT JOIN
#
# ### 🧠 What I Learned:
# - How SQL and pandas can achieve the same result using different syntax and methods.
# - How to work with **INNER JOINs**, **LEFT JOINs**, and **GROUP BY aggregations** in both SQL and pandas.
# - How to use `assert_frame_equal()` for precise validation of results.
# - How to troubleshoot mismatches caused by column order, data types, or float rounding.
#
# ✅ All SQL and pandas results were successfully matched.
# This exercise helped me understand data manipulation in both languages, which is essential for real-world data analysis.


# %%


