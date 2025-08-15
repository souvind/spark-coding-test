# Databricks notebook source
from pyspark.sql.types import *
from pyspark.sql.functions import *

# COMMAND ----------

myschema = StructType(
    [
        StructField("order_id", IntegerType(), True),
        StructField("customer_name", StringType(), True),
        StructField("age", IntegerType(), True),
        StructField("city", StringType(), True),
        StructField("product", StringType(), True),
        StructField("category", StringType(), True),
        StructField("price", DoubleType(), True),
        StructField("quantity", IntegerType(), True),
        StructField("order_date", DateType(), True),
        StructField("_metadata", StringType(), True)
    ]
)

# COMMAND ----------

df = spark.read \
    .format("csv") \
    .option("header",True)\
    .schema(myschema)\
    .load("/Volumes/test_catalog/test_schema/test_volume/test.csv")
df.display()


# COMMAND ----------

# MAGIC %md
# MAGIC fetch first 5 records

# COMMAND ----------

df.limit(5).display()

# COMMAND ----------

# MAGIC %md
# MAGIC Print the schema of the DataFrame and list all column data types

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC Select only specific columns from the DataFrame and rename them

# COMMAND ----------

rename_map = {
    "customer_name": "name",
    "order_id": "id"
}

for old_name, new_name in rename_map.items():
    df = df.withColumnRenamed(old_name, new_name)

df.display()


# COMMAND ----------

# MAGIC %md
# MAGIC Filter rows where price is greater than 500 and category is "Electronics"

# COMMAND ----------

df_new = df.filter((col("price")>500.00) & (col("category") == "Electronics"))
df_new.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Calculate total sales per category

# COMMAND ----------

df_group = df.groupBy(col("category")).agg(sum(col("price")*col("quantity")).alias("total_sum"))
df_group.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Count the number of orders per city and sort by the highest order count

# COMMAND ----------

df_count = df.groupBy("city").agg(count(col("quantity")).alias("Count"))
df_count.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Find the maximum, minimum, and average price for each category

# COMMAND ----------

df_diff = df.groupBy(col("category")).agg(max(col("price"))).alias("Maximum")
df_diff.display()

# COMMAND ----------

from pyspark.sql import functions as F

df_stats = df.groupBy("category").agg(
    F.max("price").alias("Maximum_Price"),
    F.min("price").alias("Minimum_Price"),
    F.avg("price").alias("Average_Price")
)

df_stats.display()


# COMMAND ----------

# MAGIC %md
# MAGIC Find the top 3 products by quantity sold

# COMMAND ----------

from pyspark.sql import functions as F

df_top3 = df.orderBy(F.desc("quantity")).limit(3)
df_top3.display()


# COMMAND ----------

df_new = df.withColumn("Total",round(col("price")*col("quantity"),scale=2))
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Get the top 2 most expensive orders per city using window functions

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import *
w = Window.partitionBy("city").orderBy(desc("Total"))
df_new = df_new.withColumn("rank",dense_rank().over(w))
df_new.display()

# COMMAND ----------


