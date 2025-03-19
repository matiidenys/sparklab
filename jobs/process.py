from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, desc, date_format, date_sub, to_date, lit, row_number, max as spark_max
from pyspark.sql.window import Window

def main():
    spark = SparkSession.builder.appName("Lab4").getOrCreate()
    df = spark.read.csv('./jobs/Divvy_Trips_2019_Q4.csv', header=True,escape="\"")
    
    quest_1 = avg_trip_duration_per_day(df)
    quest_1.show()
    quest_1.write.csv('./jobs/out/avg_trip_duration_per_day.csv', header=True, mode="overwrite")

    quest_2 = trips_per_day(df)
    quest_2.show()
    quest_2.write.csv('./jobs/out/trips_per_day.csv', header=True, mode="overwrite")

    quest_3 = most_popular_start_station_per_month(df)
    quest_3.show()
    quest_3.write.csv('./jobs/out/most_popular_start_station_per_month.csv', header=True, mode="overwrite")

    quest_4 = top_three_stations_last_two_weeks(df)
    quest_4.show()
    quest_4.write.csv('./jobs/out/top_three_stations_last_two_weeks.csv', header=True, mode="overwrite")

    quest_5 = avg_trip_duration_by_gender(df)
    quest_5.show()
    quest_5.write.csv('./jobs/out/avg_trip_duration_by_gender.csv', header=True, mode="overwrite")


def avg_trip_duration_per_day(df):
    return df.groupBy(date_format("start_time", "yyyy-MM-dd").alias("day"))\
            .agg(avg("tripduration").alias("avg_duration"))
def trips_per_day(df):
    return df.groupBy(date_format("start_time", "yyyy-MM-dd").alias("day"))\
            .agg(count("trip_id").alias("trip_count"))

def most_popular_start_station_per_month(df):
    return df.groupBy(date_format("start_time", "yyyy-MM").alias("month"), "from_station_name")\
            .agg(count("trip_id").alias("trip_count"))\
            .orderBy("month", desc("trip_count"))\
            .dropDuplicates(["month"])

def top_three_stations_last_two_weeks(df):
    df = df.withColumn("date", to_date("start_time"))

    # Визначаємо максимальну дату в датасеті
    max_date_row = df.agg(spark_max("date").alias("max_date")).collect()[0]
    max_date = max_date_row["max_date"]
    # Фільтруємо останні 14 днів
    df_last2weeks = df.filter(col("date") >= date_sub(lit(max_date), 14))

    grouped = df_last2weeks.groupBy("date", "from_station_name") \
        .agg(count("*").alias("trip_count"))

    window_spec = Window.partitionBy("date").orderBy(desc("trip_count"))

    result = (
        grouped.withColumn("rn", row_number().over(window_spec))
        .filter(col("rn") <= 3)
        .drop("rn")
        .orderBy("date")
    )
    return result


def avg_trip_duration_by_gender(df):
    # Фільтрація значень "Male" і "Female", виключення порожніх значень
    filtered_df = df.filter(col("gender").isin("Male", "Female"))
    
    # Обчислення середнього значення tripduration за гендером
    avg_duration_df = filtered_df.groupBy("gender").agg(avg("tripduration").alias("avg_duration"))
    
    # Сортування по середньому значенню і виведення найвищого
    return avg_duration_df.orderBy(desc("avg_duration")).limit(1)


if __name__ == '__main__':
    main()