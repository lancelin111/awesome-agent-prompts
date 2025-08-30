# 大数据处理 Prompts

## Spark数据处理优化

```
请优化以下Spark数据处理任务：

【当前代码】
```python
# PySpark代码
df = spark.read.parquet("s3://bucket/data/")
result = df.groupBy("category").agg(
    F.sum("amount").alias("total_amount"),
    F.count("*").alias("count")
)
result.write.parquet("s3://bucket/output/")
```

【性能问题】
- 处理时间：2小时
- 数据量：100TB
- 集群配置：20节点，每节点16核64GB

【优化方案】

1. **分区优化**
   ```python
   from pyspark.sql import SparkSession
   from pyspark.sql import functions as F
   from pyspark.sql.types import *
   
   # 优化Spark配置
   spark = SparkSession.builder \
       .appName("OptimizedETL") \
       .config("spark.sql.adaptive.enabled", "true") \
       .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
       .config("spark.sql.adaptive.skewJoin.enabled", "true") \
       .config("spark.sql.shuffle.partitions", "2000") \
       .config("spark.sql.files.maxPartitionBytes", "134217728") \
       .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
       .getOrCreate()
   
   # 分区读取
   df = spark.read \
       .option("mergeSchema", "false") \
       .option("recursiveFileLookup", "true") \
       .parquet("s3://bucket/data/")
   
   # 重分区优化
   optimal_partitions = df.rdd.getNumPartitions()
   print(f"Current partitions: {optimal_partitions}")
   
   # 根据数据倾斜情况调整
   if optimal_partitions > 2000:
       df = df.coalesce(2000)
   elif optimal_partitions < 200:
       df = df.repartition(200)
   ```

2. **缓存策略**
   ```python
   # 选择合适的缓存级别
   from pyspark.storagelevel import StorageLevel
   
   # 频繁访问的数据集
   df.persist(StorageLevel.MEMORY_AND_DISK_SER)
   
   # 检查缓存状态
   print(f"Is cached: {df.is_cached}")
   print(f"Storage level: {df.storageLevel}")
   
   # 广播小表
   from pyspark.sql.functions import broadcast
   
   small_df = spark.read.parquet("s3://bucket/dimension/")
   if small_df.count() < 1000000:  # 小于100万条
       result = df.join(broadcast(small_df), "key")
   ```

3. **数据倾斜处理**
   ```python
   # 检测数据倾斜
   def detect_skew(df, column):
       stats = df.groupBy(column).count() \
           .select(
               F.min("count").alias("min"),
               F.max("count").alias("max"),
               F.mean("count").alias("mean"),
               F.stddev("count").alias("stddev")
           ).collect()[0]
       
       skew_ratio = stats["max"] / stats["mean"]
       return skew_ratio > 10  # 倾斜阈值
   
   # 处理倾斜的聚合
   if detect_skew(df, "category"):
       # 两阶段聚合
       df_with_salt = df.withColumn(
           "salt", F.concat(F.col("category"), 
                           F.lit("_"), 
                           F.floor(F.rand() * 10))
       )
       
       # 第一阶段：加盐聚合
       stage1 = df_with_salt.groupBy("salt").agg(
           F.sum("amount").alias("partial_sum"),
           F.count("*").alias("partial_count")
       )
       
       # 第二阶段：去盐聚合
       stage2 = stage1.withColumn(
           "category", F.split(F.col("salt"), "_")[0]
       ).groupBy("category").agg(
           F.sum("partial_sum").alias("total_amount"),
           F.sum("partial_count").alias("count")
       )
   ```

4. **列式存储优化**
   ```python
   # 列裁剪
   required_cols = ["category", "amount", "date"]
   df_filtered = df.select(*required_cols)
   
   # 谓词下推
   df_filtered = spark.read \
       .option("pushDownPredicate", "true") \
       .parquet("s3://bucket/data/") \
       .filter(F.col("date") >= "2024-01-01")
   
   # 分区裁剪
   df_partitioned = spark.read \
       .parquet("s3://bucket/data/year=2024/month=01/")
   ```

5. **Join优化**
   ```python
   # Sort-Merge Join vs Broadcast Join
   spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "100MB")
   
   # 优化Join顺序
   # 先filter再join
   df1_filtered = df1.filter(F.col("active") == True)
   df2_filtered = df2.filter(F.col("status") == "valid")
   result = df1_filtered.join(df2_filtered, "key")
   
   # 使用bloom filter for存在性检查
   from pyspark.sql.functions import bloom_filter_agg, 
                                   bloom_filter_contains
   
   bf = df1.select(bloom_filter_agg("key", 1000000, 0.03)
                  .alias("bloom_filter")).collect()[0][0]
   
   df2_filtered = df2.filter(
       bloom_filter_contains(bf, F.col("key"))
   )
   ```

6. **输出优化**
   ```python
   # 控制输出文件数量
   result.coalesce(100) \
       .write \
       .mode("overwrite") \
       .option("compression", "snappy") \
       .partitionBy("date") \
       .parquet("s3://bucket/output/")
   
   # 动态分区写入
   result.write \
       .mode("overwrite") \
       .option("maxRecordsPerFile", 1000000) \
       .option("spark.sql.sources.partitionOverwriteMode", "dynamic") \
       .partitionBy("year", "month") \
       .parquet("s3://bucket/output/")
   ```

7. **监控与调试**
   ```python
   # 执行计划分析
   result.explain(True)  # 查看物理执行计划
   
   # Stage级别监控
   sc = spark.sparkContext
   sc.setJobDescription("Aggregation Stage")
   
   # 自定义累加器监控
   error_counter = sc.accumulator(0)
   
   def process_with_monitoring(row):
       try:
           # 处理逻辑
           return process(row)
       except:
           error_counter.add(1)
           return None
   
   # 资源使用监控
   spark.conf.set("spark.eventLog.enabled", "true")
   spark.conf.set("spark.eventLog.dir", "s3://bucket/spark-logs/")
   ```
```

## 流式数据处理

```
设计实时数据处理pipeline：

【需求】
- 数据源：Kafka
- 吞吐量：100万条/秒
- 延迟要求：< 1秒
- 处理逻辑：聚合、异常检测、实时告警

【实现方案】

1. **Kafka消费优化**
   ```python
   from pyspark.sql import SparkSession
   from pyspark.sql.functions import *
   from pyspark.sql.types import *
   
   # Structured Streaming配置
   spark = SparkSession.builder \
       .appName("RealTimeProcessing") \
       .config("spark.sql.streaming.checkpointLocation", 
               "s3://bucket/checkpoint/") \
       .config("spark.sql.streaming.stateStore.stateSchemaCheck", 
               "false") \
       .config("spark.sql.streaming.metricsEnabled", "true") \
       .getOrCreate()
   
   # Kafka源配置
   df = spark.readStream \
       .format("kafka") \
       .option("kafka.bootstrap.servers", "localhost:9092") \
       .option("subscribe", "topic1,topic2") \
       .option("startingOffsets", "latest") \
       .option("maxOffsetsPerTrigger", 100000) \
       .option("kafka.consumer.max.poll.records", 10000) \
       .option("kafka.session.timeout.ms", 30000) \
       .load()
   ```

2. **窗口聚合**
   ```python
   # 定义schema
   schema = StructType([
       StructField("timestamp", TimestampType(), True),
       StructField("user_id", StringType(), True),
       StructField("event_type", StringType(), True),
       StructField("amount", DoubleType(), True)
   ])
   
   # 解析数据
   parsed = df.select(
       from_json(col("value").cast("string"), schema).alias("data")
   ).select("data.*")
   
   # 滑动窗口聚合
   windowed = parsed \
       .withWatermark("timestamp", "10 minutes") \
       .groupBy(
           window("timestamp", "5 minutes", "1 minute"),
           "user_id"
       ).agg(
           count("*").alias("event_count"),
           sum("amount").alias("total_amount"),
           avg("amount").alias("avg_amount"),
           collect_set("event_type").alias("event_types")
       )
   
   # Session窗口
   session_window = parsed \
       .withWatermark("timestamp", "10 minutes") \
       .groupBy(
           session_window("timestamp", "5 minutes"),
           "user_id"
       ).agg(
           min("timestamp").alias("session_start"),
           max("timestamp").alias("session_end"),
           count("*").alias("events_in_session")
       )
   ```

3. **实时异常检测**
   ```python
   from pyspark.ml.feature import StandardScaler
   from pyspark.ml.clustering import KMeans
   
   # 特征工程
   def extract_features(df):
       return df.select(
           hour("timestamp").alias("hour"),
           dayofweek("timestamp").alias("dow"),
           col("amount"),
           when(col("event_type") == "purchase", 1).otherwise(0)
               .alias("is_purchase")
       )
   
   # 异常检测UDF
   @udf(returnType=BooleanType())
   def detect_anomaly(amount, avg_amount, std_amount):
       if avg_amount is None or std_amount is None:
           return False
       z_score = abs((amount - avg_amount) / std_amount)
       return z_score > 3
   
   # 应用异常检测
   with_stats = parsed.join(
       parsed.groupBy("user_id").agg(
           avg("amount").alias("avg_amount"),
           stddev("amount").alias("std_amount")
       ),
       "user_id"
   )
   
   anomalies = with_stats.withColumn(
       "is_anomaly",
       detect_anomaly(col("amount"), 
                      col("avg_amount"), 
                      col("std_amount"))
   ).filter(col("is_anomaly") == True)
   ```

4. **状态管理**
   ```python
   from pyspark.sql.streaming.state import GroupState, 
                                           GroupStateTimeout
   
   # 自定义状态处理
   def update_user_state(key, values, state: GroupState):
       # 获取当前状态
       if state.exists:
           current_state = state.get
       else:
           current_state = {"count": 0, "total": 0}
       
       # 更新状态
       for value in values:
           current_state["count"] += 1
           current_state["total"] += value.amount
       
       # 设置超时
       state.setTimeoutDuration("1 hour")
       state.update(current_state)
       
       return (key, current_state["total"] / current_state["count"])
   
   # 应用状态函数
   stateful = parsed.groupByKey(lambda x: x.user_id) \
       .mapGroupsWithState(
           update_user_state,
           timeout=GroupStateTimeout.ProcessingTimeTimeout
       )
   ```

5. **输出与告警**
   ```python
   # 多路输出
   # 1. 写入数据湖
   query1 = windowed.writeStream \
       .outputMode("append") \
       .format("parquet") \
       .option("path", "s3://bucket/streaming/aggregated/") \
       .option("checkpointLocation", "s3://bucket/checkpoint1/") \
       .trigger(processingTime="1 minute") \
       .start()
   
   # 2. 写入告警系统
   def send_alert(df, epoch_id):
       alerts = df.collect()
       for alert in alerts:
           # 发送告警逻辑
           send_to_pagerduty(alert)
   
   query2 = anomalies.writeStream \
       .outputMode("append") \
       .foreachBatch(send_alert) \
       .trigger(processingTime="10 seconds") \
       .start()
   
   # 3. 写入实时仪表板
   query3 = windowed.writeStream \
       .outputMode("complete") \
       .format("memory") \
       .queryName("dashboard_data") \
       .trigger(continuous="1 second") \
       .start()
   
   # 等待所有查询
   spark.streams.awaitAnyTermination()
   ```
```