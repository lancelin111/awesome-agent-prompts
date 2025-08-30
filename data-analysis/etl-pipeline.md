# ETL数据管道 Prompts

## ETL Pipeline设计

```
请设计一个完整的ETL数据管道：

【数据源信息】
- 源系统：[数据库/API/文件/流式]
- 数据量：[日增量/总量]
- 更新频率：[实时/批处理]
- 数据格式：[JSON/CSV/Parquet等]

【目标系统】
- 数据仓库：[Snowflake/BigQuery/Redshift]
- 存储格式：[列式/行式]
- 分区策略：[时间/地理/业务]

【ETL实现方案】

1. **数据抽取(Extract)**
   ```python
   import pandas as pd
   import requests
   from sqlalchemy import create_engine
   import pyarrow.parquet as pq
   
   class DataExtractor:
       def __init__(self, config):
           self.config = config
           self.connection = self.setup_connection()
       
       def extract_from_database(self, query, chunk_size=10000):
           """分批抽取数据库数据"""
           engine = create_engine(self.config['db_url'])
           
           for chunk in pd.read_sql(query, engine, 
                                    chunksize=chunk_size):
               yield chunk
       
       def extract_from_api(self, endpoint, params=None):
           """API数据抽取with重试机制"""
           from tenacity import retry, stop_after_attempt, 
                              wait_exponential
           
           @retry(stop=stop_after_attempt(3),
                  wait=wait_exponential(multiplier=1, min=4, max=10))
           def fetch():
               response = requests.get(endpoint, params=params)
               response.raise_for_status()
               return response.json()
           
           return fetch()
       
       def extract_incremental(self, table, timestamp_col, 
                              last_sync_time):
           """增量数据抽取"""
           query = f"""
           SELECT * FROM {table}
           WHERE {timestamp_col} > '{last_sync_time}'
           ORDER BY {timestamp_col}
           """
           return self.extract_from_database(query)
   ```

2. **数据转换(Transform)**
   ```python
   import numpy as np
   from datetime import datetime
   import hashlib
   
   class DataTransformer:
       def __init__(self, rules):
           self.rules = rules
       
       def clean_data(self, df):
           """数据清洗"""
           # 去重
           df = df.drop_duplicates(
               subset=self.rules.get('unique_keys', None)
           )
           
           # 处理缺失值
           for col, strategy in self.rules['missing_values'].items():
               if strategy == 'drop':
                   df = df.dropna(subset=[col])
               elif strategy == 'fill_mean':
                   df[col].fillna(df[col].mean(), inplace=True)
               elif strategy == 'fill_forward':
                   df[col].fillna(method='ffill', inplace=True)
               else:
                   df[col].fillna(strategy, inplace=True)
           
           return df
       
       def validate_data(self, df):
           """数据验证"""
           errors = []
           
           # 数据类型验证
           for col, dtype in self.rules['dtypes'].items():
               if col in df.columns:
                   try:
                       df[col] = df[col].astype(dtype)
                   except Exception as e:
                       errors.append(f"Column {col}: {str(e)}")
           
           # 业务规则验证
           for rule in self.rules['business_rules']:
               invalid = df.query(f"not ({rule['condition']})")
               if len(invalid) > 0:
                   errors.append(f"{rule['name']}: 
                                {len(invalid)} violations")
           
           return df, errors
       
       def transform_dimensions(self, df):
           """维度转换"""
           # 时间维度
           df['date_key'] = pd.to_datetime(df['timestamp'])
                             .dt.strftime('%Y%m%d').astype(int)
           df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
           df['weekday'] = pd.to_datetime(df['timestamp'])
                          .dt.dayofweek
           
           # 地理维度
           df['region'] = df['country'].map(self.rules['geo_mapping'])
           
           # 生成代理键
           df['customer_key'] = df.apply(
               lambda x: hashlib.md5(
                   f"{x['customer_id']}_{x['source']}".encode()
               ).hexdigest()[:8], axis=1
           )
           
           return df
       
       def aggregate_metrics(self, df):
           """聚合计算"""
           aggregations = {
               'revenue': ['sum', 'mean', 'std'],
               'quantity': ['sum', 'count'],
               'customer_id': 'nunique'
           }
           
           return df.groupby(
               ['date_key', 'product_category']
           ).agg(aggregations).reset_index()
   ```

3. **数据加载(Load)**
   ```python
   from concurrent.futures import ThreadPoolExecutor
   import boto3
   
   class DataLoader:
       def __init__(self, target_config):
           self.config = target_config
           self.engine = create_engine(target_config['db_url'])
       
       def load_to_warehouse(self, df, table_name, 
                            mode='append', partition_cols=None):
           """加载到数据仓库"""
           # 分区写入
           if partition_cols:
               for partition_value in df[partition_cols[0]].unique():
                   partition_df = df[df[partition_cols[0]] == 
                                    partition_value]
                   
                   partition_table = f"{table_name}_
                                     {partition_value}"
                   partition_df.to_sql(
                       partition_table,
                       self.engine,
                       if_exists=mode,
                       index=False,
                       method='multi',
                       chunksize=10000
                   )
           else:
               df.to_sql(table_name, self.engine, 
                        if_exists=mode, index=False)
       
       def load_to_s3(self, df, bucket, key, format='parquet'):
           """加载到S3"""
           s3 = boto3.client('s3')
           
           if format == 'parquet':
               buffer = BytesIO()
               df.to_parquet(buffer, engine='pyarrow', 
                           compression='snappy')
               buffer.seek(0)
               s3.put_object(Bucket=bucket, Key=key, 
                           Body=buffer.getvalue())
           
       def upsert_scd_type2(self, df, dimension_table, 
                           business_key, effective_date):
           """缓慢变化维Type 2处理"""
           # 获取当前维度数据
           current = pd.read_sql(
               f"SELECT * FROM {dimension_table} 
                WHERE end_date IS NULL",
               self.engine
           )
           
           # 识别变化的记录
           merged = df.merge(current, on=business_key, 
                           how='outer', indicator=True)
           
           # 新记录
           new_records = merged[merged['_merge'] == 'left_only']
           new_records['start_date'] = effective_date
           new_records['end_date'] = None
           new_records['is_current'] = True
           
           # 更新的记录
           updated = merged[merged['_merge'] == 'both']
           # 比较所有非键列...
           
           return new_records, updated
   ```

4. **任务编排**
   ```python
   from airflow import DAG
   from airflow.operators.python import PythonOperator
   from airflow.providers.postgres.operators.postgres 
        import PostgresOperator
   
   default_args = {
       'owner': 'data-team',
       'retries': 2,
       'retry_delay': timedelta(minutes=5)
   }
   
   dag = DAG(
       'etl_pipeline',
       default_args=default_args,
       schedule_interval='0 2 * * *',  # 每天凌晨2点
       catchup=False
   )
   
   # 任务依赖关系
   extract_task >> validate_task >> transform_task >> load_task
   ```

5. **监控与日志**
   ```python
   import logging
   from datadog import statsd
   
   class ETLMonitor:
       def __init__(self):
           self.logger = self.setup_logger()
       
       def log_metrics(self, stage, records_processed, 
                      duration, errors=0):
           # 发送metrics
           statsd.gauge(f'etl.{stage}.records', records_processed)
           statsd.gauge(f'etl.{stage}.duration', duration)
           statsd.gauge(f'etl.{stage}.errors', errors)
           
           # 日志记录
           self.logger.info(
               f"Stage: {stage}, "
               f"Records: {records_processed}, "
               f"Duration: {duration}s, "
               f"Errors: {errors}"
           )
       
       def alert_on_failure(self, error_msg):
           # 发送告警
           pass
   ```

6. **数据质量检查**
   ```python
   class DataQualityChecker:
       def __init__(self, rules):
           self.rules = rules
       
       def check_completeness(self, df, required_cols):
           """完整性检查"""
           missing = set(required_cols) - set(df.columns)
           null_counts = df[required_cols].isnull().sum()
           return missing, null_counts
       
       def check_uniqueness(self, df, unique_cols):
           """唯一性检查"""
           duplicates = df[df.duplicated(subset=unique_cols)]
           return len(duplicates)
       
       def check_consistency(self, df):
           """一致性检查"""
           # 外键一致性、业务规则一致性等
           pass
       
       def check_timeliness(self, df, date_col):
           """时效性检查"""
           latest = df[date_col].max()
           delay = (datetime.now() - latest).days
           return delay
   ```
```