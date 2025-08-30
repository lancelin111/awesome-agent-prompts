# 统计分析 Prompts

## A/B测试分析

```
请为以下A/B测试设计完整的统计分析方案：

【实验背景】
- 测试目标：[描述要验证的假设]
- 控制组：[描述A版本]
- 实验组：[描述B版本]
- 关键指标：[主要指标和次要指标]
- 测试时长：[预计时间]

【样本信息】
- 用户总量：[数量]
- 分组方式：[随机/分层/聚类]
- 分流比例：[如50:50]

【分析方案】

1. **样本量计算**
   ```python
   from statsmodels.stats.power import tt_solve_power
   
   # 参数设置
   effect_size = 0.2  # 期望检测的效应量
   alpha = 0.05       # 显著性水平
   power = 0.8        # 统计功效
   
   # 计算所需样本量
   n = tt_solve_power(effect_size=effect_size, 
                      alpha=alpha, 
                      power=power)
   print(f"每组所需样本量: {n:.0f}")
   ```

2. **数据质量检查**
   - 随机性检验（卡方检验）
   - 样本分布检查（AA测试）
   - 新奇效应识别
   - 数据完整性验证

3. **统计检验选择**
   ```python
   # 连续变量
   if continuous_variable:
       if normal_distribution:
           test = "t-test"  # 参数检验
       else:
           test = "Mann-Whitney U"  # 非参数检验
   
   # 比例变量
   elif proportion_variable:
       if large_sample:
           test = "z-test"
       else:
           test = "Fisher's exact test"
   
   # 多组比较
   elif multiple_groups:
       test = "ANOVA" if normal else "Kruskal-Wallis"
   ```

4. **效应量计算**
   - Cohen's d（均值差异）
   - 相对提升率
   - 置信区间估计
   - 实际显著性 vs 统计显著性

5. **多重检验校正**
   ```python
   from statsmodels.stats.multitest import multipletests
   
   # Bonferroni校正
   corrected = multipletests(p_values, method='bonferroni')
   
   # FDR控制
   corrected = multipletests(p_values, method='fdr_bh')
   ```

6. **结果解释模板**
   - 统计显著性判断
   - 业务影响评估
   - 决策建议
   - 风险提示
```

## 时间序列分析

```
请对以下时间序列数据进行全面分析：

【数据描述】
- 时间范围：[起止时间]
- 采样频率：[日/周/月]
- 数据特征：[单变量/多变量]
- 业务含义：[描述指标含义]

【数据示例】
```python
df.head()
# timestamp  value
# 2024-01-01  100
# 2024-01-02  105
# ...
```

【分析需求】

1. **数据探索**
   ```python
   # 基本统计
   df['value'].describe()
   
   # 时序图
   fig, axes = plt.subplots(4, 1, figsize=(12, 10))
   
   # 原始序列
   df['value'].plot(ax=axes[0])
   axes[0].set_title('原始时间序列')
   
   # 移动平均
   df['value'].rolling(7).mean().plot(ax=axes[1])
   axes[1].set_title('7日移动平均')
   
   # 一阶差分
   df['value'].diff().plot(ax=axes[2])
   axes[2].set_title('一阶差分')
   
   # 季节性分解
   from statsmodels.tsa.seasonal import seasonal_decompose
   decomposition = seasonal_decompose(df['value'], period=7)
   ```

2. **平稳性检验**
   ```python
   from statsmodels.tsa.stattools import adfuller, kpss
   
   # ADF检验
   adf_result = adfuller(df['value'])
   print(f'ADF统计量: {adf_result[0]}')
   print(f'p-value: {adf_result[1]}')
   
   # KPSS检验
   kpss_result = kpss(df['value'])
   ```

3. **自相关分析**
   ```python
   from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
   
   # ACF和PACF图
   fig, axes = plt.subplots(2, 1, figsize=(12, 8))
   plot_acf(df['value'], lags=40, ax=axes[0])
   plot_pacf(df['value'], lags=40, ax=axes[1])
   ```

4. **模型选择与预测**
   ```python
   # ARIMA模型
   from statsmodels.tsa.arima.model import ARIMA
   from pmdarima import auto_arima
   
   # 自动选择参数
   model = auto_arima(df['value'], 
                      seasonal=True, 
                      m=7,  # 季节周期
                      stepwise=True)
   
   # Prophet模型
   from prophet import Prophet
   model = Prophet(daily_seasonality=True)
   
   # LSTM模型框架
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import LSTM, Dense
   ```

5. **异常检测**
   ```python
   # 基于统计的方法
   from scipy import stats
   z_scores = stats.zscore(df['value'])
   outliers = df[np.abs(z_scores) > 3]
   
   # 基于预测的方法
   residuals = actual - predicted
   threshold = 3 * residuals.std()
   anomalies = np.abs(residuals) > threshold
   ```

6. **预测评估**
   - MAE, RMSE, MAPE
   - 预测区间
   - 残差分析
   - 交叉验证
```

## 因果推断分析

```
请设计因果推断分析方案：

【研究问题】
- 处理变量(Treatment)：[描述]
- 结果变量(Outcome)：[描述]
- 混杂因素(Confounders)：[列出]
- 数据类型：[观察性/实验性]

【因果推断方法】

1. **倾向得分匹配(PSM)**
   ```python
   from sklearn.linear_model import LogisticRegression
   from sklearn.neighbors import NearestNeighbors
   
   # 计算倾向得分
   X = df[confounders]
   y = df['treatment']
   ps_model = LogisticRegression()
   ps_model.fit(X, y)
   df['propensity_score'] = ps_model.predict_proba(X)[:, 1]
   
   # 1:1匹配
   treated = df[df['treatment'] == 1]
   control = df[df['treatment'] == 0]
   
   # 卡钳匹配
   caliper = 0.1 * df['propensity_score'].std()
   ```

2. **双重差分(DID)**
   ```python
   import statsmodels.formula.api as smf
   
   # DID模型
   model = smf.ols('outcome ~ treatment * post + controls', 
                   data=df).fit()
   
   # 平行趋势检验
   pre_trends = smf.ols('outcome ~ treatment * time + controls',
                        data=df[df['post'] == 0]).fit()
   ```

3. **断点回归(RDD)**
   ```python
   from rdd import rdd
   
   # 确定带宽
   bandwidth = rdd.optimal_bandwidth(df['running_var'], 
                                    df['outcome'])
   
   # 估计处理效应
   result = rdd.rdd(df['running_var'], 
                    df['outcome'],
                    cutoff=threshold,
                    bandwidth=bandwidth)
   ```

4. **工具变量(IV)**
   ```python
   from linearmodels.iv import IV2SLS
   
   # 2SLS估计
   iv_model = IV2SLS(dependent=df['outcome'],
                     exog=df[controls],
                     endog=df['treatment'],
                     instruments=df['instrument']).fit()
   
   # 弱工具变量检验
   f_stat = iv_model.first_stage.fstatistic
   ```

5. **合成控制法**
   ```python
   from synth import Synth
   
   # 构建合成控制
   synth = Synth(df, 
                 outcome='outcome',
                 unit='unit_id',
                 time='time',
                 treatment='treatment_unit',
                 treatment_time=treatment_period)
   ```

6. **敏感性分析**
   - 遗漏变量偏差
   - Rosenbaum bounds
   - E-value计算
   - 安慰剂检验
```