# 数据可视化 Prompts

## 可视化方案设计

```
请为以下数据设计可视化方案：

【数据概述】
- 数据主题：[描述]
- 数据维度：[列出主要字段]
- 数据量级：[数据规模]
- 更新频率：[静态/动态]

【目标受众】
- 主要用户：[管理层/分析师/大众]
- 技术水平：[专业/一般]
- 使用场景：[报告/监控/探索]

【可视化需求】

1. **仪表板整体设计**
   - 布局建议（网格系统）
   - 颜色方案（主题色板）
   - 交互设计（筛选器、钻取）
   - 响应式设计考虑

2. **核心指标展示**
   ```python
   # KPI卡片设计
   kpi_metrics = [
       {'name': '指标1', 'chart': 'number_card', 'format': ',.0f'},
       {'name': '指标2', 'chart': 'gauge', 'range': [0, 100]},
       {'name': '指标3', 'chart': 'sparkline', 'trend': True}
   ]
   ```

3. **图表类型选择**
   | 数据类型 | 分析目的 | 推荐图表 | 备选方案 | 不建议使用 |
   |---------|---------|---------|---------|------------|
   | 时间序列 | 趋势分析 | 折线图 | 面积图 | 饼图 |
   | 分类对比 | 大小比较 | 柱状图 | 条形图 | 3D图表 |
   | 占比分析 | 构成关系 | 饼图 | 环形图 | 折线图 |
   | 相关性 | 关系探索 | 散点图 | 热力图 | 柱状图 |
   | 分布 | 数据分布 | 直方图 | 箱线图 | 饼图 |

4. **Python实现代码**
   ```python
   import plotly.graph_objects as go
   from plotly.subplots import make_subplots
   import plotly.express as px
   
   # 创建交互式仪表板
   fig = make_subplots(
       rows=2, cols=2,
       subplot_titles=('图表1', '图表2', '图表3', '图表4'),
       specs=[[{'type': 'bar'}, {'type': 'scatter'}],
              [{'type': 'indicator'}, {'type': 'pie'}]]
   )
   ```

5. **进阶可视化**
   - 地理空间可视化
   - 网络关系图
   - 桑基图（流程分析）
   - 树图（层级关系）
   - 平行坐标图（多维分析）
```

## Matplotlib/Seaborn美化专家

```
请优化以下matplotlib/seaborn图表的视觉效果：

【原始代码】
```python
[粘贴现有绘图代码]
```

【美化要求】
- 风格：[学术/商务/现代/极简]
- 配色：[指定颜色或主题]
- 用途：[论文/报告/网页/PPT]

【优化方案】

1. **全局样式设置**
   ```python
   # 自定义样式
   plt.style.use('seaborn-v0_8-darkgrid')
   
   # 字体设置
   plt.rcParams['font.family'] = 'DejaVu Sans'
   plt.rcParams['font.size'] = 12
   
   # 解决中文显示
   plt.rcParams['font.sans-serif'] = ['SimHei']
   plt.rcParams['axes.unicode_minus'] = False
   ```

2. **配色方案**
   ```python
   # 自定义调色板
   colors = {
       'primary': '#2E86AB',
       'secondary': '#F24236',
       'accent': '#F6AE2D',
       'neutral': '#2F4858'
   }
   
   # 渐变色
   cmap = sns.color_palette("coolwarm", as_cmap=True)
   ```

3. **图表元素优化**
   ```python
   # 优化后的代码
   fig, ax = plt.subplots(figsize=(12, 6))
   
   # 添加网格
   ax.grid(True, alpha=0.3, linestyle='--')
   
   # 优化坐标轴
   ax.spines['top'].set_visible(False)
   ax.spines['right'].set_visible(False)
   
   # 添加标注
   for i, v in enumerate(values):
       ax.text(i, v, f'{v:.1f}', ha='center', va='bottom')
   ```

4. **添加统计信息**
   ```python
   # 添加均值线
   ax.axhline(y=mean_value, color='red', linestyle='--', alpha=0.7)
   
   # 添加置信区间
   ax.fill_between(x, lower_bound, upper_bound, alpha=0.2)
   
   # 添加回归线
   sns.regplot(x='x', y='y', data=df, ax=ax)
   ```

5. **导出设置**
   ```python
   # 高质量输出
   plt.savefig('figure.png', dpi=300, bbox_inches='tight')
   plt.savefig('figure.pdf', format='pdf', bbox_inches='tight')
   ```
```