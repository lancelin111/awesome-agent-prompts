# 数据分析 Prompts

用于数据分析和可视化的prompt集合。

## 📊 主要分类

### 基础分析
- **[数据分析](data-analysis.md)** - 探索性数据分析、SQL优化、数据报告生成
- **[统计分析](statistical-analysis.md)** - A/B测试、时间序列分析、因果推断
- **[数据可视化](data-visualization.md)** - 可视化方案设计、图表美化

### 高级分析
- **[机器学习](machine-learning.md)** - 模型选择、特征工程、模型调优
- **[深度学习数据处理](deep-learning-data.md)** - 数据集准备、数据增强、训练监控

### 数据工程
- **[ETL数据管道](etl-pipeline.md)** - ETL设计、数据清洗、数据加载
- **[大数据处理](big-data-processing.md)** - Spark优化、流式处理

## 🛠 使用指南

### 1. 选择合适的Prompt
根据你的具体需求，选择对应的prompt模板：
- 数据探索 → 探索性数据分析
- 建模预测 → 机器学习
- 报表展示 → 数据可视化
- 实时处理 → 流式数据处理

### 2. 定制化修改
每个prompt都包含：
- **输入参数** - 根据实际情况填写
- **处理逻辑** - 可根据需求调整
- **输出格式** - 按需求定制

### 3. 最佳实践
- 📌 **明确目标** - 清楚定义分析目的
- 📊 **数据质量** - 确保数据准确完整
- 🔄 **迭代优化** - 根据结果持续改进
- 📈 **性能考虑** - 大数据场景注意优化

## 💡 快速开始示例

### 数据分析项目模板
```python
# 1. 导入必要库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. 数据加载
df = pd.read_csv('your_data.csv')

# 3. 使用对应的prompt进行分析
# 例如：探索性数据分析
# - 查看data-analysis.md中的探索性数据分析prompt
# - 根据提示完成数据概览、单变量分析、多变量分析等

# 4. 生成报告
# - 参考数据报告生成prompt
```

## 🔧 工具推荐

### Python生态
- **Pandas** - 数据处理
- **NumPy** - 数值计算
- **Scikit-learn** - 机器学习
- **TensorFlow/PyTorch** - 深度学习
- **PySpark** - 大数据处理

### 可视化工具
- **Matplotlib/Seaborn** - 统计图表
- **Plotly** - 交互式可视化
- **Tableau/PowerBI** - 商业智能

### 大数据平台
- **Apache Spark** - 分布式计算
- **Apache Kafka** - 流式处理
- **Hadoop** - 数据存储

## 📚 学习资源

1. **基础知识**
   - 统计学基础
   - SQL查询优化
   - Python数据分析

2. **进阶技能**
   - 机器学习算法
   - 深度学习框架
   - 大数据技术栈

3. **实战项目**
   - Kaggle竞赛
   - 开源数据集
   - 企业案例分析

## 🤝 贡献指南

欢迎贡献新的数据分析prompts！请确保：
- ✅ 提供完整的使用示例
- ✅ 包含实际可运行的代码
- ✅ 说明适用场景和限制
- ✅ 遵循现有的文档格式

---

💬 如有问题或建议，欢迎提交Issue或PR！