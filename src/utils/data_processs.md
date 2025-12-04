专业数据处理与可视化实战指南

本文档专注于数据预处理 (Data Preprocessing) 环节，提供工业级的数据清洗逻辑，并为每个步骤匹配了最有效的可视化验证方法。在健康数据分析中，可视化不仅是展示结果的手段，更是发现数据质量问题（如异常录入、设备故障）的核心工具。

1. 缺失值处理 (Missing Data Handling)

核心逻辑：医疗数据缺失通常不是随机的（例如：健康的人可能没做某些昂贵的检查）。盲目填充会引入偏差。

1.1 侦测与可视化

在处理之前，必须先看清“哪里缺了”以及“缺失是否存在规律”。

缺失数据矩阵图 (Nullity Matrix)

用途：直观展示数据密度的分布，识别特定样本段的系统性缺失（如某台设备故障导致一段时间内所有指标缺失）。

可视化方案：使用黑白条纹图，黑色代表有数据，白色代表缺失。

缺失相关性热力图 (Nullity Correlation Heatmap)

用途：分析指标缺失之间的关联。例如：如果“腰围”缺失，是否“臀围”也大概率缺失？

1.2 处理策略 (专业建议)

数据类型

缺失率 < 5%

缺失率 5% - 30%

缺失率 > 30%

数值型 (连续)

优先用 中位数 (Median) 填充（抗干扰强于均值）

使用 KNN Imputer 或 RandomForest 预测填充

建议删除该列，或转换为二值特征（“是否有数据”）

类别型 (离散)

用 众数 (Mode) 填充

填为一个新类别（如 "Unknown"）

删除该列

2. 异常值处理 (Outlier Detection)

核心逻辑：在健康领域，异常值可能是录入错误（如身高 180cm 录成 18cm），也可能是真实的病理状态（如极高血压）。处理时需极其谨慎，通常建议盖帽法 (Capping) 而非直接删除。

2.1 侦测与可视化

箱线图 (Box Plot) - 最经典

用途：展示数据的四分位数。超过 $1.5 \times IQR$ 的点被定义为异常值。

解读：对于体检数据，箱线图能一眼看出某个指标是否存在极值。

小提琴图 (Violin Plot) - 进阶

用途：结合了箱线图和核密度图。不仅能看异常值，还能看数据分布的“形状”（是单峰还是双峰）。

场景：分析不同性别（分组）的身高分布。

2.2 处理策略

IQR 截断法 (Tukey's Fences)：

计算 $Q1$ (25%) 和 $Q3$ (75%)。

定义上下界：$Lower = Q1 - 1.5 \times IQR$, $Upper = Q3 + 1.5 \times IQR$。

处理：将超过 Upper 的值强制设为 Upper，低于 Lower 的值设为 Lower（即“盖帽法”），避免删除真实病患数据。

Z-Score 标准分：

适用于近似正态分布的数据。

$|Z| > 3$ 视为异常。

3. 数据分布校正 (Distribution Transformation)

核心逻辑：许多统计模型（如线性回归）假设数据服从正态分布。但健康数据常呈现偏态（如体重数据通常右偏，因为肥胖人群长尾）。

3.1 侦测与可视化

直方图 + KDE 曲线 (Histogram with Kernel Density Estimate)

用途：最直观地查看数据是否偏斜。

关注点：曲线的峰值是否偏移中心，尾部是否拖得很长。

Q-Q 图 (Quantile-Quantile Plot) - 统计学标准

用途：验证数据是否符合正态分布。

解读：如果散点紧贴红色对角线，则为正态分布；如果两端发散，则说明有厚尾或偏态。

3.2 处理策略

对数变换 (Log Transformation)：$x' = \log(x + 1)$。适用于右偏数据（如收入、体重）。

Box-Cox 变换：自动寻找最佳变换参数 $\lambda$，使数据最接近正态分布。

4. 相关性分析 (Correlation Analysis)

核心逻辑：在建模前排除多重共线性（两个特征高度相关，如“体重”和“BMI”），以精简模型。

4.1 可视化方案

相关性热力图 (Correlation Heatmap)

用途：使用颜色深浅表示 Pearson 相关系数 ($-1$ 到 $1$)。

操作：通常将系数绝对值 $> 0.8$ 的高相关特征组中剔除一个。

配对图 (Pair Plot / Scatter Matrix)

用途：展示所有数值变量两两之间的散点关系。

场景：快速浏览身高、体重、体脂率、血压这四个变量之间的两两关系。

5. Python 实战代码示例

以下代码展示了如何结合 Seaborn 和 Scikit-learn 完成上述流程。

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# 模拟健康数据
data = pd.DataFrame({
    'Weight': np.random.normal(70, 15, 200),
    'Height': np.random.normal(175, 10, 200),
    'BodyFat': np.random.gamma(2, 2, 200) # 偏态分布
})
# 引入缺失和异常
data.loc[0:10, 'Weight'] = np.nan
data.loc[20, 'Weight'] = 500 # 异常值

# ==========================================
# 1. 缺失值可视化
# ==========================================
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Matrix")
plt.show()

# 处理：用中位数填充
data['Weight'].fillna(data['Weight'].median(), inplace=True)

# ==========================================
# 2. 异常值可视化与处理 (IQR)
# ==========================================
# 可视化：箱线图
plt.figure(figsize=(8, 4))
sns.boxplot(x=data['Weight'])
plt.title("Weight Distribution (Before Cleaning)")
plt.show()

# 处理：盖帽法
Q1 = data['Weight'].quantile(0.25)
Q3 = data['Weight'].quantile(0.75)
IQR = Q3 - Q1
upper_limit = Q3 + 1.5 * IQR
lower_limit = Q1 - 1.5 * IQR

# 将超出部分截断
data['Weight_Capped'] = data['Weight'].clip(lower=lower_limit, upper=upper_limit)

# ==========================================
# 3. 分布校正与可视化 (Q-Q图)
# ==========================================
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# 原始偏态分布
sns.histplot(data['BodyFat'], kde=True, ax=ax[0])
ax[0].set_title("Original BodyFat (Skewed)")

# Log 变换后
data['BodyFat_Log'] = np.log1p(data['BodyFat'])
sns.histplot(data['BodyFat_Log'], kde=True, ax=ax[1])
ax[1].set_title("Log Transformed BodyFat (Normalized)")
plt.show()

# Q-Q 图验证
plt.figure()
stats.probplot(data['BodyFat_Log'], dist="norm", plot=plt)
plt.title("Q-Q Plot for Log-Transformed Data")
plt.show()

# ==========================================
# 4. 相关性热力图
# ==========================================
plt.figure(figsize=(8, 6))
# 计算相关系数矩阵
corr = data[['Weight_Capped', 'Height', 'BodyFat_Log']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Feature Correlation Heatmap")
plt.show()
