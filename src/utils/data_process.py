"""
数据预处理模块
包含：缺失值处理、异常值处理、数据分布校正、相关性分析
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.impute import KNNImputer
from typing import Tuple, List, Optional, Dict
import warnings
import matplotlib
from pathlib import Path

# 设置中文字体支持
def setup_matplotlib_chinese():
    """配置 matplotlib 中文字体"""
    import subprocess
    
    # 优先使用的中文字体列表（按可靠性排序）
    preferred_fonts = [
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei', 
        'Noto Sans CJK SC',
        'Source Han Sans SC',
        'Droid Sans Fallback',
        'SimHei',
        'Microsoft YaHei',
    ]
    
    plt.rcParams['axes.unicode_minus'] = False
    
    # 获取 matplotlib 实际可用的字体
    try:
        from matplotlib.font_manager import fontManager, findfont, FontProperties
        available = {f.name for f in fontManager.ttflist}
        
        for font in preferred_fonts:
            if font in available:
                # 验证字体确实可用（不会 fallback）
                try:
                    fp = FontProperties(family=font)
                    font_path = findfont(fp, fallback_to_default=False)
                    if font_path and 'DejaVu' not in font_path:
                        plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
                        return font
                except:
                    continue
    except:
        pass
    
    # 没有找到中文字体，使用默认英文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    return None

# 初始化字体（setup 函数已经验证过字体可用性）
_CHINESE_FONT = setup_matplotlib_chinese()
_HAS_CHINESE = _CHINESE_FONT is not None

warnings.filterwarnings('ignore')

# 最大显示特征数
MAX_DISPLAY_FEATURES = 8


def get_display_text(chinese: str, english: str) -> str:
    """根据字体支持返回中文或英文文本"""
    if _HAS_CHINESE:
        return chinese
    return english


class DataProcessor:
    """数据预处理类"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化数据处理器
        
        Args:
            df: 输入的 DataFrame
        """
        self.df = df.copy()
        self.original_df = df.copy()
        self.processing_log = []
    
    def get_data(self) -> pd.DataFrame:
        """获取处理后的数据"""
        return self.df
    
    def reset(self):
        """重置为原始数据"""
        self.df = self.original_df.copy()
        self.processing_log = []
    
    def _select_representative_features(self, columns: List[str] = None, 
                                          max_features: int = MAX_DISPLAY_FEATURES) -> List[str]:
        """
        选择代表性特征（基于方差和相关性）
        
        Args:
            columns: 候选列
            max_features: 最大特征数
            
        Returns:
            选中的特征列表
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(columns) <= max_features:
            return columns
        
        # 计算每列的方差（标准化后）
        df_numeric = self.df[columns].dropna()
        if len(df_numeric) == 0:
            return columns[:max_features]
        
        # 标准化后计算方差
        df_scaled = (df_numeric - df_numeric.mean()) / (df_numeric.std() + 1e-8)
        variances = df_scaled.var().sort_values(ascending=False)
        
        # 选择方差最大的特征，同时避免高相关性
        selected = []
        corr_matrix = df_numeric.corr().abs()
        
        for col in variances.index:
            if len(selected) >= max_features:
                break
            
            # 检查与已选特征的相关性
            is_redundant = False
            for s in selected:
                if corr_matrix.loc[col, s] > 0.85:  # 相关系数阈值
                    is_redundant = True
                    break
            
            if not is_redundant:
                selected.append(col)
        
        # 如果选的不够，补充高方差特征
        for col in variances.index:
            if len(selected) >= max_features:
                break
            if col not in selected:
                selected.append(col)
        
        return selected
    
    # ==================== 1. 缺失值处理 ====================
    
    def get_missing_info(self) -> pd.DataFrame:
        """
        获取缺失值统计信息
        
        Returns:
            包含缺失值统计的 DataFrame
        """
        total_rows = len(self.df)
        missing_count = self.df.isnull().sum()
        missing_percent = (missing_count / total_rows * 100).round(2)
        non_missing_count = total_rows - missing_count
        
        # 使用列名作为索引
        info = pd.DataFrame({
            'Column': self.df.columns,
            'Missing': missing_count.values,
            'Missing%': missing_percent.values,
            'Non-Missing': non_missing_count.values,
            'Type': self.df.dtypes.values.astype(str)
        })
        
        # 只返回有缺失值的列，按缺失比例排序
        info = info[info['Missing'] > 0].sort_values('Missing%', ascending=False)
        info = info.reset_index(drop=True)
        
        return info
    
    def plot_missing_matrix(self, figsize: Tuple[int, int] = (14, 10), 
                             max_features: int = 15) -> plt.Figure:
        """
        绘制缺失数据矩阵图（带列名和缺失比例）
        
        Args:
            figsize: 图形大小
            max_features: 最大显示特征数
            
        Returns:
            matplotlib Figure 对象
        """
        # 计算每列缺失比例
        missing_pct = (self.df.isnull().sum() / len(self.df) * 100).round(1)
        
        # 按缺失比例排序，优先显示有缺失的列
        sorted_cols = missing_pct.sort_values(ascending=False).index.tolist()
        
        # 选择要显示的列
        display_cols = sorted_cols[:max_features]
        
        # 创建带列名标注的标签
        col_labels = []
        for col in display_cols:
            pct = missing_pct[col]
            if pct > 0:
                col_labels.append(f"{col}\n({pct:.1f}%)")
            else:
                col_labels.append(col)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制矩阵
        plot_data = self.df[display_cols].isnull()
        
        # 使用清晰的配色
        cmap = plt.cm.colors.ListedColormap(['#2ecc71', '#e74c3c'])  # 绿=有数据, 红=缺失
        
        im = ax.imshow(plot_data.values, aspect='auto', cmap=cmap, interpolation='nearest')
        
        # 设置列标签
        ax.set_xticks(range(len(display_cols)))
        ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=9)
        
        # 设置行标签（只显示部分）
        n_rows = len(self.df)
        if n_rows > 20:
            yticks = np.linspace(0, n_rows-1, 10, dtype=int)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticks)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.5, ticks=[0.25, 0.75])
        cbar.ax.set_yticklabels(['Present', 'Missing'])
        
        # 添加缺失统计摘要
        total_missing = self.df.isnull().sum().sum()
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_ratio = total_missing / total_cells * 100
        
        title = get_display_text(
            f"缺失值矩阵图\n总缺失: {total_missing:,} / {total_cells:,} ({missing_ratio:.2f}%)",
            f"Missing Data Matrix\nTotal Missing: {total_missing:,} / {total_cells:,} ({missing_ratio:.2f}%)"
        )
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel(get_display_text("特征 (缺失比例)", "Features (Missing %)"), fontsize=11)
        ax.set_ylabel(get_display_text("样本索引", "Sample Index"), fontsize=11)
        
        plt.tight_layout()
        return fig
    
    def plot_missing_correlation(self, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        绘制缺失值相关性热力图
        
        Args:
            figsize: 图形大小
            
        Returns:
            matplotlib Figure 对象
        """
        # 只选择有缺失值的列
        missing_cols = self.df.columns[self.df.isnull().any()].tolist()
        if len(missing_cols) < 2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "缺失值列数少于2，无法计算相关性", 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig
        
        # 计算缺失值相关性
        missing_corr = self.df[missing_cols].isnull().corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(missing_corr, annot=True, cmap='coolwarm', 
                    center=0, vmin=-1, vmax=1, ax=ax)
        ax.set_title("缺失值相关性热力图", fontsize=14)
        plt.tight_layout()
        return fig
    
    def fill_missing(self, strategy: str = 'auto', columns: List[str] = None) -> 'DataProcessor':
        """
        填充缺失值
        
        Args:
            strategy: 填充策略
                - 'auto': 自动选择（数值用中位数，类别用众数）
                - 'median': 中位数填充
                - 'mean': 均值填充
                - 'mode': 众数填充
                - 'knn': KNN填充（仅数值型）
                - 'drop': 删除缺失行
            columns: 要处理的列，None表示所有列
            
        Returns:
            self，支持链式调用
        """
        cols = columns if columns else self.df.columns.tolist()
        
        for col in cols:
            if self.df[col].isnull().sum() == 0:
                continue
                
            if strategy == 'auto':
                if self.df[col].dtype in ['float64', 'int64']:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                    self.processing_log.append(f"列 '{col}' 使用中位数填充")
                else:
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                    self.processing_log.append(f"列 '{col}' 使用众数填充")
                    
            elif strategy == 'median':
                self.df[col] = self.df[col].fillna(self.df[col].median())
                self.processing_log.append(f"列 '{col}' 使用中位数填充")
                
            elif strategy == 'mean':
                self.df[col] = self.df[col].fillna(self.df[col].mean())
                self.processing_log.append(f"列 '{col}' 使用均值填充")
                
            elif strategy == 'mode':
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                self.processing_log.append(f"列 '{col}' 使用众数填充")
                
            elif strategy == 'drop':
                self.df.dropna(subset=[col], inplace=True)
                self.processing_log.append(f"删除列 '{col}' 的缺失行")
        
        if strategy == 'knn':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                imputer = KNNImputer(n_neighbors=5)
                self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])
                self.processing_log.append("使用KNN填充数值型缺失值")
        
        return self
    
    # ==================== 2. 异常值处理 ====================
    
    def detect_outliers_iqr(self, column: str) -> Dict:
        """
        使用IQR方法检测异常值
        
        Args:
            column: 列名
            
        Returns:
            包含异常值信息的字典
        """
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        
        return {
            'column': column,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_count': len(outliers),
            'outlier_percent': round(len(outliers) / len(self.df) * 100, 2)
        }
    
    def detect_outliers_zscore(self, column: str, threshold: float = 3.0) -> Dict:
        """
        使用Z-Score方法检测异常值
        
        Args:
            column: 列名
            threshold: Z-Score阈值，默认3
            
        Returns:
            包含异常值信息的字典
        """
        z_scores = np.abs(stats.zscore(self.df[column].dropna()))
        outlier_mask = z_scores > threshold
        
        return {
            'column': column,
            'threshold': threshold,
            'outlier_count': outlier_mask.sum(),
            'outlier_percent': round(outlier_mask.sum() / len(self.df) * 100, 2)
        }
    
    def plot_boxplot(self, columns: List[str] = None, figsize: Tuple[int, int] = (14, 8),
                      max_features: int = MAX_DISPLAY_FEATURES) -> plt.Figure:
        """
        绘制美观的箱线图（横向展示，自动选择代表性特征）
        
        Args:
            columns: 要绘制的列，None表示自动选择
            figsize: 图形大小
            max_features: 最大显示特征数
            
        Returns:
            matplotlib Figure 对象
        """
        if columns is None:
            all_numeric = self.df.select_dtypes(include=[np.number]).columns.tolist()
            columns = self._select_representative_features(all_numeric, max_features)
        
        n_cols = len(columns)
        if n_cols == 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No numeric columns to plot", ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # 使用横向箱线图，更美观
        fig, ax = plt.subplots(figsize=figsize)
        
        # 准备数据（标准化用于可视化，便于比较）
        df_plot = self.df[columns].copy()
        
        # 创建美观的箱线图
        colors = sns.color_palette("husl", n_cols)
        
        bp = ax.boxplot(
            [df_plot[col].dropna() for col in columns],
            tick_labels=columns,
            vert=False,  # 横向
            patch_artist=True,
            notch=True,  # 显示置信区间
            showfliers=True,
            flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.5),
            medianprops=dict(color='darkblue', linewidth=2),
            whiskerprops=dict(color='gray', linewidth=1.5),
            capprops=dict(color='gray', linewidth=1.5)
        )
        
        # 设置箱体颜色
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
        
        # 添加异常值数量标注
        for i, col in enumerate(columns):
            outlier_info = self.detect_outliers_iqr(col)
            outlier_count = outlier_info['outlier_count']
            if outlier_count > 0:
                ax.annotate(f'{outlier_count} outliers', 
                           xy=(df_plot[col].max() * 1.02, i + 1),
                           fontsize=9, color='red', alpha=0.8)
        
        ax.set_xlabel(get_display_text('数值', 'Value'), fontsize=12)
        ax.set_title(get_display_text(
            '箱线图 - 异常值检测\n(按方差选取代表性特征)', 
            'Box Plot - Outlier Detection\n(Top features by variance)'
        ), fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # 调整布局
        plt.tight_layout()
        return fig
    
    def plot_violin(self, columns: List[str] = None, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        绘制小提琴图
        
        Args:
            columns: 要绘制的列
            figsize: 图形大小
            
        Returns:
            matplotlib Figure 对象
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()[:6]
        
        n_cols = len(columns)
        if n_cols == 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "没有数值型列可绘制", ha='center', va='center')
            ax.axis('off')
            return fig
        
        fig, axes = plt.subplots(1, n_cols, figsize=figsize)
        axes = [axes] if n_cols == 1 else axes
        
        for i, col in enumerate(columns):
            sns.violinplot(y=self.df[col], ax=axes[i])
            axes[i].set_title(f'{col}')
        
        plt.suptitle("小提琴图 - 数据分布", fontsize=14)
        plt.tight_layout()
        return fig
    
    def handle_outliers(self, columns: List[str] = None, method: str = 'cap') -> 'DataProcessor':
        """
        处理异常值
        
        Args:
            columns: 要处理的列，None表示所有数值列
            method: 处理方法
                - 'cap': 盖帽法（截断到IQR边界）
                - 'drop': 删除异常值行
                - 'median': 用中位数替换
                
        Returns:
            self，支持链式调用
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            if method == 'cap':
                self.df[col] = self.df[col].clip(lower=lower, upper=upper)
                self.processing_log.append(f"列 '{col}' 使用盖帽法处理异常值")
                
            elif method == 'drop':
                mask = (self.df[col] >= lower) & (self.df[col] <= upper)
                self.df = self.df[mask]
                self.processing_log.append(f"删除列 '{col}' 的异常值行")
                
            elif method == 'median':
                median = self.df[col].median()
                mask = (self.df[col] < lower) | (self.df[col] > upper)
                self.df.loc[mask, col] = median
                self.processing_log.append(f"列 '{col}' 用中位数替换异常值")
        
        return self
    
    # ==================== 3. 数据分布校正 ====================
    
    def plot_distribution(self, columns: List[str] = None, figsize: Tuple[int, int] = (14, 10),
                           max_features: int = 6) -> plt.Figure:
        """
        绘制数据分布图（直方图 + KDE）
        
        Args:
            columns: 要绘制的列
            figsize: 图形大小
            max_features: 最大显示特征数
            
        Returns:
            matplotlib Figure 对象
        """
        if columns is None:
            all_numeric = self.df.select_dtypes(include=[np.number]).columns.tolist()
            columns = self._select_representative_features(all_numeric, max_features)
        
        n_cols = len(columns)
        if n_cols == 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No numeric columns to plot", ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        n_rows = (n_cols + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        colors = sns.color_palette("viridis", n_cols)
        
        for i, col in enumerate(columns):
            data = self.df[col].dropna()
            
            # 使用美观的直方图+KDE
            sns.histplot(data, kde=True, ax=axes[i], color=colors[i], 
                        edgecolor='white', alpha=0.7, linewidth=0.5)
            
            # 计算统计信息
            skewness = data.skew()
            mean_val = data.mean()
            
            # 添加均值线
            axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
            
            # 设置标题（使用英文避免字体问题）
            skew_status = "Right-skewed" if skewness > 0.5 else ("Left-skewed" if skewness < -0.5 else "Normal")
            axes[i].set_title(f'{col}\nSkewness: {skewness:.2f} ({skew_status})', fontsize=11)
            axes[i].set_xlabel('')
            axes[i].grid(axis='y', alpha=0.3, linestyle='--')
        
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle(get_display_text(
            "数据分布分析\n(按方差选取代表性特征)", 
            "Distribution Analysis\n(Top features by variance)"
        ), fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_qq(self, column: str, figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        绘制Q-Q图
        
        Args:
            column: 列名
            figsize: 图形大小
            
        Returns:
            matplotlib Figure 对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        stats.probplot(self.df[column].dropna(), dist="norm", plot=ax)
        ax.set_title(f"Q-Q图 - {column}")
        plt.tight_layout()
        return fig
    
    def transform_log(self, columns: List[str]) -> 'DataProcessor':
        """
        对数变换（适用于右偏数据）
        
        Args:
            columns: 要变换的列
            
        Returns:
            self，支持链式调用
        """
        for col in columns:
            # 确保数据为正数
            min_val = self.df[col].min()
            if min_val <= 0:
                shift = abs(min_val) + 1
                self.df[f'{col}_log'] = np.log1p(self.df[col] + shift)
            else:
                self.df[f'{col}_log'] = np.log1p(self.df[col])
            self.processing_log.append(f"列 '{col}' 进行对数变换，生成 '{col}_log'")
        
        return self
    
    def transform_boxcox(self, columns: List[str]) -> 'DataProcessor':
        """
        Box-Cox变换
        
        Args:
            columns: 要变换的列
            
        Returns:
            self，支持链式调用
        """
        for col in columns:
            # Box-Cox要求数据为正数
            data = self.df[col].dropna()
            if data.min() <= 0:
                data = data + abs(data.min()) + 1
            
            transformed, lambda_param = stats.boxcox(data)
            self.df[f'{col}_boxcox'] = np.nan
            self.df.loc[self.df[col].notna(), f'{col}_boxcox'] = transformed
            self.processing_log.append(f"列 '{col}' 进行Box-Cox变换 (λ={lambda_param:.2f})")
        
        return self
    
    # ==================== 4. 相关性分析 ====================
    
    def plot_correlation_heatmap(self, columns: List[str] = None, 
                                  figsize: Tuple[int, int] = (12, 10),
                                  max_features: int = 12) -> plt.Figure:
        """
        绘制美观的相关性热力图
        
        Args:
            columns: 要分析的列
            figsize: 图形大小
            max_features: 最大显示特征数
            
        Returns:
            matplotlib Figure 对象
        """
        if columns is None:
            all_numeric = self.df.select_dtypes(include=[np.number]).columns.tolist()
            columns = self._select_representative_features(all_numeric, max_features)
        
        if len(columns) > max_features:
            columns = columns[:max_features]
        
        corr = self.df[columns].corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # 使用更美观的配色
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        
        sns.heatmap(corr, mask=mask, annot=True, cmap=cmap, 
                    center=0, vmin=-1, vmax=1, ax=ax, fmt='.2f',
                    square=True, linewidths=0.5, linecolor='white',
                    annot_kws={"size": 9},
                    cbar_kws={"shrink": 0.8, "label": "Correlation"})
        
        ax.set_title(get_display_text(
            "特征相关性热力图\n(按方差选取代表性特征)", 
            "Feature Correlation Heatmap\n(Top features by variance)"
        ), fontsize=14, fontweight='bold', pad=20)
        
        # 旋转标签使其更易读
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        return fig
    
    def plot_pairplot(self, columns: List[str] = None, hue: str = None) -> plt.Figure:
        """
        绘制配对图
        
        Args:
            columns: 要绘制的列（建议不超过5个）
            hue: 用于分组的列
            
        Returns:
            matplotlib Figure 对象
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()[:4]
        
        if hue and hue not in columns:
            plot_data = self.df[columns + [hue]]
        else:
            plot_data = self.df[columns]
        
        fig = sns.pairplot(plot_data, hue=hue, diag_kind='kde')
        fig.fig.suptitle("配对散点图", y=1.02)
        return fig.fig
    
    def get_high_correlation_pairs(self, threshold: float = 0.8) -> pd.DataFrame:
        """
        获取高相关性特征对
        
        Args:
            threshold: 相关系数阈值
            
        Returns:
            高相关性特征对的 DataFrame
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr = self.df[numeric_cols].corr()
        
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                if abs(corr.iloc[i, j]) >= threshold:
                    pairs.append({
                        '特征1': corr.columns[i],
                        '特征2': corr.columns[j],
                        '相关系数': round(corr.iloc[i, j], 3)
                    })
        
        return pd.DataFrame(pairs).sort_values('相关系数', key=abs, ascending=False)
    
    # ==================== 辅助方法 ====================
    
    def get_processing_log(self) -> List[str]:
        """获取处理日志"""
        return self.processing_log
    
    def get_summary(self) -> pd.DataFrame:
        """获取数据摘要"""
        summary = self.df.describe(include='all').T
        summary['缺失值'] = self.df.isnull().sum()
        summary['缺失率(%)'] = (self.df.isnull().sum() / len(self.df) * 100).round(2)
        return summary


# ==================== 便捷函数 ====================

def load_and_process(filepath: str, 
                     fill_missing: str = 'auto',
                     handle_outliers: str = 'cap') -> Tuple[pd.DataFrame, DataProcessor]:
    """
    一键加载和预处理数据
    
    Args:
        filepath: 文件路径（支持 csv, xlsx）
        fill_missing: 缺失值处理策略
        handle_outliers: 异常值处理策略
        
    Returns:
        (处理后的DataFrame, DataProcessor对象)
    """
    # 加载数据
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    else:
        raise ValueError("仅支持 csv 和 xlsx 格式")
    
    # 创建处理器并处理
    processor = DataProcessor(df)
    processor.fill_missing(strategy=fill_missing)
    processor.handle_outliers(method=handle_outliers)
    
    return processor.get_data(), processor

