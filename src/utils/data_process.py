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
                      max_features: int = MAX_DISPLAY_FEATURES, 
                      normalize: bool = True) -> plt.Figure:
        """
        绘制美观的箱线图（横向展示，自动选择代表性特征）
        
        Args:
            columns: 要绘制的列，None表示自动选择
            figsize: 图形大小
            max_features: 最大显示特征数
            normalize: 是否归一化数据（便于不同尺度特征比较）
            
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
        
        # 准备数据
        df_plot = self.df[columns].copy()
        
        # 统计原始异常值信息（在归一化之前）
        outlier_counts = {}
        for col in columns:
            outlier_info = self.detect_outliers_iqr(col)
            outlier_counts[col] = outlier_info['outlier_count']
        
        # 归一化处理（Min-Max 标准化到 0-1）
        if normalize:
            for col in columns:
                col_min = df_plot[col].min()
                col_max = df_plot[col].max()
                if col_max > col_min:
                    df_plot[col] = (df_plot[col] - col_min) / (col_max - col_min)
                else:
                    df_plot[col] = 0.5  # 常量列
        
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
        
        # 添加异常值数量标注（使用原始数据的统计）
        for i, col in enumerate(columns):
            outlier_count = outlier_counts[col]
            if outlier_count > 0:
                # 标注在图的右侧
                ax.annotate(f'{outlier_count}', 
                           xy=(1.05, i + 1),
                           fontsize=9, color='red', alpha=0.8,
                           fontweight='bold')
        
        # 设置标签
        xlabel = get_display_text('归一化数值 (0-1)', 'Normalized Value (0-1)') if normalize else get_display_text('数值', 'Value')
        ax.set_xlabel(xlabel, fontsize=12)
        
        title = get_display_text(
            '箱线图 - 异常值检测\n(已归一化，红色数字为异常值数量)', 
            'Box Plot - Outlier Detection\n(Normalized, red numbers = outlier count)'
        ) if normalize else get_display_text(
            '箱线图 - 异常值检测', 
            'Box Plot - Outlier Detection'
        )
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        if normalize:
            ax.set_xlim(-0.05, 1.15)  # 留出标注空间
        
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
    
    def encode_categorical(self, strategy: str = 'auto', 
                           columns: List[str] = None,
                           target_column: str = None) -> 'DataProcessor':
        """
        编码分类变量（字符串类型转换为数值）
        
        Args:
            strategy: 编码策略
                - 'auto': 自动选择（唯一值≤2用label，3-10用onehot，>10用label）
                - 'label': 标签编码（0,1,2,3...）
                - 'onehot': 独热编码（每个类别一列）
                - 'target': 目标编码（用目标变量均值替换，需要target_column）
            columns: 要编码的列，None表示所有字符串/对象类型列
            target_column: 目标列名（仅target编码需要）
            
        Returns:
            self，支持链式调用
        """
        from sklearn.preprocessing import LabelEncoder
        
        # 获取需要编码的列
        if columns is None:
            columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            # 排除目标列
            if target_column and target_column in columns:
                columns.remove(target_column)
        
        if not columns:
            self.processing_log.append("没有需要编码的分类列")
            return self
        
        encoded_info = []
        
        for col in columns:
            unique_count = self.df[col].nunique()
            
            # 确定使用的策略
            if strategy == 'auto':
                if unique_count <= 2:
                    col_strategy = 'label'
                elif unique_count <= 10:
                    col_strategy = 'onehot'
                else:
                    col_strategy = 'label'  # 高基数用 label，避免维度爆炸
            else:
                col_strategy = strategy
            
            # 执行编码
            if col_strategy == 'label':
                le = LabelEncoder()
                # 处理缺失值
                mask = self.df[col].notna()
                self.df.loc[mask, col] = le.fit_transform(self.df.loc[mask, col].astype(str))
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                encoded_info.append(f"'{col}' (Label, {unique_count}类)")
                
            elif col_strategy == 'onehot':
                # 独热编码
                dummies = pd.get_dummies(self.df[col], prefix=col, dummy_na=False)
                self.df = pd.concat([self.df.drop(col, axis=1), dummies], axis=1)
                encoded_info.append(f"'{col}' (OneHot, {unique_count}类→{len(dummies.columns)}列)")
                
            elif col_strategy == 'target' and target_column:
                # 目标编码
                if target_column in self.df.columns:
                    target_mean = self.df.groupby(col)[target_column].mean()
                    self.df[col] = self.df[col].map(target_mean)
                    encoded_info.append(f"'{col}' (Target, {unique_count}类)")
                else:
                    # fallback to label
                    le = LabelEncoder()
                    mask = self.df[col].notna()
                    self.df.loc[mask, col] = le.fit_transform(self.df.loc[mask, col].astype(str))
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    encoded_info.append(f"'{col}' (Label-fallback, {unique_count}类)")
        
        if encoded_info:
            self.processing_log.append(f"分类变量编码: {', '.join(encoded_info)}")
        
        return self
    
    def get_categorical_columns(self) -> Dict[str, int]:
        """
        获取所有分类列及其唯一值数量
        
        Returns:
            {列名: 唯一值数量}
        """
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        return {col: self.df[col].nunique() for col in cat_cols}
    
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
    
    def get_high_correlation_pairs(
        self, 
        threshold: float = None, 
        top_k: int = None,
        adaptive: bool = True,
        min_pairs: int = 5
    ) -> pd.DataFrame:
        """
        获取高相关性特征对
        
        Args:
            threshold: 相关系数阈值（None 时使用自适应）
            top_k: 返回前 k 个最高相关性对（优先于 threshold）
            adaptive: 是否使用自适应阈值（当 threshold 和 top_k 都为 None 时）
            min_pairs: 自适应模式下最少返回的特征对数量
            
        Returns:
            高相关性特征对的 DataFrame
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return pd.DataFrame(columns=['特征1', '特征2', '相关系数'])
        
        corr = self.df[numeric_cols].corr()
        
        # 收集所有特征对及其相关系数
        all_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                corr_val = corr.iloc[i, j]
                if not np.isnan(corr_val):
                    all_pairs.append({
                        '特征1': corr.columns[i],
                        '特征2': corr.columns[j],
                        '相关系数': round(corr_val, 3)
                    })
        
        if not all_pairs:
            return pd.DataFrame(columns=['特征1', '特征2', '相关系数'])
        
        # 按绝对值排序
        df_pairs = pd.DataFrame(all_pairs)
        df_pairs = df_pairs.sort_values('相关系数', key=abs, ascending=False).reset_index(drop=True)
        
        # 模式1: top_k - 返回前 k 个
        if top_k is not None:
            return df_pairs.head(top_k)
        
        # 模式2: 固定阈值
        if threshold is not None:
            result = df_pairs[df_pairs['相关系数'].abs() >= threshold]
            # 如果结果为空且开启自适应，降级到自适应模式
            if len(result) == 0 and adaptive:
                pass  # 继续到自适应模式
            else:
                return result
        
        # 模式3: 自适应阈值
        if adaptive:
            # 确保至少返回 min_pairs 个结果
            if len(df_pairs) <= min_pairs:
                return df_pairs
            
            # 计算自适应阈值：取前 min_pairs 个的最小绝对值，或者使用分位数
            abs_corrs = df_pairs['相关系数'].abs()
            
            # 策略1: 使用 75 分位数作为阈值
            q75_threshold = abs_corrs.quantile(0.75)
            
            # 策略2: 确保至少有 min_pairs 个结果
            if len(df_pairs) > min_pairs:
                min_threshold = abs_corrs.iloc[min_pairs - 1]
            else:
                min_threshold = abs_corrs.min()
            
            # 取两者中较小的（返回更多结果）
            adaptive_threshold = min(q75_threshold, min_threshold)
            
            result = df_pairs[abs_corrs >= adaptive_threshold]
            
            # 兜底：如果仍然太少，返回前 min_pairs 个
            if len(result) < min_pairs:
                return df_pairs.head(min_pairs)
            
            return result
        
        # 默认返回所有
        return df_pairs
    
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
    
    def generate_report(self, include_recommendations: bool = True) -> dict:
        """
        生成数据分析报告
        
        Args:
            include_recommendations: 是否包含处理建议
            
        Returns:
            包含报告各部分的字典
        """
        report = {
            'overview': {},
            'missing_analysis': {},
            'outlier_analysis': {},
            'distribution_analysis': {},
            'correlation_analysis': {},
            'recommendations': [],
            'processing_log': self.processing_log,
            'markdown': '',
            'llm_prompt': ''
        }
        
        # 1. 数据概览
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        report['overview'] = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(categorical_cols),
            'memory_usage_mb': round(self.df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            'column_names': self.df.columns.tolist()
        }
        
        # 2. 缺失值分析
        missing_info = self.get_missing_info()
        missing_cols = missing_info[missing_info['Missing'] > 0]
        report['missing_analysis'] = {
            'total_missing_cells': int(self.df.isnull().sum().sum()),
            'missing_rate': round(self.df.isnull().sum().sum() / self.df.size * 100, 2),
            'columns_with_missing': len(missing_cols),
            'details': missing_cols.to_dict('records') if len(missing_cols) > 0 else []
        }
        
        # 3. 异常值分析
        outlier_info = []
        for col in numeric_cols:
            outlier_result = self.detect_outliers_iqr(col)
            outlier_count = outlier_result['outlier_count']
            if outlier_count > 0:
                # 获取异常值的实际数据
                lower_bound = outlier_result['lower_bound']
                upper_bound = outlier_result['upper_bound']
                outlier_values = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)][col]
                
                outlier_info.append({
                    'column': col,
                    'outlier_count': outlier_count,
                    'outlier_rate': round(outlier_count / len(self.df) * 100, 2),
                    'min_outlier': round(float(outlier_values.min()), 3) if len(outlier_values) > 0 else None,
                    'max_outlier': round(float(outlier_values.max()), 3) if len(outlier_values) > 0 else None
                })
        report['outlier_analysis'] = {
            'columns_with_outliers': len(outlier_info),
            'details': outlier_info
        }
        
        # 4. 分布分析
        dist_info = []
        for col in numeric_cols[:10]:  # 限制数量
            try:
                skewness = float(self.df[col].skew())
                kurtosis = float(self.df[col].kurtosis())
                dist_info.append({
                    'column': col,
                    'mean': round(self.df[col].mean(), 3),
                    'std': round(self.df[col].std(), 3),
                    'skewness': round(skewness, 3),
                    'kurtosis': round(kurtosis, 3),
                    'distribution_type': '右偏' if skewness > 1 else ('左偏' if skewness < -1 else '近似正态')
                })
            except:
                pass
        report['distribution_analysis'] = {
            'analyzed_columns': len(dist_info),
            'details': dist_info
        }
        
        # 5. 相关性分析
        corr_pairs = self.get_high_correlation_pairs(top_k=10)
        report['correlation_analysis'] = {
            'high_correlation_pairs': corr_pairs.to_dict('records') if len(corr_pairs) > 0 else [],
            'max_correlation': round(corr_pairs['相关系数'].abs().max(), 3) if len(corr_pairs) > 0 else 0
        }
        
        # 6. 生成建议
        if include_recommendations:
            recommendations = []
            
            # 缺失值建议
            if report['missing_analysis']['missing_rate'] > 30:
                recommendations.append("⚠️ 数据缺失率较高(>30%)，建议检查数据来源或考虑删除高缺失列")
            elif report['missing_analysis']['missing_rate'] > 5:
                recommendations.append("💡 存在一定缺失值，建议使用 KNN 或中位数填充")
            
            # 异常值建议
            if len(outlier_info) > len(numeric_cols) * 0.5:
                recommendations.append("⚠️ 多数数值列存在异常值，建议检查数据质量或使用截断处理")
            
            # 分布建议
            skewed_cols = [d['column'] for d in dist_info if abs(d.get('skewness', 0)) > 1]
            if skewed_cols:
                recommendations.append(f"💡 以下列分布偏斜，建议进行对数或Box-Cox变换: {', '.join(skewed_cols[:5])}")
            
            # 相关性建议
            if report['correlation_analysis']['max_correlation'] > 0.9:
                recommendations.append("⚠️ 存在高度相关特征(>0.9)，建议进行特征选择或PCA降维")
            
            if not recommendations:
                recommendations.append("✅ 数据质量良好，可直接用于建模")
            
            report['recommendations'] = recommendations
        
        # 7. 生成 Markdown 格式报告
        report['markdown'] = self._generate_markdown_report(report)
        
        # 8. 生成 LLM 分析提示词
        report['llm_prompt'] = self._generate_llm_prompt(report)
        
        return report
    
    def _generate_markdown_report(self, report: dict) -> str:
        """生成 Markdown 格式的报告"""
        md = []
        md.append("# 📊 数据分析报告\n")
        
        # 概览
        ov = report['overview']
        md.append("## 1. 数据概览")
        md.append(f"- **总行数**: {ov['total_rows']:,}")
        md.append(f"- **总列数**: {ov['total_columns']}")
        md.append(f"- **数值列**: {ov['numeric_columns']} 列")
        md.append(f"- **分类列**: {ov['categorical_columns']} 列")
        md.append(f"- **内存占用**: {ov['memory_usage_mb']} MB\n")
        
        # 缺失值
        ma = report['missing_analysis']
        md.append("## 2. 缺失值分析")
        md.append(f"- **总缺失单元格**: {ma['total_missing_cells']:,}")
        md.append(f"- **整体缺失率**: {ma['missing_rate']}%")
        md.append(f"- **含缺失值的列数**: {ma['columns_with_missing']}")
        if ma['details']:
            md.append("\n| 列名 | 缺失数量 | 缺失率 |")
            md.append("|------|----------|--------|")
            for d in ma['details'][:10]:
                md.append(f"| {d.get('Column', 'N/A')} | {d.get('Missing', 0)} | {d.get('Missing%', 0)}% |")
        md.append("")
        
        # 异常值
        oa = report['outlier_analysis']
        md.append("## 3. 异常值分析")
        md.append(f"- **含异常值的列数**: {oa['columns_with_outliers']}")
        if oa['details']:
            md.append("\n| 列名 | 异常值数量 | 异常率 |")
            md.append("|------|------------|--------|")
            for d in oa['details'][:10]:
                md.append(f"| {d['column']} | {d['outlier_count']} | {d['outlier_rate']}% |")
        md.append("")
        
        # 分布
        da = report['distribution_analysis']
        md.append("## 4. 数据分布")
        if da['details']:
            md.append("\n| 列名 | 均值 | 标准差 | 偏度 | 分布类型 |")
            md.append("|------|------|--------|------|----------|")
            for d in da['details'][:10]:
                md.append(f"| {d['column']} | {d['mean']} | {d['std']} | {d['skewness']} | {d['distribution_type']} |")
        md.append("")
        
        # 相关性
        ca = report['correlation_analysis']
        md.append("## 5. 相关性分析")
        md.append(f"- **最高相关系数**: {ca['max_correlation']}")
        if ca['high_correlation_pairs']:
            md.append("\n| 特征1 | 特征2 | 相关系数 |")
            md.append("|-------|-------|----------|")
            for d in ca['high_correlation_pairs'][:10]:
                md.append(f"| {d['特征1']} | {d['特征2']} | {d['相关系数']} |")
        md.append("")
        
        # 建议
        if report['recommendations']:
            md.append("## 6. 处理建议")
            for rec in report['recommendations']:
                md.append(f"- {rec}")
        md.append("")
        
        # 处理日志
        if report['processing_log']:
            md.append("## 7. 处理日志")
            for log in report['processing_log']:
                md.append(f"- {log}")
        
        return "\n".join(md)
    
    def _generate_llm_prompt(self, report: dict) -> str:
        """生成用于 LLM 分析的提示词"""
        prompt = []
        prompt.append("请分析以下数据集的特征，并给出专业的数据处理和建模建议：\n")
        
        # 数据概览
        ov = report['overview']
        prompt.append(f"【数据规模】{ov['total_rows']} 行 × {ov['total_columns']} 列")
        prompt.append(f"【列类型】数值列 {ov['numeric_columns']} 个，分类列 {ov['categorical_columns']} 个")
        prompt.append(f"【列名】{', '.join(ov['column_names'][:20])}{'...' if len(ov['column_names']) > 20 else ''}\n")
        
        # 数据质量
        ma = report['missing_analysis']
        prompt.append(f"【缺失情况】整体缺失率 {ma['missing_rate']}%，{ma['columns_with_missing']} 列有缺失")
        
        oa = report['outlier_analysis']
        prompt.append(f"【异常值】{oa['columns_with_outliers']} 列检测到异常值")
        
        # 分布特征
        da = report['distribution_analysis']
        skewed = [d['column'] for d in da['details'] if abs(d.get('skewness', 0)) > 1]
        if skewed:
            prompt.append(f"【偏斜分布】{', '.join(skewed)}")
        
        # 相关性
        ca = report['correlation_analysis']
        if ca['high_correlation_pairs']:
            high_corr = [f"{d['特征1']}-{d['特征2']}({d['相关系数']})" for d in ca['high_correlation_pairs'][:5]]
            prompt.append(f"【高相关特征对】{'; '.join(high_corr)}")
        
        prompt.append("\n请基于以上信息：")
        prompt.append("1. 评估数据质量和潜在问题")
        prompt.append("2. 推荐数据预处理步骤")
        prompt.append("3. 建议适合的机器学习模型")
        prompt.append("4. 提供特征工程建议")
        
        return "\n".join(prompt)


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

