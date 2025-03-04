import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

# 基础图表函数
def plot_pie_chart(data, labels, title=None, colors=None, cmap='viridis', autopct='%1.1f%%', startangle=90):
    """
    绘制饼图
    :param data: 数值数据列表
    :param labels: 标签列表
    :param title: 图表标题
    :param colors: 自定义颜色列表
    :param cmap: 颜色映射名称
    :param autopct: 百分比格式
    :param startangle: 起始角度
    :return: fig, ax对象
    """
    if colors is None:
        colors = colormaps[cmap](np.linspace(0, 1, len(data)))
    fig, ax = plt.subplots()
    ax.pie(data, labels=labels, colors=colors, autopct=autopct, startangle=startangle)
    ax.set_title(title)
    return fig, ax

def plot_histogram(data, bins=10, title=None, xlabel=None, ylabel=None, density=False, cmap='plasma', edgecolor='black'):
    """
    绘制直方图
    :param data: 输入数据
    :param bins: 分箱数量
    :param title: 图表标题
    :param xlabel: x轴标签
    :param ylabel: y轴标签
    :param density: 是否标准化
    :param cmap: 颜色映射
    :param edgecolor: 边框颜色
    :return: fig, ax对象
    """
    fig, ax = plt.subplots()
    cmap = colormaps[cmap]
    n, bins, patches = ax.hist(data, bins=bins, density=density, edgecolor=edgecolor)
    for i in range(len(patches)):
        patches[i].set_facecolor(cmap(i/len(patches)))
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return fig, ax

def plot_box_plot(data, labels=None, title=None, colors=None, cmap='tab20'):
    """
    绘制箱线图
    :param data: 多维数据列表
    :param labels: 分类标签
    :param title: 图表标题
    :param colors: 颜色列表
    :param cmap: 颜色映射
    :return: fig, ax对象
    """
    fig, ax = plt.subplots()
    if colors is None:
        colors = colormaps[cmap](np.linspace(0, 1, len(data)))
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_title(title)
    return fig, ax

def plot_circos_plot(categories, matrix, title=None, cmap='viridis'):
    """
    绘制环形布局图（基础实现）
    :param categories: 类别列表
    :param matrix: 连接矩阵
    :param title: 图表标题
    :param cmap: 颜色映射
    :return: fig, ax对象
    """
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    n = len(categories)
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    width = 2 * np.pi / n
    values = np.random.rand(n) * 100  # 示例数据
    colors = colormaps[cmap](values/100)
    bars = ax.bar(theta, values, width=width, color=colors)
    ax.set_title(title)
    return fig, ax

def plot_network_graph(G=None, pos=None, node_size=300, node_color='skyblue', 
                      edge_color='gray', cmap='cool', title=None):
    """
    绘制网络图
    :param G: NetworkX图对象
    :param pos: 布局位置
    :param node_size: 节点大小
    :param node_color: 节点颜色
    :param edge_color: 边颜色
    :param cmap: 颜色映射
    :param title: 图表标题
    :return: fig, ax对象
    """
    fig, ax = plt.subplots()
    if G is None:
        G = nx.erdos_renyi_graph(10, 0.3)
    if pos is None:
        pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=node_size, 
            node_color=node_color, edge_color=edge_color, 
            cmap=cmap, ax=ax)
    ax.set_title(title)
    return fig, ax

def plot_rose_diagram(data, sectors=12, title=None, cmap='plasma'):
    """
    绘制玫瑰图
    :param data: 输入数据
    :param sectors: 扇区数量
    :param title: 图表标题
    :param cmap: 颜色映射
    :return: fig, ax对象
    """
    theta = np.linspace(0, 2*np.pi, sectors, endpoint=False)
    values = np.histogram(data, bins=sectors)[0]
    width = 2 * np.pi / sectors
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    colors = colormaps[cmap](values / max(values))
    bars = ax.bar(theta, values, width=width, color=colors)
    ax.set_title(title)
    return fig, ax

def plot_line_chart(x, y, title=None, xlabel=None, ylabel=None, 
                   line_color=None, cmap='viridis', linewidth=2):
    """
    绘制折线图
    :param x: x轴数据
    :param y: y轴数据（支持多维）
    :param title: 图表标题
    :param xlabel: x轴标签
    :param ylabel: y轴标签
    :param line_color: 线条颜色
    :param cmap: 颜色映射
    :param linewidth: 线宽
    :return: fig, ax对象
    """
    fig, ax = plt.subplots()
    if line_color is None:
        line_color = colormaps[cmap](np.linspace(0, 1, len(y)))
    for i in range(len(y)):
        ax.plot(x, y[i], color=line_color[i], linewidth=linewidth)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return fig, ax

def plot_scatter_plot(x, y, sizes=None, colors=None, title=None, 
                     xlabel=None, ylabel=None, cmap='viridis', alpha=0.6):
    """
    绘制散点图
    :param x: x轴数据
    :param y: y轴数据
    :param sizes: 点的大小数组
    :param colors: 颜色数据
    :param title: 图表标题
    :param xlabel: x轴标签
    :param ylabel: y轴标签
    :param cmap: 颜色映射
    :param alpha: 透明度
    :return: fig, ax对象
    """
    fig, ax = plt.subplots()
    sc = ax.scatter(x, y, s=sizes, c=colors, cmap=cmap, alpha=alpha)
    if colors is not None:
        plt.colorbar(sc)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return fig, ax

def plot_heatmap(data, row_labels, col_labels, title=None, 
                cmap='magma', cbar_label='Value'):
    """
    绘制热力图
    :param data: 二维数据数组
    :param row_labels: 行标签
    :param col_labels: 列标签
    :param title: 图表标题
    :param cmap: 颜色映射
    :param cbar_label: 颜色条标签
    :return: fig, ax对象
    """
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap=cmap)
    cbar = fig.colorbar(im)
    cbar.set_label(cbar_label)
    ax.set(title=title, 
           xticks=np.arange(len(col_labels)), 
           yticks=np.arange(len(row_labels)),
           xticklabels=col_labels, 
           yticklabels=row_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    return fig, ax

def plot_radar_chart(categories, values, title=None, cmap='tab20'):
    """
    绘制雷达图
    :param categories: 类别标签列表
    :param values: 数值列表
    :param title: 图表标题
    :param cmap: 颜色映射
    :return: fig, ax对象
    """
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.fill(angles, values, color=colormaps[cmap](0.3), alpha=0.4)
    ax.plot(angles, values, color=colormaps[cmap](0.5), linewidth=2)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_title(title)
    return fig, ax

# 额外添加的5个专业图表
def plot_area_chart(x, y, title=None, xlabel=None, ylabel=None, cmap='Blues'):
    """
    绘制面积图
    :param x: x轴数据
    :param y: y轴数据
    :param title: 图表标题
    :param xlabel: x轴标签
    :param ylabel: y轴标签
    :param cmap: 颜色映射
    :return: fig, ax对象
    """
    fig, ax = plt.subplots()
    color = colormaps[cmap](0.5)
    ax.fill_between(x, y, color=color, alpha=0.4)
    ax.plot(x, y, color=color, alpha=0.8)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return fig, ax

def plot_bubble_chart(x, y, sizes, colors, title=None, cmap='viridis'):
    """
    绘制气泡图
    :param x: x轴数据
    :param y: y轴数据
    :param sizes: 气泡大小数组
    :param colors: 颜色数据
    :param title: 图表标题
    :param cmap: 颜色映射
    :return: fig, ax对象
    """
    fig, ax = plt.subplots()
    sc = ax.scatter(x, y, s=sizes, c=colors, cmap=cmap, alpha=0.6)
    plt.colorbar(sc)
    ax.set_title(title)
    return fig, ax

def plot_violin_plot(data, labels=None, title=None, cmap='coolwarm'):
    """
    绘制小提琴图
    :param data: 多维数据列表
    :param labels: 分类标签
    :param title: 图表标题
    :param cmap: 颜色映射
    :return: fig, ax对象
    """
    fig, ax = plt.subplots()
    cmap = colormaps[cmap]
    parts = ax.violinplot(data, showmeans=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(cmap(i/len(data)))
        pc.set_edgecolor('black')
    ax.set_xticks(np.arange(1, len(data)+1))
    if labels:
        ax.set_xticklabels(labels)
    ax.set_title(title)
    return fig, ax

def plot_stacked_bar_chart(labels, data_series, series_labels, title=None, cmap='tab20'):
    """
    绘制堆叠柱状图
    :param labels: x轴标签列表
    :param data_series: 多维数据（每个系列的数据）
    :param series_labels: 系列标签
    :param title: 图表标题
    :param cmap: 颜色映射
    :return: fig, ax对象
    """
    fig, ax = plt.subplots()
    colors = colormaps[cmap](np.linspace(0, 1, len(series_labels)))
    bottom = np.zeros(len(labels))
    for i, data in enumerate(data_series):
        ax.bar(labels, data, bottom=bottom, label=series_labels[i], color=colors[i])
        bottom += data
    ax.set_title(title)
    ax.legend()
    return fig, ax

def plot_3d_scatter(x, y, z, colors=None, title=None, cmap='viridis'):
    """
    绘制3D散点图
    :param x: x轴数据
    :param y: y轴数据
    :param z: z轴数据
    :param colors: 颜色数据
    :param title: 图表标题
    :param cmap: 颜色映射
    :return: fig, ax对象
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=colors, cmap=cmap)
    if colors is not None:
        fig.colorbar(sc)
    ax.set_title(title)
    return fig, ax