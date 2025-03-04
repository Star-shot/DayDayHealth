import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
from plot import *

class TestPlotFunctions(unittest.TestCase):
    def setUp(self):
        # 初始化测试数据
        self.test_data = np.random.rand(5)
        self.test_labels = ['A', 'B', 'C', 'D', 'E']
        self.test_matrix = np.random.rand(5, 5)
        
        # 创建输出目录
        self.output_dir = "test_output"
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        # 保存当前测试的图形并清理
        if hasattr(self, '_testMethodName'):
            filename = os.path.join(self.output_dir, f"{self._testMethodName}.png")
            plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close('all')  # 关闭所有图形

    def test_plot_pie_chart(self):
        fig, _ = plot_pie_chart(self.test_data, self.test_labels)
        self.assertIsNotNone(fig)

    def test_plot_histogram(self):
        fig, _ = plot_histogram(np.random.randn(100))
        self.assertIsNotNone(fig)

    def test_plot_box_plot(self):
        data = [np.random.randn(50) for _ in range(3)]
        fig, _ = plot_box_plot(data)
        self.assertIsNotNone(fig)

    def test_plot_circos_plot(self):
        fig, _ = plot_circos_plot(self.test_labels, self.test_matrix)
        self.assertIsNotNone(fig)

    def test_plot_network_graph(self):
        fig, _ = plot_network_graph()
        self.assertIsNotNone(fig)

    def test_plot_rose_diagram(self):
        fig, _ = plot_rose_diagram(np.random.rand(100))
        self.assertIsNotNone(fig)

    def test_plot_line_chart(self):
        x = np.linspace(0, 10, 100)
        y = [np.sin(x), np.cos(x)]
        fig, _ = plot_line_chart(x, y)
        self.assertIsNotNone(fig)

    def test_plot_scatter_plot(self):
        x = np.random.randn(100)
        y = np.random.randn(100)
        fig, _ = plot_scatter_plot(x, y)
        self.assertIsNotNone(fig)

    def test_plot_heatmap(self):
        data = np.random.rand(5, 5)
        fig, _ = plot_heatmap(data, self.test_labels, self.test_labels)
        self.assertIsNotNone(fig)

    # def test_plot_radar_chart(self):
    #     # 确保数据长度与标签一致
    #     data = np.random.rand(len(self.test_labels))
    #     fig, _ = plot_radar_chart(self.test_labels, data)
    #     self.assertIsNotNone(fig)

    def test_plot_area_chart(self):
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        fig, _ = plot_area_chart(x, y)
        self.assertIsNotNone(fig)

    def test_plot_bubble_chart(self):
        x = np.random.randn(50)
        y = np.random.randn(50)
        sizes = np.random.randint(10, 100, 50)
        fig, _ = plot_bubble_chart(x, y, sizes, sizes)
        self.assertIsNotNone(fig)

    def test_plot_violin_plot(self):
        data = [np.random.randn(50) for _ in range(3)]
        fig, _ = plot_violin_plot(data)
        self.assertIsNotNone(fig)

    def test_plot_stacked_bar_chart(self):
        labels = ['G1', 'G2', 'G3']
        data = [np.random.randint(1, 5, 3) for _ in range(3)]
        fig, _ = plot_stacked_bar_chart(labels, data, ['S1', 'S2', 'S3'])
        self.assertIsNotNone(fig)

    def test_plot_3d_scatter(self):
        x = np.random.randn(50)
        y = np.random.randn(50)
        z = np.random.randn(50)
        fig, _ = plot_3d_scatter(x, y, z)
        self.assertIsNotNone(fig)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
    # 所有测试完成后清理图形
    plt.close('all')
