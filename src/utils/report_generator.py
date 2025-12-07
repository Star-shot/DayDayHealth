"""
报告生成工具
用于生成训练和评估的详细报告（Markdown格式）
"""
import os
import json
import zipfile
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


class ReportGenerator:
    """生成训练和评估报告（Markdown格式，包含图表）"""
    
    def __init__(self, output_dir: str = "reports"):
        """
        初始化报告生成器
        
        参数:
            output_dir: 报告保存目录
        """
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.temp_dir = os.path.join(output_dir, "temp")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # 保存最近生成的报告内容
        self.last_training_report = ""
        self.last_evaluation_report = ""
    
    def get_last_training_report(self) -> str:
        """获取最近生成的训练报告内容"""
        return self.last_training_report
    
    def get_last_evaluation_report(self) -> str:
        """获取最近生成的评估报告内容"""
        return self.last_evaluation_report
    
    def _create_zip_package(self, md_file: str, image_files: List[str], report_name: str) -> str:
        """
        创建包含报告和图片的zip文件
        
        参数:
            md_file: markdown报告文件路径
            image_files: 图片文件列表
            report_name: 报告名称
        
        返回:
            zip文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"{report_name}_{timestamp}.zip"
        zip_filepath = os.path.join(self.output_dir, zip_filename)
        
        # 创建临时目录结构
        temp_report_dir = os.path.join(self.temp_dir, f"{report_name}_{timestamp}")
        temp_images_dir = os.path.join(temp_report_dir, "images")
        os.makedirs(temp_images_dir, exist_ok=True)
        
        # 复制markdown文件
        temp_md_path = os.path.join(temp_report_dir, os.path.basename(md_file))
        shutil.copy(md_file, temp_md_path)
        
        # 复制所有图片文件
        for img_file in image_files:
            if os.path.exists(img_file):
                shutil.copy(img_file, os.path.join(temp_images_dir, os.path.basename(img_file)))
        
        # 创建zip文件
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 添加markdown文件
            zipf.write(temp_md_path, os.path.basename(md_file))
            
            # 添加所有图片
            for img_file in os.listdir(temp_images_dir):
                img_path = os.path.join(temp_images_dir, img_file)
                zipf.write(img_path, os.path.join("images", img_file))
        
        # 清理临时目录
        shutil.rmtree(temp_report_dir)
        
        return zip_filepath
    
    def _save_class_distribution_chart(self, class_distribution: Dict, filename: str) -> str:
        """
        保存类别分布柱状图
        
        参数:
            class_distribution: 类别分布字典
            filename: 文件名
        
        返回:
            图片相对路径
        """
        plt.figure(figsize=(8, 5))
        classes = list(class_distribution.keys())
        counts = list(class_distribution.values())
        
        plt.bar(range(len(classes)), counts, color='steelblue', alpha=0.8)
        plt.xlabel('类别', fontsize=12)
        plt.ylabel('样本数', fontsize=12)
        plt.title('类别分布', fontsize=14, fontweight='bold')
        plt.xticks(range(len(classes)), classes)
        plt.grid(axis='y', alpha=0.3)
        
        # 在柱子上添加数值标签
        for i, v in enumerate(counts):
            plt.text(i, v + max(counts)*0.01, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        filepath = os.path.join(self.images_dir, filename)
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        return os.path.join("images", filename)
    
    def _save_feature_importance_chart(self, feature_cols: List[str], filename: str) -> str:
        """
        创建特征列表可视化（简单表格形式）
        
        参数:
            feature_cols: 特征列表
            filename: 文件名
        
        返回:
            图片相对路径
        """
        fig, ax = plt.subplots(figsize=(10, max(6, len(feature_cols) * 0.3)))
        ax.axis('tight')
        ax.axis('off')
        
        # 创建表格数据
        table_data = [[i+1, col] for i, col in enumerate(feature_cols)]
        
        table = ax.table(
            cellText=table_data,
            colLabels=['序号', '特征名称'],
            cellLoc='left',
            loc='center',
            colWidths=[0.2, 0.8]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表头样式
        for i in range(2):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 设置交替行颜色
        for i in range(1, len(feature_cols) + 1):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E7E6E6')
        
        plt.title('使用的特征列表', fontsize=14, fontweight='bold', pad=20)
        
        filepath = os.path.join(self.images_dir, filename)
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        return os.path.join("images", filename)
    
    def generate_training_report(
        self,
        model_type: str,
        dataset_info: Dict[str, Any],
        training_params: Dict[str, Any],
        results: Dict[str, Any],
        split_method: str = "简单切分"
    ) -> str:
        """
        生成训练报告（Markdown格式）
        
        参数:
            model_type: 模型类型
            dataset_info: 数据集信息
            training_params: 训练参数
            results: 训练结果
            split_method: 数据切分方法
        
        返回:
            报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_report_{timestamp}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        # 生成类别分布图
        class_dist_img = self._save_class_distribution_chart(
            dataset_info.get('class_distribution', {}),
            f"class_dist_train_{timestamp}.png"
        )
        
        # 生成特征列表图
        feature_img = self._save_feature_importance_chart(
            dataset_info.get('feature_cols', []),
            f"features_train_{timestamp}.png"
        )
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # 标题
            f.write("# 智能医疗系统 - 模型训练报告\n\n")
            f.write("---\n\n")
            
            # 基本信息
            f.write("## 📋 报告信息\n\n")
            f.write(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **报告类型**: 模型训练\n")
            f.write(f"- **模型类型**: {model_type}\n")
            f.write(f"- **数据切分方法**: {split_method}\n\n")
            
            f.write("---\n\n")
            
            # 数据集信息
            f.write("## 📊 数据集信息\n\n")
            f.write(f"### 基本统计\n\n")
            f.write(f"- **总样本数**: {dataset_info.get('total_samples', 'N/A')}\n")
            f.write(f"- **特征数量**: {dataset_info.get('n_features', 'N/A')}\n")
            f.write(f"- **标签列**: `{dataset_info.get('label_col', 'N/A')}`\n\n")
            
            # 特征列表
            f.write(f"### 使用的特征\n\n")
            f.write(f"共 **{len(dataset_info.get('feature_cols', []))}** 个特征：\n\n")
            
            feature_cols = dataset_info.get('feature_cols', [])
            # 分列显示特征
            cols_per_row = 3
            for i in range(0, len(feature_cols), cols_per_row):
                row_features = feature_cols[i:i+cols_per_row]
                f.write("| " + " | ".join(f"`{f}`" for f in row_features) + " |\n")
                if i == 0:
                    f.write("| " + " | ".join(["---"] * len(row_features)) + " |\n")
            
            f.write(f"\n![特征列表]({feature_img})\n\n")
            
            # 类别分布
            if 'class_distribution' in dataset_info:
                f.write(f"### 类别分布\n\n")
                f.write("| 类别 | 样本数 | 占比 |\n")
                f.write("|------|--------|------|\n")
                total = dataset_info['total_samples']
                for cls, count in dataset_info['class_distribution'].items():
                    percentage = count / total * 100
                    f.write(f"| {cls} | {count} | {percentage:.2f}% |\n")
                f.write(f"\n![类别分布]({class_dist_img})\n\n")
            
            f.write("---\n\n")
            
            # 训练参数
            f.write("## ⚙️ 训练配置\n\n")
            f.write(f"### 数据切分参数\n\n")
            f.write(f"- **随机种子**: {training_params.get('random_seed', 'N/A')}\n")
            
            if split_method == "简单切分":
                test_size = training_params.get('test_size', 'N/A')
                train_size = 1 - test_size if isinstance(test_size, float) else 'N/A'
                f.write(f"- **训练集比例**: {train_size:.0%} ({results.get('train_samples', 'N/A')} 样本)\n")
                f.write(f"- **测试集比例**: {test_size:.0%} ({results.get('test_samples', 'N/A')} 样本)\n\n")
            else:
                f.write(f"- **K折数**: {training_params.get('k_folds', 'N/A')}\n")
                f.write(f"- **交叉验证类型**: {results.get('cv_type', 'N/A')}K折\n")
                f.write(f"- **最终训练集**: {results.get('train_samples', 'N/A')} 样本\n")
                f.write(f"- **最终测试集**: {results.get('test_samples', 'N/A')} 样本\n\n")
            
            # 模型参数
            f.write(f"### {model_type} 模型参数\n\n")
            if model_type == "Random Forest":
                f.write(f"| 参数 | 值 | 说明 |\n")
                f.write(f"|------|-----|------|\n")
                f.write(f"| `n_estimators` | {training_params.get('rf_n_estimators', 'N/A')} | 树的数量 |\n")
                f.write(f"| `max_depth` | {training_params.get('rf_max_depth', 'None')} | 树的最大深度 |\n")
                f.write(f"| `max_features` | {training_params.get('rf_max_features', 'N/A')} | 每次分裂考虑的最大特征数 |\n")
            elif model_type == "SVM":
                f.write(f"| 参数 | 值 | 说明 |\n")
                f.write(f"|------|-----|------|\n")
                f.write(f"| `kernel` | {training_params.get('svm_kernel', 'N/A')} | 核函数类型 |\n")
                f.write(f"| `C` | {training_params.get('svm_C', 'N/A')} | 正则化参数 |\n")
                f.write(f"| `gamma` | {training_params.get('svm_gamma', 'N/A')} | 核系数 |\n")
            elif model_type == "Logistic Regression":
                f.write(f"| 参数 | 值 | 说明 |\n")
                f.write(f"|------|-----|------|\n")
                f.write(f"| `penalty` | {training_params.get('lr_penalty', 'N/A')} | 正则化类型 |\n")
                f.write(f"| `C` | {training_params.get('lr_C', 'N/A')} | 正则化强度的倒数 |\n")
                f.write(f"| `solver` | {training_params.get('lr_solver', 'N/A')} | 优化算法 |\n")
            f.write("\n")
            
            f.write("---\n\n")
            
            # 训练结果
            f.write("## 📈 训练结果\n\n")
            if split_method == "K折交叉验证":
                f.write("### 交叉验证评分\n\n")
                f.write("> 💡 交叉验证通过多次切分数据集来评估模型性能，结果更加稳定可靠。\n\n")
                
                # 准确率
                f.write(f"**准确率 (Accuracy)**\n")
                f.write(f"- 平均值: {results.get('cv_accuracy_mean', 0):.4f} ± {results.get('cv_accuracy_std', 0):.4f}\n")
                acc_scores = results.get('cv_accuracy_scores', [])
                if acc_scores:
                    f.write(f"- 各折: {', '.join([f'{s:.4f}' for s in acc_scores])}\n\n")
                
                # 召回率
                f.write(f"**召回率 (Recall)**\n")
                f.write(f"- 平均值: {results.get('cv_recall_mean', 0):.4f} ± {results.get('cv_recall_std', 0):.4f}\n")
                rec_scores = results.get('cv_recall_scores', [])
                if rec_scores:
                    f.write(f"- 各折: {', '.join([f'{s:.4f}' for s in rec_scores])}\n\n")
                
                # F1分数
                f.write(f"**F1分数 (F1-Score)**\n")
                f.write(f"- 平均值: {results.get('cv_f1_mean', 0):.4f} ± {results.get('cv_f1_std', 0):.4f}\n")
                f1_scores = results.get('cv_f1_scores', [])
                if f1_scores:
                    f.write(f"- 各折: {', '.join([f'{s:.4f}' for s in f1_scores])}\n\n")
                
                f.write(f"### 最终模型\n\n")
                f.write(f"使用 **{results.get('train_ratio', 80)}%** 的数据训练最终模型，保留 **{100-results.get('train_ratio', 80)}%** 用于测试评估。\n\n")
            else:
                f.write("### 模型训练完成\n\n")
                f.write(f"✅ 模型已使用训练集（{results.get('train_samples', 'N/A')} 样本）训练完成\n\n")
                f.write(f"📊 测试集（{results.get('test_samples', 'N/A')} 样本）已保留用于后续评估\n\n")
            
            f.write("---\n\n")
            
            # 后续步骤
            f.write("## 📝 后续步骤\n\n")
            f.write("1. ✅ 模型训练已完成\n")
            f.write("2. 🔜 请查看**评估报告**了解模型在测试集上的详细性能\n")
            f.write("3. 🔜 如需优化，可以调整模型参数后重新训练\n")
            f.write("4. 🔜 性能满意后，可以使用模型进行预测\n\n")
            
            f.write("---\n\n")
            f.write(f"*报告生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        # 收集所有图片文件
        image_files = [
            os.path.join(self.images_dir, os.path.basename(class_dist_img)),
            os.path.join(self.images_dir, os.path.basename(feature_img))
        ]
        
        # 读取并保存报告内容用于预览
        with open(filepath, 'r', encoding='utf-8') as f:
            self.last_training_report = f.read()
        
        # 创建zip包
        zip_path = self._create_zip_package(filepath, image_files, "training_report")
        
        return zip_path
    
    def generate_evaluation_report(
        self,
        model_type: str,
        dataset_info: Dict[str, Any],
        metrics: Dict[str, Any],
        data_source: str = "测试集",
        confusion_matrix_fig=None,
        roc_fig=None,
        pr_fig=None
    ) -> str:
        """
        生成评估报告（Markdown格式）
        
        参数:
            model_type: 模型类型
            dataset_info: 数据集信息
            metrics: 评估指标
            data_source: 数据来源
            confusion_matrix_fig: 混淆矩阵图表对象
            roc_fig: ROC曲线图表对象
            pr_fig: PR曲线图表对象
        
        返回:
            报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_report_{timestamp}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        # 保存图表
        cm_img = None
        roc_img = None
        pr_img = None
        
        if confusion_matrix_fig is not None:
            cm_filename = f"confusion_matrix_{timestamp}.png"
            cm_path = os.path.join(self.images_dir, cm_filename)
            confusion_matrix_fig.write_image(cm_path)
            cm_img = os.path.join("images", cm_filename)
        
        if roc_fig is not None:
            roc_filename = f"roc_curve_{timestamp}.png"
            roc_path = os.path.join(self.images_dir, roc_filename)
            roc_fig.write_image(roc_path)
            roc_img = os.path.join("images", roc_filename)
        
        if pr_fig is not None:
            pr_filename = f"pr_curve_{timestamp}.png"
            pr_path = os.path.join(self.images_dir, pr_filename)
            pr_fig.write_image(pr_path)
            pr_img = os.path.join("images", pr_filename)
        
        # 生成类别分布图
        class_dist_img = self._save_class_distribution_chart(
            dataset_info.get('class_distribution', {}),
            f"class_dist_eval_{timestamp}.png"
        )
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # 标题
            f.write("# 智能医疗系统 - 模型评估报告\n\n")
            f.write("---\n\n")
            
            # 基本信息
            f.write("## 📋 报告信息\n\n")
            f.write(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **报告类型**: 模型评估\n")
            f.write(f"- **模型类型**: {model_type}\n")
            f.write(f"- **数据来源**: {data_source}\n\n")
            
            f.write("---\n\n")
            
            # 数据集信息
            f.write("## 📊 评估数据集\n\n")
            f.write(f"- **样本数量**: {dataset_info.get('n_samples', 'N/A')}\n")
            f.write(f"- **特征数量**: {dataset_info.get('n_features', 'N/A')}\n\n")
            
            if 'class_distribution' in dataset_info:
                f.write(f"### 类别分布\n\n")
                f.write("| 类别 | 样本数 | 占比 |\n")
                f.write("|------|--------|------|\n")
                total = dataset_info['n_samples']
                for cls, count in dataset_info['class_distribution'].items():
                    percentage = count / total * 100
                    f.write(f"| {cls} | {count} | {percentage:.2f}% |\n")
                f.write(f"\n![类别分布]({class_dist_img})\n\n")
            
            f.write("---\n\n")
            
            # 整体性能指标
            f.write("## 📈 整体性能指标\n\n")
            f.write("| 指标 | 值 | 说明 |\n")
            f.write("|------|-----|------|\n")
            f.write(f"| **准确率 (Accuracy)** | {metrics.get('accuracy', 'N/A'):.4f} | 正确预测的样本比例 |\n")
            f.write(f"| **精确率 (Precision)** | {metrics.get('precision_macro', 'N/A'):.4f} | 预测为正的样本中真正为正的比例 |\n")
            f.write(f"| **召回率 (Recall)** | {metrics.get('recall_macro', 'N/A'):.4f} | 真实为正的样本中被正确预测的比例 |\n")
            f.write(f"| **F1分数 (F1-Score)** | {metrics.get('f1_macro', 'N/A'):.4f} | 精确率和召回率的调和平均 |\n")
            
            if 'auc' in metrics and metrics['auc'] is not None:
                f.write(f"| **AUC** | {metrics['auc']:.4f} | ROC曲线下的面积 |\n")
            elif 'auc_ovr' in metrics:
                f.write(f"| **AUC (OVR)** | {metrics['auc_ovr']:.4f} | One-vs-Rest AUC |\n")
                f.write(f"| **AUC (OVO)** | {metrics['auc_ovo']:.4f} | One-vs-One AUC |\n")
            f.write("\n")
            
            f.write("---\n\n")
            
            # 每类别详细指标
            if 'classification_report' in metrics:
                f.write("## 📊 各类别详细指标\n\n")
                report = metrics['classification_report']
                
                # 表头
                f.write("| 类别 | 精确率 | 召回率 | F1分数 | 样本数 |\n")
                f.write("|------|--------|--------|--------|--------|\n")
                
                # 各类别数据
                for label, values in report.items():
                    if label not in ['accuracy', 'macro avg', 'weighted avg']:
                        precision = values.get('precision', 0)
                        recall = values.get('recall', 0)
                        f1 = values.get('f1-score', 0)
                        support = values.get('support', 0)
                        f.write(f"| {label} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {int(support)} |\n")
                
                f.write("|------|--------|--------|--------|--------|\n")
                
                # 平均值
                if 'macro avg' in report:
                    macro = report['macro avg']
                    f.write(f"| **宏平均** | {macro.get('precision', 0):.4f} | {macro.get('recall', 0):.4f} | {macro.get('f1-score', 0):.4f} | - |\n")
                
                if 'weighted avg' in report:
                    weighted = report['weighted avg']
                    f.write(f"| **加权平均** | {weighted.get('precision', 0):.4f} | {weighted.get('recall', 0):.4f} | {weighted.get('f1-score', 0):.4f} | - |\n")
                
                f.write("\n")
            
            f.write("---\n\n")
            
            # 可视化结果
            f.write("## 📉 可视化分析\n\n")
            
            if cm_img:
                f.write("### 混淆矩阵\n\n")
                f.write("混淆矩阵展示了模型预测结果与真实标签的对比情况：\n\n")
                f.write(f"![混淆矩阵]({cm_img})\n\n")
            
            if roc_img:
                f.write("### ROC曲线\n\n")
                f.write("ROC曲线展示了不同阈值下真正例率与假正例率的关系：\n\n")
                f.write(f"![ROC曲线]({roc_img})\n\n")
            
            if pr_img:
                f.write("### PR曲线\n\n")
                f.write("PR曲线展示了不同阈值下精确率与召回率的关系：\n\n")
                f.write(f"![PR曲线]({pr_img})\n\n")
            
            f.write("---\n\n")
            
            # 性能解读
            f.write("## 💡 性能解读\n\n")
            acc = metrics.get('accuracy', 0)
            f1 = metrics.get('f1_macro', 0)
            
            f.write("### 准确率评价\n\n")
            if acc >= 0.9:
                f.write("✅ **优秀** (≥90%) - 模型准确率表现优异\n\n")
            elif acc >= 0.8:
                f.write("✅ **良好** (≥80%) - 模型准确率表现良好\n\n")
            elif acc >= 0.7:
                f.write("⚠️ **中等** (≥70%) - 模型准确率尚可，建议优化\n\n")
            else:
                f.write("❌ **较低** (<70%) - 模型准确率较低，建议调整模型参数或增加训练数据\n\n")
            
            f.write("### F1分数评价\n\n")
            if f1 >= 0.85:
                f.write("✅ **优秀** (≥85%) - 模型综合性能优异\n\n")
            elif f1 >= 0.75:
                f.write("✅ **良好** (≥75%) - 模型综合性能良好\n\n")
            elif f1 >= 0.65:
                f.write("⚠️ **中等** (≥65%) - 模型综合性能尚可\n\n")
            else:
                f.write("❌ **较低** (<65%) - 模型可能存在过拟合或欠拟合问题\n\n")
            
            f.write("---\n\n")
            
            # 优化建议
            f.write("## 🎯 优化建议\n\n")
            if acc < 0.8 or f1 < 0.75:
                f.write("### 建议改进方向\n\n")
                f.write("1. **参数调优**: 尝试调整模型超参数（学习率、正则化系数等）\n")
                f.write("2. **特征工程**: 考虑特征选择、特征变换或特征组合\n")
                f.write("3. **数据增强**: 增加训练样本数量，特别是少数类样本\n")
                f.write("4. **模型选择**: 尝试其他机器学习算法进行对比\n")
                f.write("5. **集成学习**: 考虑使用模型集成方法提升性能\n\n")
            else:
                f.write("### 后续工作建议\n\n")
                f.write("1. **泛化验证**: 在更多真实数据上验证模型泛化能力\n")
                f.write("2. **错误分析**: 分析预测错误的样本，找出改进方向\n")
                f.write("3. **模型部署**: 性能满意后可以部署到生产环境\n")
                f.write("4. **持续监控**: 定期评估模型性能，及时更新维护\n\n")
            
            f.write("---\n\n")
            f.write(f"*报告生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        # 收集所有图片文件
        image_files = [os.path.join(self.images_dir, os.path.basename(class_dist_img))]
        
        if cm_img:
            image_files.append(os.path.join(self.images_dir, os.path.basename(cm_img)))
        if roc_img:
            image_files.append(os.path.join(self.images_dir, os.path.basename(roc_img)))
        if pr_img:
            image_files.append(os.path.join(self.images_dir, os.path.basename(pr_img)))
        
        # 读取并保存报告内容用于预览
        with open(filepath, 'r', encoding='utf-8') as f:
            self.last_evaluation_report = f.read()
        
        # 创建zip包
        zip_path = self._create_zip_package(filepath, image_files, "evaluation_report")
        
        return zip_path
