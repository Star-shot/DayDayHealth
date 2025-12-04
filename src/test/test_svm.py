from ..models import svm
from ..utils import load_data
from sklearn.model_selection import train_test_split
import argparse
import os

parser = argparse.ArgumentParser(description='数据处理脚本')
# 输入文件参数
parser.add_argument('--input', 
                    type=str, 
                    default='data/data_cleaned.xlsx',
                    help='输入文件路径 (默认: %(default)s)')

# 输出目录参数
parser.add_argument('--output',
                    type=str, 
                    default='out_dir',
                    help='输出目录路径 (默认: %(default)s)')

args = parser.parse_args()

# 路径处理
input_path = os.path.abspath(args.input)
out_dir = os.path.abspath(args.output)

# 验证输入文件存在
if not os.path.exists(input_path):
    raise FileNotFoundError(f"输入文件不存在: {input_path}")

filepath = '.\data\data_cleaned.xlsx'
X, y = load_data(filepath)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练模型
svm_model = svm.SVM(kernel='rbf', C=1.0, gamma='scale', out_dir=out_dir)
svm_model.train(X_train, y_train)

# 评估模型
metrics = svm_model.evaluate(X_test, y_test)
print("评估指标已保存至output目录")