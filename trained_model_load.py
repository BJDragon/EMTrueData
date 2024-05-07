import sys

from torchsummary import summary
from train_util import *
import torch
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import torch
import pandas as pd


def save_predictions_to_csv(model, dataloader, file_path):
    # 确保模型在评估模式
    model.eval()

    # 存储结果的列表
    actuals = []
    predictions = []

    # 不更新梯度
    with torch.no_grad():
        for data, target in dataloader:
            # 假设data和target都已经在正确的设备上（例如GPU或CPU）
            output = model(data)
            # 将输出转换为一维数组
            output = output.view(-1).tolist()
            target = target.view(-1).tolist()

            # 收集实际值和预测值
            actuals.extend(target)
            predictions.extend(output)

    # 计算绝对误差和相对误差
    actuals = torch.tensor(actuals)
    predictions = torch.tensor(predictions)
    abs_errors = torch.abs(actuals - predictions)
    rel_errors = torch.abs((actuals - predictions) / actuals) * 100  # 百分比形式

    # 将tensor转换为list，准备写入DataFrame
    abs_errors = abs_errors.tolist()
    rel_errors = rel_errors.tolist()

    # 创建DataFrame
    df = pd.DataFrame({
        'Num': range(1, len(abs_errors) + 1),
        'Actual': actuals.tolist(),
        'Prediction': predictions.tolist(),
        'Absolute Error': abs_errors,
        'Relative Error (%)': rel_errors
    })

    # 舍入数值到两位小数
    df = df.round(2)

    # 保存到CSV
    df.to_csv(file_path, index=False)
    print(f'Data saved to {file_path}')


def plot_model_predictions(model, train_loader, val_loader, title='Model Predictions'):
    # 初始化数据存储
    train_true, train_preds = [], []
    val_true, val_preds = [], []

    # 不计算梯度，进行预测
    model.to('cpu').eval()  # 设置为评估模式
    with torch.no_grad():
        for inputs, labels in train_loader:
            outputs = model(inputs)
            train_preds.extend(outputs.flatten().tolist())  # 假设输出是单个值
            train_true.extend(labels.tolist())

        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_preds.extend(outputs.flatten().tolist())
            val_true.extend(labels.tolist())

    # 绘图
    plt.figure(figsize=(14, 6))
    plt.suptitle(title)  # 添加整体标题

    plt.subplot(1, 2, 1)
    plt.scatter(range(len(train_true)), train_true, color='blue', label='Train True', alpha=0.6, marker='o')
    plt.scatter(range(len(train_preds)), train_preds, color='red', alpha=0.6, label='Train Pred', marker='x')
    plt.title('Training Data')
    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(range(len(val_true)), val_true, color='green', label='Validation True', alpha=0.6, marker='o')
    plt.scatter(range(len(val_preds)), val_preds, color='orange', alpha=0.6, label='Validation Pred', marker='x')
    plt.title('Validation Data')
    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)

    plt.show()


def transform(x, y):
    x = x.reshape(-1, 1, 25) - 0.1
    return x, y


def main(args: argparse.Namespace):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('---------Train on: ' + device + '----------')

    if args.seed is not None:
        set_seed(args.seed)

    train_loader, val_loader = get_loader(args)

    input_size = tuple(train_loader.dataset.x.shape[1:])

    # ECA_ResNet18 ResNet18
    n = int(input())
    model_path = f'trained_model/optuna_model_2024-04-29_{n}_best.pt'
    # model_path = f'trained_model/model_2024-04-29_trailNone.pt'
    model = torch.load(model_path).to(device)
    summary(model, input_size=input_size)
    criterion = nn.MSELoss(reduction='sum')

    #
    model.to(device)
    train_type = 'regression'
    epoch = 0
    train_batch_loss = val(epoch, train_loader, model, criterion, args, train_type)
    val_batch_acc = val(epoch, val_loader, model, criterion, args, train_type)
    print(train_batch_loss + val_batch_acc)

    plot_model_predictions(model, train_loader, val_loader)

    save_predictions_to_csv(model, train_loader, 'temp_file/train_loader.csv')
    save_predictions_to_csv(model, val_loader, 'temp_file/val_loader.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sd', '--seed', default=42, type=int)
    parser.add_argument('-tp', '--x_path', default='data/reproducted/coding_data.npy')
    parser.add_argument('-vp', '--y_path', default='data/reproducted/shift_value.npy')
    parser.add_argument('-bs', '--batch_size', type=int, default=40)
    parser.add_argument('-ep', '--epochs', type=int, default=1)
    parser.add_argument('-ts', '--test_size', type=float, default=0.1)

    args = parser.parse_args()
    main(args)
