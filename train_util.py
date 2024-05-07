import json
import os
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch
from matplotlib import pyplot as plt
from train import transform
from torch import nn


def set_seed(seed: int):
    # 随机种子设定
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    pass


def init_weights(m: nn.Module):
    # 初始化网络参数层权重
    class_name = m.__class__.__name__
    if class_name.find('Conv2d') != -1 or class_name.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif class_name.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class CustomImageDataset(Dataset):
    # 只是将输入的x与y输出称为dataset类型
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        xi = self.x[idx]
        yi = self.y[idx]
        return xi, yi


def path_exists(path):
    if os.path.exists(path):
        print("路径已存在:", path)
    else:
        print("路径不存在:", path)
        os.makedirs(path)
        print("路径已创建:", path)


def get_loader(args: argparse.Namespace):
    # 数据加载并填装进入loader
    x = np.load(args.x_path)
    y = np.load(args.y_path)

    x, y = transform(x, y)

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=args.test_size, random_state=42)

    train_dataset = CustomImageDataset(x_train, y_train)
    val_dataset = CustomImageDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader


def predictions_errors(actuals, predictions):
    actuals = actuals.view(-1)
    predictions = predictions.view(-1)
    abs_errors = torch.abs(actuals - predictions)
    rel_errors = torch.abs((actuals - predictions) / actuals) * 100  # 百分比形式

    # 将tensor转换为list，准备写入DataFrame
    abs_errors = abs_errors.tolist()
    rel_errors = rel_errors.tolist()

    mean_rel_error = np.mean(rel_errors)
    return abs_errors, rel_errors, mean_rel_error


def train(epoch: int, train_loader: DataLoader, model, optimizer, criterion,
          args: argparse.Namespace, train_type, device='cuda'):
    if train_type == 'regression':
        # 训练过程
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            # 梯度清除
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 训练集在模型上的评估
        model.eval()
        train_loss_list = []
        rel_error_list = []
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                train_loss_list = np.append(train_loss_list, loss.item())
                _, batch_rel_error, _ = predictions_errors(y, out)
                rel_error_list.extend(batch_rel_error)
            train_loss = sum(train_loss_list) / train_loader.dataset.x.shape[0]
            rel_error = np.mean(rel_error_list)
        print(f"[ Train | {epoch + 1:03d}/{args.epochs:03d} ] loss = {train_loss:.5f} \n"
              f"                    train_rel_error = {rel_error:.2f} %")
        return train_loss


def val(epoch: int, val_loader: DataLoader, model, criterion,
        args: argparse.Namespace, train_type, device='cuda'):
    if train_type == 'regression':
        model.eval()
        val_loss_lis = np.array([])
        rel_error_list = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                _, batch_rel_error, _ = predictions_errors(y, out)
                val_loss_lis = np.append(val_loss_lis, loss.item())
                rel_error_list.extend(batch_rel_error)

            val_loss = sum(val_loss_lis) / val_loader.dataset.x.shape[0]
            rel_error = np.mean(rel_error_list)
        print(f"[ Validation | {epoch + 1:03d}/{args.epochs:03d} ]  loss = {val_loss:.5f} \n"
              f"                    val_rel_error = {rel_error:.2f} %")
        return val_loss


def draw_regression_loss(train_losses, val_losses, save_path, title=None, show=True):
    train_losses = train_losses.reshape(-1)
    val_losses = val_losses.reshape(-1)

    epochs = range(1, len(train_losses) + 1)  # 7个epoch，从1开始
    fig, ax1 = plt.subplots()

    # 绘制训练损失
    color = 'dodgerblue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color=color)
    ax1.plot(epochs, train_losses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.yscale('log')
    plt.grid(True)

    # 创建一个共享x轴的第二个轴对象
    ax2 = ax1.twinx()
    color = 'orange'
    ax2.set_ylabel('Validation Loss', color=color)  # 设置第二个y轴的标签
    ax2.plot(epochs, val_losses, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.yscale('log')

    if title != None:
        plt.title(title)
    else:
        title = 'Training and Validation Losses(log y)'
        plt.title(title)
    # path_exists('trials_loss_png')
    plt.savefig(save_path)
    print(f'Saved {title} to: {save_path}')
    if show:
        plt.show()


def save_losses(train_losses, val_losses, file_path):
    # 创建DataFrame
    df = pd.DataFrame({
        'Num': range(1, len(train_losses) + 1),
        'TrainLosses': train_losses.tolist(),
        'ValLosses': val_losses.tolist()
    })

    # 保存到CSV
    df.to_csv(file_path, index=False)
    print(f'Data saved to: {file_path}')


def modelError(model, dataloader, device='cuda', save_data=False, save_file_name=None):
    """
    计算并输出模型的平均绝对误差和平均相对误差，并可选择保存误差数据。

    参数:
    - model: 训练好的模型，用于进行预测。
    - dataloader: 包含真实标签的数据加载器，用于模型的评估。
    - device: 指定运行设备，默认为'cuda'，即使用GPU加速。
    - save_data: 布尔值，选择是否保存误差数据到CSV文件，默认为False。
    - save_path: 字符串，指定保存CSV文件的路径，默认为False，即使用默认路径。

    返回值:
    - 无返回值，但会打印平均绝对误差和平均相对误差，并可选择保存误差数据。
    """
    model.to(device)
    model.eval()
    pred_y = []
    true_y = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            true_y.extend(y)
            out = model(x)
            pred_y.extend(out)

    pred_y = torch.tensor(pred_y)
    true_y = torch.tensor(true_y)
    # 计算绝对误差
    abs_error = torch.abs(pred_y - true_y)
    # 计算相对误差
    rel_error = torch.abs((pred_y - true_y) / true_y) * 100  # 百分制

    mean_abs_error = torch.mean(abs_error)
    mean_rel_error = torch.mean(rel_error)

    # 选择是否保存误差数据
    if save_data:
        # 创建DataFrame
        df = pd.DataFrame({
            'Num': range(1, len(true_y) + 1),
            'Actual': true_y.tolist(),
            'Prediction': pred_y.tolist(),
            'Absolute Error': abs_error.tolist(),
            'Relative Error (%)': rel_error.tolist(),
            'Mean Absolute Error': mean_abs_error.tolist(),
            'Mean Relative Error': mean_rel_error.tolist()
        })

        # 舍入数值到两位小数
        df = df.round(2)

        # 根据是否指定了保存路径来保存文件
        if save_file_name is None:
            # 保存到CSV
            df.to_csv('temp_file/modelError.csv', index=False)
            print('modelError Data saved to [ temp_file/modelError.csv ]')
        else:
            # 保存到CSV
            df.to_csv(f'{save_file_name}.csv', index=False)
            print(f'modelError Data saved to [ {save_file_name}.csv ]')

    # 打印平均绝对误差和平均相对误差
    print(f'                    Mean absolute Error: {mean_abs_error: .5f}')
    print(f'                    Mean relative Error: {mean_rel_error: .3f} %')
    return mean_rel_error
