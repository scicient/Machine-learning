# -*- coding: utf-8 -*-
"""
================================================================================
《数据挖掘与机器学习》课程设计
基于ORL人脸数据集的图像多分类任务
技术方案：残差网络18层(ResNet18)深度学习 + 主成分分析(PCA)+支持向量机(SVM)传统对比
================================================================================
"""

import os
import sys
import io

# ===== Windows系统UTF-8编码适配 =====
# 强制标准输出使用UTF-8编码，解决Windows下中文显示为乱码的问题
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，Windows下避免弹窗
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ==================== 0. 全局配置 ====================
# 数据集根目录（Windows本地固定路径）
DATA_ROOT = r"E:\桌面\work\The ORL Database of Faces"
# 实验结果输出目录
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "实验结果")
# 设备选择：优先使用GPU，否则使用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数配置
BATCH_SIZE = 16          # 批量大小
LEARNING_RATE = 0.001    # 学习率
NUM_EPOCHS = 80          # 最大训练轮数
EARLY_STOP_PATIENCE = 15 # 早停耐心值：连续N轮验证集准确率不提升则停止
IMG_SIZE = 112           # 统一缩放后的图片尺寸（像素）
NUM_CLASSES = 40         # 分类类别数（s1~s40共40个类别）

# 中文显示配置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

print("=" * 70)
print("《数据挖掘与机器学习》课程设计 — ORL人脸多分类任务")
print(f"运行设备：{DEVICE}")
print(f"PyTorch版本：{torch.__version__}")
print("=" * 70)


# ==================== 1. 数据集读取模块 ====================
def load_orl_dataset(data_root):
    """
    读取ORL人脸数据集，以文件夹为分类单元加载所有PGM灰度图片。

    参数:
        data_root (str): 数据集根目录路径

    返回:
        images (list[np.ndarray]): 所有图像数据列表，每张为二维灰度数组
        labels (list[int]): 每张图像对应的数字标签（内部0~39，显示时映射为1~40）
        folder_map (dict): 文件夹名→类别编号的映射关系
    """
    print("\n【数据集读取模块】")
    print(f"正在从路径读取数据集：{data_root}")

    images = []
    labels = []
    folder_map = {}

    # 获取所有以's'开头的文件夹并排序
    all_folders = sorted(
        [d for d in os.listdir(data_root)
         if os.path.isdir(os.path.join(data_root, d)) and d.startswith("s")],
        key=lambda x: int(x[1:])  # 按s后的数字排序，s1, s2, ..., s40
    )

    print(f"检测到文件夹数量：{len(all_folders)}")

    for folder in all_folders:
        # 解析文件夹名获取类别编号：s1→1, s2→2, ..., s40→40（内部存储0~39，显示时+1）
        class_idx = int(folder[1:]) - 1  # 内部标签0~39，适配PyTorch交叉熵损失
        folder_map[folder] = class_idx + 1  # 对外显示1~40

        folder_path = os.path.join(data_root, folder)
        # 读取文件夹内所有pgm图片
        pgm_files = sorted(
            [f for f in os.listdir(folder_path) if f.endswith(".pgm")],
            key=lambda x: int(x.split(".")[0])  # 按文件名数字排序：1.pgm, 2.pgm, ...
        )

        for pgm_file in pgm_files:
            img_path = os.path.join(folder_path, pgm_file)
            try:
                # 使用PIL读取pgm灰度图片
                img = Image.open(img_path)
                img_array = np.array(img, dtype=np.uint8)
                images.append(img_array)
                labels.append(class_idx)
            except Exception as e:
                print(f"  [警告] 读取文件失败：{img_path}，错误：{e}")

    print(f"成功读取图片总数：{len(images)} 张")
    print(f"分类类别总数：{len(set(labels))} 类（s1~s40 → 类别1~40）")
    print(f"文件夹→类别映射：{folder_map}")

    return images, labels, folder_map


# ==================== 2. 数据划分模块 ====================
def split_dataset_by_folder(images, labels, train_count=5):
    """
    按文件夹（类别）内部拆分训练集与测试集。
    每个类别的前train_count张图片作为训练集，剩余作为测试集。

    参数:
        images (list): 图像数据列表
        labels (list): 标签列表
        train_count (int): 每个类别用于训练的张数（5或7）

    返回:
        train_images, train_labels, test_images, test_labels
    """
    print(f"\n【数据划分模块 — 方案：每类前{train_count}张训练，剩余测试】")

    train_images, train_labels = [], []
    test_images, test_labels = [], []

    # 按类别分组
    from collections import defaultdict
    class_to_samples = defaultdict(list)
    for img, lbl in zip(images, labels):
        class_to_samples[lbl].append(img)

    for cls_idx in sorted(class_to_samples.keys()):
        samples = class_to_samples[cls_idx]
        # 前train_count张训练，剩余测试
        train_imgs = samples[:train_count]
        test_imgs = samples[train_count:]

        train_images.extend(train_imgs)
        train_labels.extend([cls_idx] * len(train_imgs))
        test_images.extend(test_imgs)
        test_labels.extend([cls_idx] * len(test_imgs))

    print(f"训练集样本数：{len(train_images)}（{40}类 × {train_count}张 = {40 * train_count}）")
    print(f"测试集样本数：{len(test_images)}（{40}类 × {10 - train_count}张 = {40 * (10 - train_count)}）")

    return train_images, train_labels, test_images, test_labels


# ==================== 3. 自定义PyTorch数据集类 ====================
class ORLFaceDataset(Dataset):
    """
    自定义ORL人脸数据集类，继承torch.utils.data.Dataset。
    负责将numpy灰度图像转换为PyTorch张量，并应用数据增强和归一化等预处理。
    """

    def __init__(self, images, labels, transform=None, is_train=True):
        """
        参数:
            images (list[np.ndarray]): 灰度图像列表
            labels (list[int]): 图像标签列表
            transform (callable): 图像预处理变换
            is_train (bool): 是否为训练模式（训练集启用数据增强）
        """
        self.images = images
        self.labels = labels
        self.is_train = is_train

        # 定义预处理流程
        if transform is None:
            if is_train:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),                        # numpy→PIL图像
                    transforms.Resize((IMG_SIZE, IMG_SIZE)),       # 统一缩放
                    transforms.RandomHorizontalFlip(p=0.5),        # 随机水平翻转（数据增强）
                    transforms.RandomRotation(degrees=10),         # 随机旋转±10度（数据增强）
                    transforms.ToTensor(),                          # PIL→张量 [0,1]
                    transforms.Normalize(mean=[0.5], std=[0.5]),   # 归一化到[-1, 1]
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((IMG_SIZE, IMG_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ])
        else:
            self.transform = transform

    def __len__(self):
        """返回数据集样本总数"""
        return len(self.images)

    def __getitem__(self, idx):
        """
        获取单个样本。

        返回:
            image (Tensor): 形状[1, H, W]的单通道图像张量
            label (int): 类别标签
        """
        img = self.images[idx]
        label = self.labels[idx]
        img = self.transform(img)
        return img, label


# ==================== 4. 残差网络18层(ResNet18)模型模块 ====================
class ResidualBlock(nn.Module):
    """
    残差块：残差网络的基本构建单元。
    包含两个3×3卷积层，通过跳跃连接将输入直接加到输出上，
    有效缓解深层网络的梯度消失问题。
    """

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        参数:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            stride (int): 卷积步长
            downsample (nn.Module): 下采样层（当输入输出尺寸不匹配时使用）
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        """前向传播：卷积→批归一化→ReLU→卷积→批归一化→残差加和→ReLU"""
        identity = x  # 保存输入作为残差连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 若存在下采样层，调整输入尺寸以匹配输出
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 残差连接（跳跃连接）
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    """
    改进版残差网络18层(ResNet18)，专门适配单通道灰度图像。
    网络结构：
        初始卷积层 → 4个残差层(每层含2个残差块) → 全局平均池化 → 全连接分类层
    总计：1个卷积层 + 8个残差块(16层卷积) + 1个全连接层 = 18层
    """

    def __init__(self, num_classes=NUM_CLASSES, in_channels=1):
        """
        参数:
            num_classes (int): 分类类别数（ORL数据集为40）
            in_channels (int): 输入通道数（灰度图为1）
        """
        super(ResNet18, self).__init__()

        # 初始卷积与池化层
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差层1：输入64通道→输出64通道（不改变尺寸和通道数）
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)

        # 残差层2：输入64通道→输出128通道（尺寸减半）
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)

        # 残差层3：输入128通道→输出256通道（尺寸减半）
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)

        # 残差层4：输入256通道→输出512通道（尺寸减半）
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)

        # 全局平均池化层，将特征图压缩为向量
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接分类层
        self.fc = nn.Linear(512, num_classes)

        # 权重初始化
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """
        构建一个残差层，包含num_blocks个残差块。

        参数:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            num_blocks (int): 残差块数量
            stride (int): 第一个残差块的步长

        返回:
            nn.Sequential: 残差层
        """
        downsample = None
        # 当输入输出通道数不同或步长不为1时，需要下采样层匹配维度
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        # 第一个残差块可能需要下采样
        layers.append(
            ResidualBlock(in_channels, out_channels, stride, downsample)
        )
        # 后续残差块不需要改变尺寸
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Kaiming初始化：为卷积层和全连接层赋予合理的初始权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """前向传播"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)   # 展平为[batch, 512]向量
        x = self.fc(x)            # 全连接分类

        return x


# ==================== 5. 训练循环模块 ====================
class EarlyStopping:
    """
    早停机制：在验证集性能连续多轮不提升时自动停止训练，
    避免过拟合，节省训练时间。
    """

    def __init__(self, patience=EARLY_STOP_PATIENCE, verbose=True, mode="max"):
        """
        参数:
            patience (int): 容忍轮数
            verbose (bool): 是否打印提示信息
            mode (str): "max"表示指标越大越好，"min"表示越小越好
        """
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        self.early_stop = False

        if mode == "max":
            self.monitor_op = lambda x, y: x > y
        else:
            self.monitor_op = lambda x, y: x < y

    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.verbose:
                print(f"  [首轮] 初始参考值：{self.best_score:.4f}")
        elif not self.monitor_op(score, self.best_score):
            self.counter += 1
            if self.verbose:
                print(f"  [未提升] 本次{score:.4f} ≤ 最优{self.best_score:.4f}，累计{self.counter}/{self.patience}轮")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  [早停触发] 在第{epoch+1}轮停止训练，最优轮数为第{self.best_epoch+1}轮")
        else:
            old_best = self.best_score
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"  [性能提升] 本次{score:.4f} > 旧纪录{old_best:.4f}，计数器归零 ✓")

        return self.early_stop


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    执行一个训练轮次。

    参数:
        model (nn.Module): 待训练的模型
        dataloader (DataLoader): 训练数据加载器
        criterion (loss): 损失函数
        optimizer (Optimizer): 优化器
        device (torch.device): 运行设备

    返回:
        epoch_loss (float): 本轮的训练损失均值
        epoch_acc (float): 本轮训练集准确率
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播与参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate_model(model, dataloader, criterion, device):
    """
    在给定数据集上评估模型性能。

    参数:
        model (nn.Module): 模型
        dataloader (DataLoader): 数据加载器
        criterion (loss): 损失函数
        device (torch.device): 运行设备

    返回:
        avg_loss (float): 平均损失
        accuracy (float): 准确率
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def train_model(model, train_loader, test_loader, device, save_name="最优模型_方案"):
    """
    完整训练流程，包含早停机制和最优模型保存。

    参数:
        model (nn.Module): 待训练模型
        train_loader (DataLoader): 训练集数据加载器
        test_loader (DataLoader): 测试集数据加载器
        device (torch.device): 运行设备
        save_name (str): 模型保存文件名前缀

    返回:
        history (dict): 训练历史记录（损失、准确率曲线数据）
    """
    print(f"\n【训练循环模块 — {save_name}】")
    print(f"训练设备：{device}")
    print(f"批大小：{BATCH_SIZE}，学习率：{LEARNING_RATE}，最大轮数：{NUM_EPOCHS}")

    # 损失函数：交叉熵损失，用于多分类任务
    criterion = nn.CrossEntropyLoss()
    # 优化器：Adam优化器，自适应学习率，收敛速度快
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 早停机制
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, mode="max")

    # 训练历史记录
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    best_test_acc = 0.0
    best_epoch = 0
    best_model_path = os.path.join(RESULT_DIR, f"{save_name}.pth")

    for epoch in range(NUM_EPOCHS):
        # 训练一轮
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        # 测试集评估
        test_loss, test_acc = evaluate_model(
            model, test_loader, criterion, device
        )

        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        # 保存最优模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            # 保存完整模型对象（pickle序列化，供GUI与评估加载）
            torch.save(model.cpu(), best_model_path)
            model.to(device)  # 恢复到原设备继续训练

        # 每5轮或首尾轮打印信息
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  第{epoch+1:3d}轮 | 训练损失：{train_loss:.4f} | "
                  f"训练准确率：{train_acc:.4f} | 测试准确率：{test_acc:.4f} | "
                  f"最优测试准确率：{best_test_acc:.4f}")

        # 早停检查
        if early_stopping(test_acc, epoch):
            break

    print(f"\n训练完成！最优测试准确率：{best_test_acc:.4f}（第{best_epoch}轮）")
    print(f"最优模型已保存至：{best_model_path}")

    return history


# ==================== 6. 指标评估模块 ====================
def compute_all_metrics(model, dataloader, device, class_names):
    """
    计算并打印全部定量评价指标：总体准确率、各类精确率、召回率。

    参数:
        model (nn.Module): 训练好的模型
        dataloader (DataLoader): 测试集数据加载器
        device (torch.device): 运行设备
        class_names (list[str]): 类别名称列表

    返回:
        all_preds (np.ndarray): 所有预测标签
        all_labels (np.ndarray): 所有真实标签
    """
    print(f"\n【指标评估模块】")

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 总体准确率
    total_acc = accuracy_score(all_labels, all_preds)
    # 每一类精确率
    per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    # 每一类召回率
    per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)

    # 控制台打印完整指标
    print(f"\n{'='*60}")
    print(f">>> 总体准确率：{total_acc:.4f}（{total_acc*100:.2f}%）")
    print(f"{'='*60}")
    print(f"{'类别':<10}{'精确率':<12}{'召回率':<12}")
    print("-" * 34)
    for i in range(NUM_CLASSES):
        print(f"s{i+1:<8} {per_class_precision[i]:.4f}       {per_class_recall[i]:.4f}")
    print("-" * 34)
    print(f"{'宏平均':<10}{np.mean(per_class_precision):.4f}       {np.mean(per_class_recall):.4f}")
    print(f"{'='*60}")

    return all_preds, all_labels, total_acc, per_class_precision, per_class_recall


# ==================== 7. 可视化绘图模块 ====================
def plot_training_curves(history, split_name, save_dir):
    """
    绘制并保存训练损失变化曲线、训练集与测试集准确率对比曲线。

    参数:
        history (dict): 训练历史记录
        split_name (str): 划分方案名称
        save_dir (str): 保存目录
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    # ---- 子图1：训练损失变化曲线 ----
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-", linewidth=1.5, label="训练损失")
    plt.xlabel("训练轮数", fontsize=12)
    plt.ylabel("损失值", fontsize=12)
    plt.title(f"训练损失变化曲线（{split_name}）", fontsize=13)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    # ---- 子图2：训练集与测试集准确率对比曲线 ----
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], "g-", linewidth=1.5, label="训练集准确率")
    plt.plot(epochs, history["test_acc"], "r-", linewidth=1.5, label="测试集准确率")
    plt.xlabel("训练轮数", fontsize=12)
    plt.ylabel("准确率", fontsize=12)
    plt.title(f"训练集与测试集准确率对比曲线（{split_name}）", fontsize=13)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"训练曲线_{split_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  训练曲线图已保存：{save_path}")


def plot_confusion_matrix_heatmap(cm, split_name, save_dir, class_names):
    """
    绘制并保存40×40混淆矩阵热力图，标注全中文。

    参数:
        cm (np.ndarray): 混淆矩阵（40×40）
        split_name (str): 划分方案名称
        save_dir (str): 保存目录
        class_names (list[str]): 类别名称列表
    """
    plt.figure(figsize=(16, 14))

    # 归一化混淆矩阵（每行除以该行总和，得到召回率视角）
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)  # 处理除零

    im = plt.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.YlOrRd)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="归一化比例")

    # 坐标轴标签（每隔5个类别标注一次，避免拥挤）
    tick_positions = list(range(0, NUM_CLASSES, 5))
    tick_labels = [f"s{i+1}" for i in tick_positions]
    plt.xticks(tick_positions, tick_labels, fontsize=8)
    plt.yticks(tick_positions, tick_labels, fontsize=8)

    plt.xlabel("预测类别", fontsize=14)
    plt.ylabel("真实类别", fontsize=14)
    plt.title(f"测试集40分类混淆矩阵热力图（{split_name}）", fontsize=15)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"混淆矩阵_{split_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  混淆矩阵图已保存：{save_path}")


# ==================== 8. 错分样本分析模块 ====================
def analyze_misclassified(model, test_images, test_labels, dataloader, device,
                          save_dir, split_name):
    """
    筛选测试集中分类错误的样本，保存错分拼接图片，控制台中文打印详情。

    参数:
        model (nn.Module): 训练好的模型
        test_images (list): 原始测试图像列表
        test_labels (list): 原始测试标签列表
        dataloader (DataLoader): 测试集数据加载器
        device (torch.device): 运行设备
        save_dir (str): 保存目录
        split_name (str): 划分方案名称
    """
    print(f"\n【错分样本分析模块 — {split_name}】")

    model.eval()
    all_preds = []
    all_labels_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels_list.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels_list = np.array(all_labels_list)

    # 找到所有错分样本的索引
    mis_indices = np.where(all_preds != all_labels_list)[0]

    print(f"测试集共{len(all_labels_list)}个样本，错分{len(mis_indices)}个，"
          f"错误率{len(mis_indices)/len(all_labels_list)*100:.2f}%")
    print(f"\n{'='*50}")
    print(f"{'样本序号':<10}{'真实文件夹':<12}{'预测文件夹':<12}")
    print("-" * 34)

    mis_images = []
    for idx in mis_indices:
        true_cls = all_labels_list[idx]
        pred_cls = all_preds[idx]
        true_folder = f"s{true_cls + 1}"
        pred_folder = f"s{pred_cls + 1}"
        print(f"{idx:<10}{true_folder:<12}{pred_folder:<12}")
        mis_images.append(test_images[idx])

    print(f"{'='*50}")

    # 保存错分样本拼接图（最多显示40张）
    if len(mis_images) > 0:
        n_display = min(len(mis_images), 40)
        cols = 8
        rows = (n_display + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = axes.flatten() if n_display > 1 else [axes]

        for i in range(n_display):
            axes[i].imshow(mis_images[i], cmap="gray")
            axes[i].set_title(
                f"真:s{all_labels_list[mis_indices[i]]+1}\n"
                f"预:s{all_preds[mis_indices[i]]+1}",
                fontsize=6
            )
            axes[i].axis("off")

        # 隐藏多余的子图
        for i in range(n_display, len(axes)):
            axes[i].axis("off")

        plt.suptitle(f"测试集错分样本汇总（{split_name}）", fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"错分样本_{split_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  错分样本图已保存：{save_path}")
    else:
        print("\n  [提示] 无错分样本，不生成错分样本图。")


# ==================== 9. 传统机器学习对比模块（PCA+SVM）====================
def pca_svm_classification(train_images, train_labels, test_images, test_labels,
                           split_name, save_dir, n_components=80):
    """
    使用主成分分析(PCA)降维 + 支持向量机(SVM)进行人脸分类。
    作为传统机器学习方案的对比基准。

    参数:
        train_images (list): 训练集图像
        train_labels (list): 训练集标签
        test_images (list): 测试集图像
        test_labels (list): 测试集标签
        split_name (str): 划分方案名称
        save_dir (str): 保存目录
        n_components (int): PCA保留的主成分数量

    返回:
        accuracy (float): 测试集准确率
    """
    print(f"\n【传统机器学习对比模块 — PCA+SVM — {split_name}】")

    # 1. 将图像展平为向量
    X_train = np.array([img.reshape(-1) for img in train_images], dtype=np.float64)
    X_test = np.array([img.reshape(-1) for img in test_images], dtype=np.float64)
    y_train = np.array(train_labels, dtype=np.int32)
    y_test = np.array(test_labels, dtype=np.int32)

    print(f"  原始特征维度：{X_train.shape[1]}（{92}×{112}像素）")

    # 2. 数据标准化（归一化到零均值单位方差）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. PCA降维
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    explained_var = np.sum(pca.explained_variance_ratio_) * 100
    print(f"  PCA降维后特征维度：{n_components}")
    print(f"  前{n_components}个主成分累计方差解释率：{explained_var:.2f}%")

    # 4. SVM分类器（使用RBF径向基核函数）
    svm = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
              random_state=42)
    svm.fit(X_train_pca, y_train)

    # 5. 预测与评估
    y_pred = svm.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro",
                                      zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro",
                                zero_division=0)

    print(f"\n  >>> PCA+SVM总体准确率：{accuracy:.4f}（{accuracy*100:.2f}%）")
    print(f"  >>> PCA+SVM宏平均精确率：{precision_macro:.4f}")
    print(f"  >>> PCA+SVM宏平均召回率：{recall_macro:.4f}")

    # 绘制PCA特征可视化
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_,
            color="steelblue", alpha=0.7)
    plt.xlabel("主成分序号", fontsize=12)
    plt.ylabel("方差解释率", fontsize=12)
    plt.title(f"PCA各主成分方差解释率分布（{split_name}）", fontsize=13)
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(save_dir, f"PCA方差解释率_{split_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  PCA方差图已保存：{save_path}")

    return accuracy


# ==================== 10. 主流程编排模块 ====================
def run_experiment(train_images, train_labels, test_images, test_labels,
                   split_name, save_dir):
    """
    运行一轮完整实验：深度学习ResNet18训练+评估+可视化。

    参数:
        train_images, train_labels: 训练集
        test_images, test_labels: 测试集
        split_name (str): 方案名称（如"5比5划分"、"7比3划分"）
        save_dir (str): 结果保存目录

    返回:
        result_dict (dict): 实验结果汇总
    """
    print(f"\n{'#'*60}")
    print(f"# 运行实验：{split_name}")
    print(f"{'#'*60}")

    # 创建数据加载器
    train_dataset = ORLFaceDataset(train_images, train_labels, is_train=True)
    test_dataset = ORLFaceDataset(test_images, test_labels, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    # 构建模型
    model = ResNet18(num_classes=NUM_CLASSES, in_channels=1).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型总参数量：{total_params:,}，可训练参数量：{trainable_params:,}")

    # 训练模型
    save_name = f"最优模型_{split_name}"
    history = train_model(model, train_loader, test_loader, DEVICE, save_name=save_name)

    # 加载完整模型进行评估
    best_model_path = os.path.join(save_dir, f"{save_name}.pth")
    model = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
    model.eval()

    # 计算全部指标
    class_names = [f"s{i+1}" for i in range(NUM_CLASSES)]
    all_preds, all_labels, acc, precision_arr, recall_arr = compute_all_metrics(
        model, test_loader, DEVICE, class_names
    )

    # 绘制训练曲线
    plot_training_curves(history, split_name, save_dir)

    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix_heatmap(cm, split_name, save_dir, class_names)

    # 错分样本分析
    # 需要使用原始图像（未归一化的）进行可视化
    raw_test_images = test_images.copy()
    analyze_misclassified(model, raw_test_images, test_labels, test_loader,
                          DEVICE, save_dir, split_name)

    return {
        "split_name": split_name,
        "accuracy": acc,
        "precision_macro": np.mean(precision_arr),
        "recall_macro": np.mean(recall_arr),
        "history": history
    }


def main():
    """主函数：串联全部实验流程"""
    print("\n" + "=" * 70)
    print("开始执行ORL人脸多分类任务 — 全部实验流程")
    print("=" * 70)

    # 创建实验结果输出目录
    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f"\n实验结果输出目录：{RESULT_DIR}")

    # ---- 步骤1：加载数据集 ----
    images, labels, folder_map = load_orl_dataset(DATA_ROOT)

    # ---- 步骤2 & 3：两套数据划分方案 ----
    # 方案一：每类前5张训练，后5张测试（5:5）
    train_imgs_55, train_lbls_55, test_imgs_55, test_lbls_55 = \
        split_dataset_by_folder(images, labels, train_count=5)

    # 方案二：每类前7张训练，后3张测试（7:3）
    train_imgs_73, train_lbls_73, test_imgs_73, test_lbls_73 = \
        split_dataset_by_folder(images, labels, train_count=7)

    # ---- 步骤4：深度学习实验 ----
    dl_results = {}

    # 方案一：5:5划分
    result_55 = run_experiment(
        train_imgs_55, train_lbls_55, test_imgs_55, test_lbls_55,
        split_name="5比5划分", save_dir=RESULT_DIR
    )
    dl_results["5比5划分"] = result_55

    # 方案二：7:3划分
    result_73 = run_experiment(
        train_imgs_73, train_lbls_73, test_imgs_73, test_lbls_73,
        split_name="7比3划分", save_dir=RESULT_DIR
    )
    dl_results["7比3划分"] = result_73

    # ---- 步骤5：传统机器学习对比实验（PCA+SVM）----
    svm_accuracy_55 = pca_svm_classification(
        train_imgs_55, train_lbls_55, test_imgs_55, test_lbls_55,
        "5比5划分", RESULT_DIR
    )

    svm_accuracy_73 = pca_svm_classification(
        train_imgs_73, train_lbls_73, test_imgs_73, test_lbls_73,
        "7比3划分", RESULT_DIR
    )

    # ---- 步骤6：综合对比表格 ----
    print(f"\n\n{'='*70}")
    print(">>> 综合实验结果对比汇总 <<<")
    print(f"{'='*70}")

    # 表头
    header = f"{'技术方案':<30}{'5比5划分准确率':<20}{'7比3划分准确率':<20}"
    print(header)
    print("-" * 70)

    # 深度学习ResNet18
    dl_acc_55 = result_55["accuracy"]
    dl_acc_73 = result_73["accuracy"]
    print(f"{'残差网络18层(ResNet18)深度学习':<30}"
          f"{dl_acc_55:.4f}（{dl_acc_55*100:.2f}%）{'':<5}"
          f"{dl_acc_73:.4f}（{dl_acc_73*100:.2f}%）")

    # 传统PCA+SVM
    print(f"{'主成分分析(PCA)+支持向量机(SVM)':<30}"
          f"{svm_accuracy_55:.4f}（{svm_accuracy_55*100:.2f}%）{'':<5}"
          f"{svm_accuracy_73:.4f}（{svm_accuracy_73*100:.2f}%）")

    print("-" * 70)

    # 打印精度对比表（适合复制粘贴到报告）
    print(f"\n{'='*70}")
    print(">>> 精度对比表格（可直接复制粘贴至课程设计报告）<<<")
    print(f"{'='*70}")
    table = f"""
| 技术方案                                 | 5比5划分识别精度 | 7比3划分识别精度 |
|-----------------------------------------|:---------------:|:---------------:|
| 残差网络18层(ResNet18)深度学习            | {dl_acc_55*100:.2f}% | {dl_acc_73*100:.2f}% |
| 主成分分析(PCA)+支持向量机(SVM)传统算法    | {svm_accuracy_55*100:.2f}% | {svm_accuracy_73*100:.2f}% |
"""
    print(table)
    print(f"{'='*70}")

    print(f"\n全部实验完成！所有结果文件已保存至：{RESULT_DIR}")
    print("请查看目录内的训练曲线图、混淆矩阵图、错分样本图、PCA方差图。")

    # 列出输出目录文件
    print(f"\n实验结果目录文件清单：")
    for f in sorted(os.listdir(RESULT_DIR)):
        fpath = os.path.join(RESULT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  - {f}（{size_kb:.1f} KB）")


if __name__ == "__main__":
    main()
