import torch, argparse, os, random
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
import SimCLRModel
from tqdm import tqdm
from torch.backends.cudnn import deterministic


随机图像变换 = {
    "训练集": transforms.Compose([
        transforms.RandomResizedCrop(32),  # 随机选取图像中的某一部分然后再缩放至指定大小
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        # 修改亮度、对比度和饱和度
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # 随机应用添加的各种图像变换
        transforms.RandomGrayscale(p=0.2),  # todo 随机灰度化，但我本来就是灰度图啊
        transforms.ToTensor(),  # 转换为张量且维度是[C, H, W]
        # 三通道归一化
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]),
    "测试集": transforms.Compose([
        transforms.ToTensor(),
        # 三通道归一化
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
}



def 有标签训练(命令行参数):
    #  init seed 初始化随机种子
    全部随机数种子 = 222

    # 下面似乎都是控制生成相同的随机数
    random.seed(全部随机数种子)
    np.random.seed(全部随机数种子) # todo
    torch.manual_seed(全部随机数种子)
    torch.cuda.manual_seed_all(全部随机数种子)
    torch.cuda.manual_seed(全部随机数种子)
    np.random.seed(全部随机数种子) # todo

    # 禁止哈希随机化，使实验可复现
    os.environ['PYTHONHASHSEED'] = str(全部随机数种子)

    # 设置训练使用的设备
    if torch.cuda.is_available():
        硬件设备 = torch.device("cuda:0")
        # 保证每次返回的卷积算法将是确定的，如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的。
        torch.backends.cudnn.deterministic = True
        if torch.backends.cudnn.deterministic:
            print("确定卷积算法")
        torch.backends.cudnn.benchmark = False  # 为每层搜索适合的卷积算法实现，加速计算
    else:
        硬件设备 = torch.device("cpu")
    print("训练使用设备", 硬件设备)

    # 加载训练数据集和测试数据集
    有标签训练数据集 = datasets.ImageFolder(root="LabeledDataset/Train", transform=随机图像变换["测试集"])
    # win可能多线程报错，num_workers最多和CPU的超线程数目相同，若报错设为0
    # todo 线程数 = min([os.cpu_count(), 命令行参数.batch_size if 命令行参数.batch_size > 1 else 0, 8])  # number of workers
    有标签训练数据 = torch.utils.data.DataLoader(有标签训练数据集, batch_size=命令行参数.labeled_data_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    有标签测试数据集 = datasets.ImageFolder(root="LabeledDataset/Validate", transform=随机图像变换["测试集"])
    有标签测试数据 = torch.utils.data.DataLoader(有标签测试数据集, batch_size=命令行参数.labeled_data_batch_size, shuffle=True, num_workers=0, pin_memory=True)


    分类模型 =SimCLRModel.有监督simCLRresnet50(2) # 生成模型，需传入分类数目
    分类模型.load_state_dict(torch.load(命令行参数.pre_model, map_location=硬件设备),strict=False)
    损失函数 = torch.nn.CrossEntropyLoss()
    优化器 = torch.optim.Adam(分类模型.fc.parameters(), lr=1e-3, weight_decay=1e-6)

    # 开始训练
    for 当前训练周期 in range(1,命令行参数.max_epoch+1):
        分类模型.train()
        全部损失=0
        # 每一批数据训练。enumerate可以在遍历元素的同时输出元素的索引
        训练循环 = tqdm(enumerate(有标签训练数据), total=len(有标签训练数据), leave=True)
        for 当前批次, (图像数据, 标签) in 训练循环:
            图像数据, 标签 = 图像数据.to(硬件设备), 标签.to(硬件设备)
            预测值 = 分类模型(图像数据)

            训练损失 = 损失函数(预测值, 标签)
            优化器.zero_grad()
            训练损失.backward()
            优化器.step()

            全部损失 += 训练损失.item()
            全部损失 += 训练损失.detach().item()
            训练循环.set_description(f'训练迭代周期 [{当前训练周期}/{命令行参数.unlabeled_train_max_epoch}]')  # 设置进度条标题
            训练循环.set_postfix(训练损失=训练损失.detach().item())  # 每一批训练都更新损失

        # 每一批数据训练完都会更新损失值
        with open(os.path.join("Weight", "stage2_loss.txt"), "a") as f:
            f.write(str(全部损失 / len(有标签训练数据集) * 命令行参数.labeled_data_batch_size) + "，")

        if 当前训练周期 % 5==0:
            # todo 保存模型，按周期还是损失呢？
            torch.save(分类模型.state_dict(), os.path.join("Weight", 'model_stage2_epoch' + str(当前训练周期) + '.pth'))

            # 测试模型效果
            分类模型.eval()
            with torch.no_grad():
                print("batch", " " * 1, "top1 acc", " " * 1, "top5 acc")
                全部损失, total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0.0, 0
                for batch, (data, target) in enumerate(train_data):
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    pred = 分类模型(data)

                    total_num += data.size(0)
                    prediction = torch.argsort(pred, dim=-1, descending=True)
                    top1_acc = torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    top5_acc = torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    total_correct_1 += top1_acc
                    total_correct_5 += top5_acc

                    print("  {:02}  ".format(batch + 1), " {:02.3f}%  ".format(top1_acc / data.size(0) * 100),
                          "{:02.3f}%  ".format(top5_acc / data.size(0) * 100))

                print("all eval dataset:", "top1 acc: {:02.3f}%".format(total_correct_1 / total_num * 100),
                          "top5 acc:{:02.3f}%".format(total_correct_5 / total_num * 100))
                with open(os.path.join(配置.save_path, "stage2_top1_acc.txt"), "a") as f:
                    f.write(str(total_correct_1 / total_num * 100) + " ")
                with open(os.path.join(配置.save_path, "stage2_top5_acc.txt"), "a") as f:
                    f.write(str(total_correct_5 / total_num * 100) + " ")


if __name__ == '__main__':
    # 设置一个参数解析器
    命令行参数解析器 = argparse.ArgumentParser(description="无标签训练 SimCLR")

    # 添加有标签数据训练时的参数
    命令行参数解析器.add_argument('--labeled_data_batch_size', default=2, type=int, help='')
    命令行参数解析器.add_argument('--labeled_train_max_epoch', default=5, type=int, help='')
    命令行参数解析器.add_argument('--my_model', default=配置.pre_model, type=str, help='') # 加载无标签数据预训练的模型

    # 获取命令行传入的参数
    有标签训练命令行参数 = 命令行参数解析器.parse_args()
    有标签训练(有标签训练命令行参数)