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


class 无标签眼底图像数据集(torch.utils.data.Dataset):
    # 读取图像后对每一张图像进行两种不同的随机图像变换并返回两个变换结果，作为正对样本输入
    def __init__(self, 文件路径, 图像变换):
        self.文件路径 = 文件路径 # 保存图像的文件目录
        self.图像变换 = 图像变换
        self.图像列表 = os.listdir(self.文件路径) # 得到图像名称列表

    def __len__(self):
        return len(self.图像列表) # 返回数据集的大小

    def __getitem__(self, item):
        # 根据item返回对应的图像
        图像名 = self.图像列表[item] # 根据索引取出对应的图像
        图像路径 = os.path.join(self.文件路径, 图像名) # 得到图像的路径
        图像 = Image.open(图像路径).convert('RGB') # todo 三通道读取吗？读取图像并转为RGB格式

        # 对同一图片应用不同图像变换得到两个结果，作为正对样本输入
        图像变换结果1 = self.图像变换(图像)
        图像变换结果2 = self.图像变换(图像)
        return 图像变换结果1, 图像变换结果2


# train stage one
def 无标签训练(命令行参数):
    #  init seed 初始化随机种子
    全部随机数种子 = 222

    # 下面似乎都是控制生成相同的随机数
    random.seed(全部随机数种子)
    np.random.seed(全部随机数种子)
    torch.manual_seed(全部随机数种子)
    torch.cuda.manual_seed_all(全部随机数种子)
    torch.cuda.manual_seed(全部随机数种子)
    np.random.seed(全部随机数种子)
    os.environ['PYTHONHASHSEED'] = str(全部随机数种子) # 禁止哈希随机化，使实验可复现

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

    无标签训练数据集 = 无标签眼底图像数据集(文件路径='UnlabeledTrainDataset/OCTA_6M/Projection Maps/OCTA(FULL)', 图像变换=随机图像变换["训练集"])
    # win可能多线程报错，num_workers最多和CPU的超线程数目相同，若报错设为0
    # 每次输出一个批次的数据，形式取决于读图像时
    # todo nw = min([os.cpu_count(), 命令行参数.batch_size if 命令行参数.batch_size > 1 else 0, 8])  # number of workers
    训练数据 = torch.utils.data.DataLoader(无标签训练数据集, batch_size=命令行参数.unlabeled_data_batch_size, shuffle=True, num_workers=2, drop_last=True)

    # 用于无标签数据训练的模型
    网络模型 = SimCLRModel.无监督simCLRresnet50()
    残差网络预训练权重路径 = "./weight/resnet50-19c8e357.pth"
    assert os.path.exists(残差网络预训练权重路径), "文件 {} 不存在.".format(残差网络预训练权重路径)
    残差模型参数 = torch.load(残差网络预训练权重路径, map_location=硬件设备) # 字典形式读取Res50的权重
    # Resnet50Model
    simCLR模型参数 = 网络模型.state_dict() # 自己设计的模型参数字典

    # 我的模型只使用Res50模型从开头的第一个卷积层到最后一个全局自适应池化层作为编码器，所以遍历Res50的参数并赋值给我模型中对应名称的参数
    编码器参数 = {键:值 for 键,值 in 残差模型参数.items() if 键 in simCLR模型参数.keys()}
    simCLR模型参数.update(编码器参数) # 更新我模型的参数，实际就是使用Res50模型参数
    网络模型.load_state_dict(simCLR模型参数) # 加载模型参数
    网络模型.to(硬件设备)
    训练损失函数 = SimCLRModel.对比损失函数().to(硬件设备) # 使用余弦相似度损失函数
    优化器 = torch.optim.Adam(网络模型.parameters(), lr=1e-3, weight_decay=1e-6)

    for 当前训练周期 in range(1, 命令行参数.unlabeled_train_max_epoch + 1):
        网络模型.train()  # 开始训练
        全部损失 = 0
        # 每一批数据训练。enumerate可以在遍历元素的同时输出元素的索引
        训练循环 = tqdm(enumerate(训练数据), total=len(训练数据), leave=True)
        for 训练批次, (图像变换1, 图像变换2) in 训练循环:
            图像变换1, 图像变换2 = 图像变换1.to(硬件设备), 图像变换2.to(硬件设备)

            _, 特征1 = 网络模型(图像变换1)  # 特征1是最终输出特征 形状[批量大小 * 特征维度]
            _, 特征2 = 网络模型(图像变换2)  # 特征2是最终输出特征 形状[批量大小 * 特征维度]

            # 计算特征1和特征2之间的余弦相似度
            训练损失 = 训练损失函数(特征1, 特征2, 命令行参数.unlabeled_data_batch_size)
            优化器.zero_grad()
            训练损失.backward()
            优化器.step()

            # print("训练迭代次数", 当前训练周期, "训练批次", 训练批次, "损失:", 训练损失.detach().item())
            全部损失 += 训练损失.detach().item()
            训练循环.set_description(f'训练迭代周期 [{当前训练周期}/{命令行参数.unlabeled_train_max_epoch}]') # 设置进度条标题
            训练循环.set_postfix(训练损失 = 训练损失.detach().item()) # 每一批训练都更新损失

        # 参数'a',打开一个文件用于追加。若该文件已存在，文件指针将会放在文件的结尾，新的内容将会被写入到已有内容之后。若该文件不存在，创建新文件进行写入。
        with open(os.path.join("Weight", "stage1_loss.txt"), 'a') as f:
            # 将损失写入文件并用逗号分隔
            f.write(str(全部损失 / len(无标签训练数据集) * 命令行参数.unlabeled_data_batch_size) + ", ")

        if 当前训练周期 % 1 == 0:
            # todo 保存模型，按周期还是损失呢？
            torch.save(网络模型.state_dict(), os.path.join("Weight", "model_stage1_epoch" + str(当前训练周期) + ".pth"))


if __name__ == '__main__':
    # 命令行参数或许可以写在函数外，全局形式
    # 设置一个参数解析器
    命令行参数解析器 = argparse.ArgumentParser(description='无标签数据训练 SimCLR')

    # 添加无标签数据训练时的参数
    命令行参数解析器.add_argument('--unlabeled_data_batch_size', default=2, type=int, help='')
    命令行参数解析器.add_argument('--unlabeled_train_max_epoch', default=5, type=int, help='')

    # 获取命令行传入的参数
    无标签训练命令行参数 = 命令行参数解析器.parse_args()
    无标签训练(无标签训练命令行参数)











'''
def 有标签微调(args):
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

    # load dataset for train and eval
    train_dataset = CIFAR10(root='dataset', train=True, transform=配置.train_transform, download=True)
    train_data = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
    eval_dataset = CIFAR10(root='dataset', train=False, transform=配置.test_transform, download=True)
    eval_data = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    有标签训练数据集 = datasets.ImageFolder(root="LabeledDataset/Train", transform=随机图像变换["测试集"])



    # todo 注意修改模型
    model =SimCLRModel.有监督simCLRresnet50()
    model.load_state_dict(torch.load(args.pre_model, map_location='cpu'),strict=False)
    loss_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)

    os.makedirs(配置.save_path, exist_ok=True)
    for epoch in range(1,args.max_epoch+1):
        model.train()
        total_loss=0
        for batch, (data, target) in enumerate(train_data):
            data, target = data.to(DEVICE), target.to(DEVICE)
            pred = model(data)

            loss = loss_criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("epoch",epoch,"loss:", total_loss / len(train_dataset)*args.batch_size)
        with open(os.path.join(配置.save_path, "stage2_loss.txt"), "a") as f:
            f.write(str(total_loss / len(train_dataset)*args.batch_size) + " ")

        if epoch % 5==0:
            torch.save(model.state_dict(), os.path.join(配置.save_path, 'model_stage2_epoch' + str(epoch) + '.pth'))

            model.eval()
            with torch.no_grad():
                print("batch", " " * 1, "top1 acc", " " * 1, "top5 acc")
                total_loss, total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0.0, 0
                for batch, (data, target) in enumerate(train_data):
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    pred = model(data)

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
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--batch_size', default=200, type=int, help='')
    parser.add_argument('--max_epoch', default=200, type=int, help='')
    parser.add_argument('--pre_model', default=配置.pre_model, type=str, help='')

    args = parser.parse_args()
    有标签训练(args)
'''