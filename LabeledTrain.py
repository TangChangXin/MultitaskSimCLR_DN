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

    # load dataset for train and eval
    train_dataset = CIFAR10(root='dataset', train=True, transform=配置.train_transform, download=True)
    train_data = DataLoader(train_dataset, batch_size=命令行参数.batch_size, shuffle=True, num_workers=16, pin_memory=True)
    eval_dataset = CIFAR10(root='dataset', train=False, transform=配置.test_transform, download=True)
    eval_data = DataLoader(eval_dataset, batch_size=命令行参数.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    有标签训练数据集 = datasets.ImageFolder(root="LabeledDataset/Train", transform=随机图像变换["测试集"])
    # win可能多线程报错，num_workers最多和CPU的超线程数目相同，若报错设为0
    # todo nw = min([os.cpu_count(), 命令行参数.batch_size if 命令行参数.batch_size > 1 else 0, 8])  # number of workers
    有标签训练数据 = torch.utils.data.DataLoader(有标签训练数据集, batch_size=命令行参数.labeled_data_batch_size, shuffle=True, num_workers=0)



    # todo 注意修改模型
    model =SimCLRModel.有监督simCLRresnet50()
    model.load_state_dict(torch.load(命令行参数.pre_model, map_location='cpu'),strict=False)
    loss_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)

    os.makedirs(配置.save_path, exist_ok=True)
    for epoch in range(1,命令行参数.max_epoch+1):
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

        print("epoch",epoch,"loss:", total_loss / len(train_dataset)*命令行参数.batch_size)
        with open(os.path.join(配置.save_path, "stage2_loss.txt"), "a") as f:
            f.write(str(total_loss / len(train_dataset)*命令行参数.batch_size) + " ")

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
    # 设置一个参数解析器
    命令行参数解析器 = argparse.ArgumentParser(description="无标签训练 SimCLR")

    # 添加有标签数据训练时的参数
    命令行参数解析器.add_argument('--labeled_data_batch_size', default=2, type=int, help='')
    命令行参数解析器.add_argument('--labeled_train_max_epoch', default=5, type=int, help='')
    命令行参数解析器.add_argument('--my_model', default=配置.pre_model, type=str, help='') # 加载无标签数据预训练的模型

    # 获取命令行传入的参数
    有标签训练命令行参数 = 命令行参数解析器.parse_args()
    有标签训练(有标签训练命令行参数)