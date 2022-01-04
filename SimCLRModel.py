# net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# 残差块定义来源https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/Test5_resnet/model.py

class BasicBlock(nn.Module):
    # 用于18层和34层的残差块
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    用于50层及以上的残差块
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


# 阶段1，无标签数据训练
class SimCLR无标签训练阶段(nn.Module):
    """
    第一个返回值是编码器得到的特征，第二个返回值是模型最终输出的向量
    """

    def __init__(self,
                 block,
                 blocks_num,
                 include_top=True,
                 groups=1,
                 width_per_group=64,
                 特征维度=128):
        super(SimCLR无标签训练阶段, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.groups = groups
        self.width_per_group = width_per_group

        # 编码器
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            # 最后一个池化层输出直接送入投影模块,不需要接全连接层
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)

        # projection head 投影 对应论文结构中的 g(·)
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False),
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, 特征维度, bias=True))

        # 投影结构的权重初始化
        for m in self.g.modules():
            # isinstance判断实参是否为同类型，认为子类和父类是同类型，考虑了继承关系
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = [block(self.in_channel,
                        channel,
                        downsample=downsample,
                        stride=stride,
                        groups=self.groups,
                        width_per_group=self.width_per_group)]
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.include_top:
            x = self.avgpool(x)
        # 经过上方的运算，输出Resnet提取的特征
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)  # 最终输出特征向量大小 [批量大小, 特征维度]
        # -1表示按照最后一个维度进行L2正则化,就是最后一个维度的索引可以改变，其他维度的索引不变。
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


# 原本没有“加载预训练权重”这个参数，我自己加的
def simCLRresnet34():
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return SimCLR无标签训练阶段(BasicBlock, [3, 4, 6, 3])

def 无监督simCLRresnet50():
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return SimCLR无标签训练阶段(Bottleneck, [3, 4, 6, 3])

def resnet101():
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return SimCLR无标签训练阶段(Bottleneck, [3, 4, 23, 3])

def resnext50_32x4d():
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return SimCLR无标签训练阶段(Bottleneck, [3, 4, 6, 3],
                  groups=groups,
                  width_per_group=width_per_group)

def resnext101_32x8d():
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return SimCLR无标签训练阶段(Bottleneck, [3, 4, 23, 3],
                  groups=groups,
                  width_per_group=width_per_group)





# 阶段2，有标签数据训练
class SimCLR有标签微调阶段(torch.nn.Module):
    def __init__(self,
                 类别数目,
                 block,
                 blocks_num,
                 include_top=True,
                 groups=1,
                 width_per_group=64,
                 特征维度=128, # todo 原始方法把投影层去掉了所以这里没用到这形参，我自己的方法可能需要用到它
                 ):
        super(SimCLR有标签微调阶段, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.groups = groups
        self.width_per_group = width_per_group

        # 编码器
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        # todo 自己的模型需要修改
        if self.include_top:
            # 最后一个池化层输出直接送入投影模块,不需要接全连接层
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)

        # 写在上面这些层定义的后面，就可以实现对上面这些权重的冻结
        for 模型参数 in self.parameters():
            模型参数.requires_grad = False

        # 分类器
        全连接输入大小 = 2048 # 后续可能要改成自己模型适应的大小
        self.全连接 = nn.Linear(全连接输入大小, 类别数目, bias=True)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = [block(self.in_channel,
                        channel,
                        downsample=downsample,
                        stride=stride,
                        groups=self.groups,
                        width_per_group=self.width_per_group)]
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # todo 我的方法可能需要去掉最后一个全局池化层，目前先保留看看
        if self.include_top:
            x = self.avgpool(x)
        # 经过上方的运算，输出Resnet提取的特征
        feature = torch.flatten(x, start_dim=1)
        out = self.全连接(feature) # 输出分类预测结果
        return out


def 有监督simCLRresnet50(类别数目):
    return SimCLR有标签微调阶段(类别数目, Bottleneck, [3, 4, 6, 3])




# 对比损失函数，输出结果之间的余弦相似性最大化
class 对比损失函数(torch.nn.Module):
    def __init__(self):
        super(对比损失函数, self).__init__()

    def forward(self, 输出1, 输出2, 批量大小, temperature=0.5):
        # 模型输出形状是[批量大小, 特征维度]，输出中的一个行向量对应一张图像的输出结果
        # 输入1 是第一种图像变换得到输出，输入2 是第二种图像变换得到的输出，两者都经过了按行l2正则化，后续算余弦相似性时不需要计算两个向量的模了
        完整输出 = torch.cat([输出1, 输出2], dim=0)  # 沿着维度0拼接，形状[2*批量大小, 特征维度]

        # 完整输出.t().contiguous()矩阵转置后以行优先形式在内存中连续存储。
        # 输出1和输出2拼接得到完整输出，完整输出和自己的转置相乘就得到了余弦相似度矩阵，对角线表示每个图像自己和自己的相似性，需要去掉
        相似度矩阵 = torch.exp(torch.mm(完整输出, 完整输出.t().contiguous()) / temperature)  # 形状[2*批量大小, 2*批量大小]

        # torch.ones_like()根据给定的张量生成全是1的张量，torch.eye()生成指定大小和类型的对角线全是1的2维张量。
        # 掩码是对角线全False，其余全True的张量
        掩码 = (torch.ones_like(相似度矩阵) - torch.eye(2 * 批量大小, device=相似度矩阵.device)).bool()

        # 根据给定的掩码张量的二元值，取出输入张量中对应位置的值，这里就是去掉对角线的值，返回一个行优先排列的一维张量。
        # 然后改变形状，得到真正的余弦相似性矩阵，
        相似度矩阵 = 相似度矩阵.masked_select(掩码).view(2 * 批量大小, -1)  # 形状[2*批量大小, 2*批量大小-1]

        # 计算正对样本之间的相似度
        正对样本余弦相似度 = torch.exp(torch.sum(输出1 * 输出2, dim=-1) / temperature)  # 形状[1, 批量大小]
        # 想来想去，它这个损失只计算了正对样本的相似度，还有正对样本中的某一张和其余负样本的相似度，并不是计算了所有输出结果两两之间的相似度。
        正对样本余弦相似度 = torch.cat([正对样本余弦相似度, 正对样本余弦相似度], dim=0)  # 拼接是因为正样本对中的两个结果需要变换位置。形状[1, 2*批量大小]
        return (- torch.log(正对样本余弦相似度 / 相似度矩阵.sum(dim=-1))).mean()


if __name__ == "__main__":
    # 无监督训练模型 = SimCLR无标签阶段(Bottleneck, [3, 4, 6, 3])
    无监督训练模型 = 无监督simCLRresnet50()
    无监督训练模型.to(torch.device('cuda:0'))
    # print(model)
    # summary(无监督训练模型, input_size=(3, 112, 112), batch_size=4)
    # for name, module in 无监督训练模型.named_children():
    #     print(name,module)

    # for module in 无监督训练模型.g.modules():
    #     if isinstance(module, nn.Linear):
    #         print("相同")
            # nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        # elif isinstance(module, nn.BatchNorm2d):
        #     print("不相同")
            # nn.init.constant_(module.weight, 1)
            # nn.init.constant_(module.bias, 0)
        # print(module)

    # for param in 无监督训练模型.named_parameters():
    #     print(param[0])
    # print(1)
    # print(2)
    # for 参数 in 无监督训练模型.parameters():
    #     print(参数)


    有监督微调模型 = SimCLR有标签微调阶段(2, Bottleneck, [3, 4, 6, 3])
    for 参数 in 有监督微调模型.named_parameters():
        print(参数[0], '\t', 参数[1].requires_grad)