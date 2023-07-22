import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
  'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
  'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
  'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
      super(ChannelAttention, self).__init__()
      self.avg_pool = nn.AdaptiveAvgPool2d(1)
      self.max_pool = nn.AdaptiveMaxPool2d(1)

      self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
      self.relu1 = nn.ReLU()
      self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

      self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
      max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
      out = avg_out + max_out
      return self.sigmoid(out)


class SpatialAttention(nn.Module):
  def __init__(self, kernel_size=7):
    super(SpatialAttention, self).__init__()

    assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
    padding = 3 if kernel_size == 7 else 1

    self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    avg_out = torch.mean(x, dim=1, keepdim=True)
    max_out, _ = torch.max(x, dim=1, keepdim=True)
    x = torch.cat([avg_out, max_out], dim=1)
    x = self.conv1(x)
    return self.sigmoid(x)


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
  """3x3 convolution with padding"""
  # original padding is 1; original dilation is 1
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=dilation, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride, dilation)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)


    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)


    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1,reduction=16):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    # original padding is 1; original dilation is 1
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)


    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)


    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNet(nn.Module):

  def __init__(self, block, layers, last_conv_stride=2, last_conv_dilation=1):

    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=last_conv_stride, dilation=last_conv_dilation)

    self.ca = ChannelAttention(self.inplanes)
    self.sa = SpatialAttention()
    self.ca1 = ChannelAttention(self.inplanes)
    self.sa1 = SpatialAttention()

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, dilation))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.ca(x) * x
    x = self.sa(x) * x

    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.ca1(x) * x
    x = self.sa1(x) * x


    return x


def remove_fc(state_dict):
  """Remove the fc layer parameters from state_dict."""
  # for key, value in state_dict.items():
  for key, value in list(state_dict.items()):
    if key.startswith('fc.'):
      del state_dict[key]
  return state_dict


def resnet18(pretrained=False, **kwargs):
  """Constructs a ResNet-18 model.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
  if pretrained:
    model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet18'])))
  return model


def resnet34(pretrained=False, **kwargs):
  """Constructs a ResNet-34 model.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
  if pretrained:
    model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet34'])))
  return model


def resnet50(pretrained=False, **kwargs):
  """Constructs a ResNet-50 model.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model_path = R'D:\CmCEN-main\CmCEN\resnet50-0676ba61.pth'  # 预训练参数的位置
  model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs) #改写过的
  model_dict = model.state_dict()  # 网络层的参数
  pretrained_dict = torch.load(model_path)  # torch.load得到是字典，我们需要的是state_dict下的参数
  pretrained_dict = {k.replace('module.', ''): v for k, v in
                     pretrained_dict.items()}  # 因为pretrained_dict得到module.conv1.weight，但是自己建的model无module，只是conv1.weight，所以改写下。
  pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 只保留预训练模型中，自己建的model有的参数
  model_dict.update(pretrained_dict)  # 将预训练的值，更新到自己模型的dict中
  model.load_state_dict(model_dict, strict=False)  # model加载dict中的数据，更新网络的初始值

  if pretrained:
    # model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet50'])))
    model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet50'])))

  return model


def resnet101(pretrained=False, **kwargs):
  """Constructs a ResNet-101 model.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
  if pretrained:
    model.load_state_dict(
      remove_fc(model_zoo.load_url(model_urls['resnet101'])))
  return model


def resnet152(pretrained=False, **kwargs):
  """Constructs a ResNet-152 model.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
  if pretrained:
    model.load_state_dict(
      remove_fc(model_zoo.load_url(model_urls['resnet152'])))
  return model
