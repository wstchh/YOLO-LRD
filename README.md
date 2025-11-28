# YOLO-LRD
This is the official code implementation of YOLO-LRD for the *Measurement* journal paper: **" YOLO-LRD: Revisiting Multi-Scale and Attention Fusion for Lightweight Road Damage Detection"**




## 1-CSFP module
```python

class CSFPBlock(nn.Module):
    def __init__(self, c1, cm, c2, k=3, n=5, lightconv=False, shortcut=True, act=nn.ReLU()):
     super().__init__()
     block = LightConv if lightconv else DSConv
     self.m = nn.ModuleList(block(cm, cm, k=k, act=act) for i in range(n))
     
     self.agg_conv = Conv(c1 + n*cm, c2, 1, 1, act=act)  
    
     self.add = shortcut and c1 == c2
    
    def forward(self, x):
     y = [x]
     y.extend(m(y[-1]) for m in self.m)
     y = self.agg_conv(torch.cat(y, 1))
     return y + x if self.add else y


class C2fCSFP(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  
        self.cv1 = Conv(c1, 2*self.c, 1, 1)
        self.cv2 = Conv((2+n) * self.c, c2, 1)  
        self.m = nn.Sequential(*(CSFPBlock(self.c, self.c, self.c) for _ in range(n)))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
```

## 2-RSAHA mechanism
```python
import torch
from torch import nn


class SACA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.convs = nn.ModuleList()
        for k_size in [3, 5, 7]:
            padding = (k_size - 1) // 2
            self.convs.append(nn.Conv1d(1, 1, kernel_size=k_size, padding=padding, bias=False))
        self.fc = nn.Conv2d(channels * len(self.convs), channels, 1, 1, 0, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        pooled = self.pool(x)
        b, c, h, w = pooled.shape
        combined = []
        for conv in self.convs:
            pooled_temp = conv(pooled.view(b, c, -1).transpose(-1,-2))
            pooled_y = pooled_temp.view(b, -1).unsqueeze(-1).unsqueeze(-1)    
            combined.append(pooled_y)
        combined = torch.cat(combined, dim=1)
        attention_weights = self.act(self.fc(combined))
        return x * attention_weights


class SASA(nn.Module):
    def __init__(self, kernel_sizes=(3, 5, 7)):
        super().__init__()
        self.convs = nn.ModuleList()
        for ks in kernel_sizes:
            padding = (ks - 1) // 2
            self.convs.append(nn.Conv2d(2, 1, ks, padding=padding, bias=False))
        # add learnable weight parameters.
        self.weights = nn.Parameter(torch.ones(len(kernel_sizes)), requires_grad=True)
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # normalize the weights using softmax.
        normalized_weights = torch.softmax(self.weights, dim=0)
        # compute the weighted sum.
        out = sum(w * conv(x_cat) for w, conv in zip(normalized_weights, self.convs))
        return x * self.act(out)


class RSAHA(nn.Module):
    # Residual Scale-Aware Hybrid Attention
    def __init__(self, channels, kernel_sizes=(3, 5, 7)):  
        super().__init__()
        self.rsaca = SACA(channels)
        self.rsasa = SASA(kernel_sizes)

    def forward(self, x):
        # overall residual
        x_rsaca = self.rsaca(x)
        x_rsasa = x + self.rsasa(x_rsaca)
        
        # step-wise residual
        # x_rsaca = x + self.rsaca(x)
        # x_rsasa = x_rsaca + self.rsasa(x_rsaca)
        return x_rsasa


if __name__ == '__main__':
    img = torch.rand(1, 16, 80, 80)
    net = RSAHA(channels=16)
    output = net(img)
    print(output.shape)

```

## 3-MSFAAF module 
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def add_conv(in_ch, out_ch, ksize, stride, leaky=True, groups=1):
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch,
                                       kernel_size=ksize,
                                       stride=stride,
                                       padding=pad,
                                       bias=False,
                                       groups=groups))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage


def multi_scale_conv(in_ch, out_ch):
    conv_3x3 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=in_ch)
    conv_5x5 = nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=1, padding=2, groups=in_ch)
    conv_7x7 = nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=1, padding=3, groups=in_ch)
    conv_9x9 = nn.Conv2d(in_ch, out_ch, kernel_size=9, stride=1, padding=4, groups=in_ch)
    return nn.ModuleList([conv_3x3, conv_5x5, conv_7x7, conv_9x9])


class ECA(nn.Module):
    def __init__(self, kernel_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, 1, c)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


class ESA(nn.Module):
    def __init__(self):
        super(ESA, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pool_out = torch.cat([avg_out, max_out], dim=1)
        
        spatial_att = self.conv(pool_out)
        spatial_att = self.sigmoid(spatial_att)

        return x * spatial_att


class MSFAAF(nn.Module):
    def __init__(self, level):
        super(MSFAAF, self).__init__()
        self.level = level
        self.dim = [128, 128, 64]  
        self.compress_c = 32     
        self.init_layers()
        
        self.eca = ECA()
        self.multi_scale_conv = multi_scale_conv(3*self.compress_c, 3*self.compress_c)
        self.esa = ESA()
        
        self.eca_0 = ECA()
        self.eca_1 = ECA()
        self.eca_2 = ECA()

        self.multi_scale_conv_0 = multi_scale_conv(self.compress_c, self.compress_c)
        self.multi_scale_conv_1 = multi_scale_conv(self.compress_c, self.compress_c)
        self.multi_scale_conv_2 = multi_scale_conv(self.compress_c, self.compress_c)

        self.esa_0 = ESA()
        self.esa_1 = ESA()
        self.esa_2 = ESA()
        
        self.weight_levels = nn.Conv2d(self.compress_c*3, 3, kernel_size=1, stride=1, padding=0)
        
    def init_layers(self):
        if self.level == 0:
            self.stride_level_0 = add_conv(128, self.compress_c, 3, 1)
            self.stride_level_1 = add_conv(128, self.compress_c, 3, 2)
            self.stride_level_2 = add_conv(64, self.compress_c, 3, 2)
            self.expand = add_conv(3*self.compress_c, 128, 3, 1)
        elif self.level == 1:
            self.compress_level_0 = add_conv(128, self.compress_c, 1, 1)
            self.stride_level_1 = add_conv(128, self.compress_c, 3, 1)
            self.stride_level_2 = add_conv(64, self.compress_c, 3, 2)
            self.expand = add_conv(3*self.compress_c, 128, 3, 1)
        elif self.level == 2:
            self.compress_level_0 = add_conv(128, self.compress_c, 1, 1)
            self.compress_level_1 = add_conv(128, self.compress_c, 1, 1)
            self.stride_level_2 = add_conv(64, self.compress_c, 3, 1)
            self.expand = add_conv(3*self.compress_c, 64, 3, 1)

    def resize_features(self, x_level_0, x_level_1, x_level_2):
        if self.level == 0:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = self.stride_level_2(x_level_2)
        return level_0_resized, level_1_resized, level_2_resized

    
    def forward(self, x_level_0, x_level_1, x_level_2):
        resized_features = self.resize_features(x_level_0, x_level_1, x_level_2)
        level_0_resized, level_1_resized, level_2_resized = resized_features

        levels_features = torch.cat((level_0_resized, level_1_resized, level_2_resized), 1)
        
        levels_features_erca = levels_features + self.eca(levels_features)
        levels_features_multi_scale = [conv(levels_features_erca) for conv in self.multi_scale_conv]
        levels_features_multi_scale = torch.sum(torch.stack(levels_features_multi_scale), dim=0)
        levels_features_ersa = levels_features_multi_scale + self.esa(levels_features_multi_scale)
        
        out = self.expand(levels_features_ersa)
        return out
```
**Additional implementation details, along with the model inference code and trained weights, will be released upon acceptance of the paper.**
