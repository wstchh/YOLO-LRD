# YOLO-LRD
This is the official code implementation of YOLO-LRD for the *Measurement* journal paper: **" YOLO-LRD: Revisiting Multi-Scale and Attention Fusion for Lightweight Road Damage Detection"**




## 1-CSFP module


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
```
