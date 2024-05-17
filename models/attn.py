import torch  
import torch.nn as nn  
import torch.nn.functional as F  
  
class ChannelAttention(nn.Module):  
    def __init__(self, in_planes, ratio=4):  
        super(ChannelAttention, self).__init__()  
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  
        self.max_pool = nn.AdaptiveMaxPool1d(1) 
       
  
        self.fc1 = nn.ModuleList([ nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False) for i in range(2) ] )  
        self.relu1 = nn.ReLU()  
        self.fc2 = nn.ModuleList([ nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False) for i in range(2) ] ) 
  
        self.sigmoid = nn.Sigmoid()  
  
    def forward(self, x):  
        avg_out = self.fc2[0](self.relu1(self.fc1[0](self.avg_pool(x))))  
        max_out = self.fc2[1](self.relu1(self.fc1[1](self.max_pool(x))))  
        # std_out = self.fc2[2](self.relu1(self.fc1[2](torch.std(x, dim=-1, keepdim=True))))
        out = avg_out + max_out# + std_out  
        return self.sigmoid(out)  
  
  
class SpatialAttention(nn.Module):  
    def __init__(self, kernel_size=3):  
        super(SpatialAttention, self).__init__()  
  
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'  
        padding = 3 if kernel_size == 7 else 1  
  
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)  
        self.sigmoid = nn.Sigmoid()  
  
    def forward(self, x):  
        avg_out = torch.mean(x, dim=1, keepdim=True)  
        max_out, _ = torch.max(x, dim=1, keepdim=True)  
        # std_out = torch.std(x, dim=1, keepdim=True)  
        # print(std_out.shape, avg_out.shape, max_out.shape)
        x = torch.cat([avg_out, max_out], dim=1)  
        x = self.conv1(x)  
        return self.sigmoid(x)


# class STE(nn.Module):
#     def __init__(self, in_channel = 1, C = 64):
#         super().__init__()
#         self.conv = nn.Conv1d(in_channel,in_channel,1)
#         self.sigmoid = nn.Sigmoid()
#         self.std_linear = nn.Conv1d(1,1,1)
#         self.drop = nn.Dropout(0.3)
#         self.softmax = nn.Softmax()
        
#     def forward(self, x):
        
#         mean_at = self.drop(self.conv(x)) # 16 1024 10

#         mean_at = mean_at.mean(2).unsqueeze(-1)
      
#         std_at = x.permute(0,2,1).std(-1).unsqueeze(1)
#         std_at = self.drop(self.std_linear( std_at  ))
        
#         std_at = self.softmax(std_at)
#         mean_at = self.softmax(mean_at)
        
#         x = x + x * mean_at + x * std_at
#         return x

class STE(nn.Module):
    def __init__(self, in_planes = 1024, kernel_size=3):  
        super(STE, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention(kernel_size)
        self.softmax = nn.Softmax()
        self.drop = nn.Dropout(0.3)
    def forward(self, x):
        q = self.sa(x)
        v = self.ca(x)
        k = x# .clone().detach()
        x = k * self.softmax( q * v )
           # x * self.ca(x) #
        # x =  qvk
        return x

        