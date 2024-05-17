import torch
import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, in_dim, class_number = 6):
        super(DNN, self).__init__()
        hidden_dim = 64
        self.in_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(kernel_size=1, in_channels=in_dim, out_channels=hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Conv1d(kernel_size=1, in_channels=hidden_dim // 2, out_channels=hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            ) for i in range(4)
        ])
        
        self.in_proj_n = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim)
            ) for i in range(4)
        ])
        
        self.dnn= nn.Sequential(
            nn.Conv1d(1572, 512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Conv1d(512, 256,1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(256, 64,1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
      
        self.flatten = nn.Flatten()
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 64 , 512),
            nn.Linear(512, 256),
            nn.Linear(256, 64),
            nn.Linear(64, class_number),
        )
        self.hidden_dim = 2048

    def forward(self, x):
        x_arg, x_code, x_ast, x_node_type, x_deep, x_node_total, x_width, x_entropy = x
        x_arg =  self.in_proj[0](x_arg).permute(0,2,1)
        x_code =  self.in_proj[1](x_code).permute(0,2,1)
        x_ast =  self.in_proj[2](x_ast).permute(0,2,1)
        x_node_type =  self.in_proj[3](x_node_type).permute(0,2,1)
        x_deep = self.in_proj_n[0](x_deep).unsqueeze(1)
        x_node_total = self.in_proj_n[1](x_node_total).unsqueeze(1)
        x_width = self.in_proj_n[2](x_width).unsqueeze(1)
        x_entropy = self.in_proj_n[3](x_entropy).unsqueeze(1) # (1, 64)
    
        x_feature = torch.cat( [ x_arg, x_code, x_ast, x_node_type, x_deep, x_node_total, x_width, x_entropy], 1)
        #print(x_feature.shape)
        #b,t,c = x_feature.shape
        #t = self.hidden_dim//t + 1
        inp = x_feature#.repeat(1,t,1)
        #inp = inp[:,:self.hidden_dim,:]
       
        q = self.dnn(inp)
        x = self.flatten(q)
        y = self.regressor(x)
        
        return y, q


class DNN_(nn.Module):
    def __init__(self, in_dim, class_number = 6):
        super().__init__()
        hidden_dim = 64
        self.in_proj_head_32 = nn.Sequential(
                nn.Conv1d(kernel_size=1, in_channels=32, out_channels=64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            ) 
        
        self.in_proj_head_512 = nn.Sequential(
                nn.Conv1d(kernel_size=1, in_channels=512, out_channels=64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            ) 
        self.in_proj_n = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim)
            ) for i in range(4)
        ])
        
        self.dnn= nn.Sequential(
            nn.Conv1d(64, 512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Conv1d(512, 256,1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(256, 64,1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
      
        self.flatten = nn.Flatten()
        self.regressor = nn.Sequential(
            nn.Linear( 64 , 512),
            nn.Linear(512, 256),
            nn.Linear(256, 64),
            nn.Linear(64, class_number),
        )
        self.hidden_dim = 1024

    def forward(self, x):
        x = x[0]
        b,c,t = x.shape
     
        if t == 32:
            x_feature =  self.in_proj_head_32(x.permute(0,2,1)).permute(0,2,1)
        else:
            x_feature =  self.in_proj_head_512(x.permute(0,2,1)).permute(0,2,1)
    
        # b,t,c = x_feature.shape
        # # print(x_feature.shape)
        # t = self.hidden_dim//t + 1
        # inp = x_feature.repeat(1,t,1)
        # inp = inp[:,:self.hidden_dim,:]
        inp = x_feature.permute( 0, 2, 1)
        # print(inp.shape)
        q = self.dnn(inp)
        x = self.flatten(q)
        # print(x.shape)
        y = self.regressor(x)
        
        return y, q




