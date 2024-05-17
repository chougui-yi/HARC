import torch
import torch.nn as nn
from .HRE.transformer import Transformer
from .attn import ChannelAttention, SpatialAttention
# from .template import Template

class HRE(nn.Module):
    def __init__(self, 
        in_dim, hidden_dim, 
        n_head, n_encoder, n_decoder, n_query, 
        dropout, class_number = 6,
        only_token = "ALL"
    ):
        super().__init__()
        pass
    def forward(self, x):
        pass

class HREC(nn.Module):
    def __init__(self, 
            in_dim, hidden_dim, 
            n_head, n_encoder, n_decoder, n_query, dropout, 
            class_number = 6, only_token = "ALL"):
        super().__init__()
        print( hidden_dim, n_head)
        self.in_proj = nn.Sequential(
                nn.Conv1d(kernel_size=1, in_channels=32, out_channels=hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Conv1d(kernel_size=1, in_channels=hidden_dim // 2, out_channels=512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
            ) 
        # print( hidden_dim, n_head, n_encoder, n_decoder, n_query)
        
        self.in_proj_n = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim)
            ) for i in range(4)
        ])
        
        self.linear = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,6),
            nn.ReLU(),
        )
        
        hidden_dim = 960
        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers = n_encoder,
            num_decoder_layers = n_decoder,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout
        )
        n_query = n_query 

        self.prototype = nn.Embedding(n_query, hidden_dim)  
        
        self.repeat = 4
        self.dnn= nn.Sequential(
            
            nn.Linear( hidden_dim , hidden_dim // 2),
            # nn.BatchNorm1d(6),
            # nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, 256,1),
            # nn.BatchNorm1d(6),
            # nn.ReLU(),
            
            nn.Linear(256, 128,1),
            # nn.BatchNorm1d(6),
            # nn.ReLU(),
            
        )
        
        self.flatten = nn.Flatten()
        self.regressor = nn.Sequential(
            nn.Linear( 6*960 , 512), # 960
            # nn.Linear( 1024 , 512), # 960
            nn.Linear(512, 256),
            nn.Linear(256, 64),
            nn.Linear(64, class_number),
        )
        self.c = nn.AvgPool1d(2)
        self.hidden_dim = 2048
        # self.ca = ChannelAttention(6, ratio=6)
        # self.sa = SpatialAttention(3)
        # self.act = torch.nn.ReLU()
        # self.act = nn.LayerNorm( hidden_dim, eps=1e-5)

        
    def former(self, x, transformer, prototype, prompt = None):
        b,c,t = x.shape
        encode_x = transformer.encoder(x)
        if prompt is not None:
            encode_x += prompt
        q = prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        decode_x = transformer.decoder(q, encode_x)
        return decode_x
        
    def forward(self, x):
        x_arg, x_code, x_ast, x_node_type, x_deep, x_node_total, x_width, x_entropy = x
        x_arg =  self.in_proj(x_arg.permute(0,2,1)).permute(0,2,1)
        x_feature = torch.cat( [ x_arg, x_code, x_ast, x_node_type], 1) # 2 , 4 , 512
        
        x_deep = self.in_proj_n[0](x_deep).unsqueeze(1)
        x_node_total = self.in_proj_n[1](x_node_total).unsqueeze(1)
        x_width = self.in_proj_n[2](x_width).unsqueeze(1)
        x_entropy = self.in_proj_n[3](x_entropy).unsqueeze(1) # (1, 64)
        x_feature_code_2 = torch.cat( [ x_deep, x_node_total, x_width, x_entropy], 1)
       
        x_feature_1 = self.c(x_feature)
        x_feature_2 = self.c(x_feature_1)
        x_feature_3 = self.c(x_feature_2) # B  4 512

        inp = torch.cat([x_feature_3, x_feature_2, x_feature_1, x_feature],2)
      


        q = self.former( inp, self.transformer, self.prototype)
        # print(q.shape) # 256 6 960 # 240 480 720 
        # q = self.act(q + q * self.sa(q) + q * self.ca(q))
        b,c,n = q.shape
        
        # print(q[:,: ,:n//4].shape, q[:,: ,:n//2].shape, q[:,: ,:n//4 * 3].shape, self.dnn_75)
        # print(q.shape)
        # q = self.dnn(q)
        # print(q.shape)
        
        # print(q.shape)
        
        x = self.flatten(q) 
        y = self.regressor(x)

        return [y, self.cal(y)], q

    def cal(self, x):
        x = x.unsqueeze(-1)
        x1 = x.permute(0,2,1)
        self_coef = torch.bmm(x, x1)
        return self_coef
    

class HREC__(nn.Module):
    def __init__(self, 
            in_dim, hidden_dim, 
            n_head, n_encoder, n_decoder, n_query, dropout, 
            class_number = 6):
        super().__init__()
    

        self.in_proj_head_32 = nn.Sequential(
                nn.Conv1d(kernel_size=1, in_channels=32, out_channels=512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
            ) 
        
        self.in_proj_head_512 = nn.Sequential(
                nn.Conv1d(kernel_size=1, in_channels=512, out_channels=512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
            ) 
        
   
        self.linear = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,6),
            nn.ReLU(),
        )
        
        hidden_dim = 960
        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers = n_encoder,
            num_decoder_layers = n_decoder,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout
        )
        n_query = n_query 

        self.prototype = nn.Embedding(n_query, hidden_dim)  
        
        self.repeat = 4
        self.dnn= nn.Sequential(
            
            nn.Linear( hidden_dim , hidden_dim // 2),
            nn.BatchNorm1d(6),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, 256,1),
            nn.BatchNorm1d(6),
            nn.ReLU(),
            
            nn.Linear(256, 128,1),
            nn.BatchNorm1d(6),
            nn.ReLU(),
            
        )
        self.flatten = nn.Flatten()
        self.regressor = nn.Sequential(
            nn.Linear( 768, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 64),
            nn.Linear(64, class_number),
        )
        self.c = nn.AvgPool1d(2)


        
    def former(self, x, transformer, prototype, prompt = None):
        b,c,t = x.shape
        encode_x = transformer.encoder(x)
        if prompt is not None:
            encode_x += prompt
        q = prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        decode_x = transformer.decoder(q, encode_x)
        return decode_x
        
    def forward(self, x):
        # print(len(x))
        x = x[0] # [x] 解包
        b,c,t = x.shape
     
        if t == 32:
            x_feature =  self.in_proj_head_32(x.permute(0,2,1)).permute(0,2,1)
        else:
            x_feature =  self.in_proj_head_512(x.permute(0,2,1)).permute(0,2,1)
    
        
        x_feature_1  = self.c(x_feature)
        x_feature_2 = self.c(x_feature_1)
        x_feature_3 = self.c(x_feature_2)
        inp = torch.cat([x_feature_3, x_feature_2, x_feature_1, x_feature],2)
    
       
        q = self.former( inp, self.transformer, self.prototype)
     
        q = self.dnn(q)
        
     
        x = self.flatten(q) 
        
        y = self.regressor(x)

        return [y, self.cal(y)] # y, q


    def cal(self, x):
        x = x.unsqueeze(-1)
        x1 = x.permute(0,2,1)
        self_coef = torch.bmm(x, x1)
        return self_coef


class HREC_(nn.Module):
    def __init__(self, 
            in_dim, hidden_dim, 
            n_head, n_encoder, n_decoder, n_query, dropout, 
            class_number = 6, only_token = "ALL"):
        super().__init__()
        print( hidden_dim)
        self.in_proj_head_32 = nn.Sequential(
                nn.Conv1d(kernel_size=1, in_channels=32, out_channels=512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
            ) 
        
        self.in_proj_head_512 = nn.Sequential(
                nn.Conv1d(kernel_size=1, in_channels=512, out_channels=512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
            ) 
            
        self.linear = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,6),
            nn.ReLU(),
        )
        
        
        self.in_proj_n = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim)
            ) for i in range(4)
        ])
        
        self.linear = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,6),
            nn.ReLU(),
        )
        
        hidden_dim = 960
        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers = n_encoder,
            num_decoder_layers = n_decoder,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout
        )
        n_query = n_query 

        self.prototype = nn.Embedding(n_query, hidden_dim)  
        
        self.repeat = 4
        self.dnn= nn.Sequential(
            
            nn.Linear( hidden_dim , hidden_dim // 2),
            # nn.BatchNorm1d(6),
            # nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, 256,1),
            # nn.BatchNorm1d(6),
            # nn.ReLU(),
            
            nn.Linear(256, 128,1),
            # nn.BatchNorm1d(6),
            # nn.ReLU(),
            
        )
        
        self.flatten = nn.Flatten()
        self.regressor = nn.Sequential(
            nn.Linear( 6*960 , 512), # 960
            # nn.Linear( 1024 , 512), # 960
            nn.Linear(512, 256),
            nn.Linear(256, 64),
            nn.Linear(64, class_number),
        )
        self.c = nn.AvgPool1d(2)
        self.hidden_dim = 2048
        self.ca = ChannelAttention(6, ratio=6)
        self.sa = SpatialAttention(3)
        self.act = torch.nn.ReLU()
        # self.act = nn.LayerNorm( hidden_dim, eps=1e-5)

        
    def former(self, x, transformer, prototype, prompt = None):
        b,c,t = x.shape
        encode_x = transformer.encoder(x)
        if prompt is not None:
            encode_x += prompt
        q = prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        decode_x = transformer.decoder(q, encode_x)
        return decode_x
        
    def forward(self, x):
        x = x[0]
        b,c,t = x.shape
        if t == 32:
            x_feature =  self.in_proj_head_32(x.permute(0,2,1)).permute(0,2,1)
        else:
            x_feature =  self.in_proj_head_512(x.permute(0,2,1)).permute(0,2,1)
        # print( x_feature.shape )
        x_feature_1 = self.c(x_feature)
        x_feature_2 = self.c(x_feature_1)
        x_feature_3 = self.c(x_feature_2) # B  4 512

        inp = torch.cat([x_feature_3, x_feature_2, x_feature_1, x_feature],2)
      


        q = self.former( inp, self.transformer, self.prototype)
        # print(q.shape) # 256 6 960 # 240 480 720 
        # q = self.act(q + q * self.sa(q) + q * self.ca(q))
        b,c,n = q.shape
        
        # print(q[:,: ,:n//4].shape, q[:,: ,:n//2].shape, q[:,: ,:n//4 * 3].shape, self.dnn_75)
        # print(q.shape)
        # q = self.dnn(q)
        # print(q.shape)
        
        # print(q.shape)
        
        x = self.flatten(q) 
        y = self.regressor(x)

        return [y, self.cal(y)], q

    def cal(self, x):
        x = x.unsqueeze(-1)
        x1 = x.permute(0,2,1)
        self_coef = torch.bmm(x, x1)
        return self_coef
    