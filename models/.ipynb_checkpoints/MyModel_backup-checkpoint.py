import torch
import torch.nn as nn
from .HRE.transformer import Transformer


# from HRE.transformer import Transformer
class HRE(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_head, n_encoder, n_decoder, n_query, dropout, class_number=6,
                 activate_regular_restrictions=None):
        super(HRE, self).__init__()

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

        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=n_encoder,
            num_decoder_layers=n_decoder,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout
        )

        self.prototype = nn.Embedding(n_query, hidden_dim)

        self.flatten = nn.Flatten()
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * n_query, 64),
            nn.BatchNorm1d(64),
            nn.Linear(64, class_number),
            nn.Sigmoid(),
        )

    def former(self, x, transformer, prototype, prompt=None):
        b, c, t = x.shape
        encode_x = transformer.encoder(x)
        if prompt is not None:
            encode_x += prompt
        q = prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        decode_x = transformer.decoder(q, encode_x)
        return decode_x

    def forward(self, x_arg, x_code, x_ast, x_node_type, x_deep, x_node_total, x_width, x_entropy):
        #         [
        #             ( 1, 1, 10,),
        #              torch.Size([ 1, 1, 150]),
        #              torch.Size([,1 1, 150]),
        #              (1, 1,86,),
        #              torch.Size([1, 1]),
        #              torch.Size([1, 1]),
        #              torch.Size([1, 1]),
        #              torch.Size([1, 1]),
        #         ]

        x_arg = self.in_proj[0](x_arg).permute(0, 2, 1)
        x_code = self.in_proj[1](x_code).permute(0, 2, 1)
        x_ast = self.in_proj[2](x_ast).permute(0, 2, 1)
        x_node_type = self.in_proj[3](x_node_type).permute(0, 2, 1)
        # print( x_arg.shape, x_code.shape, x_ast.shape, x_node_type.shape  )
        x_deep = self.in_proj_n[0](x_deep).unsqueeze(1)
        x_node_total = self.in_proj_n[1](x_node_total).unsqueeze(1)
        x_width = self.in_proj_n[2](x_width).unsqueeze(1)
        x_entropy = self.in_proj_n[3](x_entropy).unsqueeze(1)  # (1, 64)

        x_feature = torch.cat([x_arg, x_code, x_ast, x_node_type, x_deep, x_node_total, x_width, x_entropy], 1)

        q = self.former(x_feature, self.transformer, self.prototype)

        x = self.flatten(q)

        y = self.regressor(x)
        return y, q


class RES(nn.Module):
    def __init__(self, in_dim):
        super(RES, self).__init__()
        hiddem_dim = in_dim // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, hiddem_dim, 1, ),
            nn.BatchNorm1d(hiddem_dim),
            nn.ReLU(),
            nn.Conv1d(hiddem_dim, hiddem_dim, 1),
            nn.BatchNorm1d(hiddem_dim),
            nn.ReLU(),
            nn.Conv1d(hiddem_dim, in_dim, 1),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x + self.conv(x)
        return x



class HREC(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_head, n_encoder, n_decoder, n_query, dropout, class_number = 6, activate_regular_restrictions = None):
        super(HREC, self).__init__()
        
        self.in_proj = nn.ModuleList([
            nn.Sequential(
                #                 nn.Conv1d(kernel_size=1, in_channels=in_dim, out_channels=hidden_dim // 2),
                #                 nn.BatchNorm1d(hidden_dim // 2),
                #                 # nn.BatchNorm1d(hidden_dim // 2),
                #                 nn.ReLU(),
                #                 nn.Conv1d(kernel_size=1, in_channels=hidden_dim // 2, out_channels=hidden_dim),
                #                 # nn.BatchNorm1d(hidden_dim)
                nn.Conv1d(kernel_size=1, in_channels=512, out_channels=hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Conv1d(kernel_size=1, in_channels=hidden_dim // 2, out_channels=hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            ) for i in range(4)
        ])
        
        self.in_proj[0] = nn.Sequential(
                nn.Conv1d(kernel_size=1, in_channels=32, out_channels=hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Conv1d(kernel_size=1, in_channels=hidden_dim // 2, out_channels=512),
                nn.BatchNorm1d(512),
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
        # print(n_query, hidden_dim)
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
        self.hidden_dim = 2048

        
    def former(self, x, transformer, prototype, prompt = None):
        b,c,t = x.shape
        encode_x = transformer.encoder(x)
        if prompt is not None:
            encode_x += prompt
        q = prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        decode_x = transformer.decoder(q, encode_x)
        return decode_x
        
    def forward(self, x_arg, x_code, x_ast, x_node_type, x_deep, x_node_total, x_width, x_entropy):
        #         [
        #             ( 1, 1, 10,),
        #              torch.Size([ 1, 1, 150]),
        #              torch.Size([,1 1, 150]),
        #              (1, 1,86,),
        #              torch.Size([1, 1]),
        #              torch.Size([1, 1]),
        #              torch.Size([1, 1]),
        #              torch.Size([1, 1]),
        #         ]
        # print( x_arg.shape, x_code.shape, x_ast.shape, x_node_type.shape  )
        x_arg =  self.in_proj[0](x_arg.permute(0,2,1)).permute(0,2,1)
       
        # x_code =  self.in_proj[1](x_code).permute(0,2,1)
        # print("code:",x_code.shape)
        # x_ast =  self.in_proj[2](x_ast).permute(0,2,1)
        # print("ast:",x_ast.shape)
        # x_node_type =  self.in_proj[3](x_node_type).permute(0,2,1)
        # print("x_node:",x_node_type.shape)
        # print( x_arg.shape, x_code.shape, x_ast.shape, x_node_type.shape  )
        x_feature = torch.cat( [ x_arg, x_code, x_ast, x_node_type], 1) # 2 , 4 , 512
        # print(x_feature.shape)
        
        x_deep = self.in_proj_n[0](x_deep).unsqueeze(1)
        x_node_total = self.in_proj_n[1](x_node_total).unsqueeze(1)
        x_width = self.in_proj_n[2](x_width).unsqueeze(1)
        x_entropy = self.in_proj_n[3](x_entropy).unsqueeze(1) # (1, 64)
        # print( x_deep.shape, x_node_total.shape, x_width.shape, x_entropy.shape  )
        
        x_feature_code_2 = torch.cat( [ x_deep, x_node_total, x_width, x_entropy], 1)
        # print(  x_feature_code_2.shape )
        
        
        x_feature_1  = self.c(x_feature)
        x_feature_2 = self.c(x_feature_1)
        x_feature_3 = self.c(x_feature_2)
        inp = torch.cat([x_feature_3, x_feature_2, x_feature_1, x_feature],2)
        # print( inp.shape )
        # b,t,c = inp.shape
        # t = self.hidden_dim//t + 1
        # inp = inp.repeat(1,t,1)
        # inp = inp[:,:self.hidden_dim,:]
        
        # print(inp.shape)
        q = self.former( inp, self.transformer, self.prototype)
        #print(q.shape)
    
        q = self.dnn(q)
        
        
        x = self.flatten(q) 
        # x = torch.cat( [ x, self.flatten(x_feature_code_2)], -1)
        y = self.regressor(x)
        return y, q

    def forward_token(self, x):
        x = self.in_proj[0](x).permute(0, 2, 1)
        #         x_code =  self.in_proj[1](x_code).permute(0,2,1)
        #         x_ast =  self.in_proj[2](x_ast).permute(0,2,1)
        #         x_node_type =  self.in_proj[3](x_node_type).permute(0,2,1)
        #         # print( x_arg.shape, x_code.shape, x_ast.shape, x_node_type.shape  )
        #         x_deep = self.in_proj_n[0](x_deep).unsqueeze(1)
        #         x_node_total = self.in_proj_n[1](x_node_total).unsqueeze(1)
        #         x_width = self.in_proj_n[2](x_width).unsqueeze(1)
        #         x_entropy = self.in_proj_n[3](x_entropy).unsqueeze(1) # (1, 64)

        # x_feature = torch.cat( [x_deep, x_node_total, x_width, x_entropy, x_arg, x_node_type, x_ast, x_code ], 1)
        x_feature = x
        b, t, c = x_feature.shape
        inp = x_feature.repeat(1, self.repeat, 1)

        #  print(inp.shape)
        q = self.former(inp, self.transformer, self.prototype)
        # print(q.shape)
        x = self.flatten(q)

        y = self.regressor(x)
        return y, q
