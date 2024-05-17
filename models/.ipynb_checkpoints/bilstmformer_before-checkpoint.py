
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .HRE.transformer import Transformer
from .bigrua import Attention

class BiLSTM(nn.Module):

    def __init__(self,  vocab_size, embedding_dim, n_hidden, num_classes = 6 ):
        super().__init__()
       
        self.LSTM = nn.LSTM(embedding_dim, n_hidden,
                            num_layers=1, batch_first=True,
                            bidirectional=True)
        # 因为是双向 LSTM, 所以要乘2
        self.ffn = nn.Linear(n_hidden * 2, n_hidden)
        self.relu = nn.ReLU()

    
    def forward(self, inputs):
  
        lstm_hidden_states, context = self.LSTM(inputs)
        # lstm_hidden_states = lstm_hidden_states[:, -1, :]
        ffn_outputs = self.relu(self.ffn(lstm_hidden_states))

        return ffn_outputs, context


    
class CBiLSTMAF(nn.Module):
    def __init__(self, in_dim, class_number = 6):
        super(CBiLSTMAF, self).__init__()
        hidden_dim = 64
        n_head = 4
        n_encoder = 1
        n_decoder = 2 
        dropout = 0.3
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
        
        self.bilstm = nn.Sequential(
            BiLSTM( hidden_dim, hidden_dim, hidden_dim),
        )
        self.attn = Attention(hidden_dim)
        
        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers = n_encoder,
            num_decoder_layers = n_decoder,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout
        )
        self.flatten = nn.Flatten()
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.Sigmoid(),
            nn.Linear(32, class_number),
            nn.Sigmoid(),
        )
        
    def former(self, x, transformer, prototype, prompt = None):
        # print(x.shape, prototype.shape)
        b,c,t = x.shape
        encode_x = transformer.encoder(x)
        if prompt is not None:
            encode_x += prompt
        q = prototype.repeat(1, c, t)
        # print(q.shape, encode_x.shape)
        decode_x = transformer.decoder(q, encode_x)
        return decode_x
        
    def forward(self, x_arg, x_code, x_ast, x_node_type, x_deep, x_node_total, x_width, x_entropy):
        x_arg =  self.in_proj[0](x_arg).permute(0,2,1)
        x_code =  self.in_proj[1](x_code).permute(0,2,1)
        x_ast =  self.in_proj[2](x_ast).permute(0,2,1)
        x_node_type =  self.in_proj[3](x_node_type).permute(0,2,1)
        # print( x_arg.shape, x_code.shape, x_ast.shape, x_node_type.shape  )
        x_deep = self.in_proj_n[0](x_deep).unsqueeze(1)
        x_node_total = self.in_proj_n[1](x_node_total).unsqueeze(1)
        x_width = self.in_proj_n[2](x_width).unsqueeze(1)
        x_entropy = self.in_proj_n[3](x_entropy).unsqueeze(1) # (1, 64)
        
        x_feature = torch.cat( [ x_arg, x_code, x_ast, x_node_type, x_deep, x_node_total, x_width, x_entropy], 1)
        
        # print(x_feature.shape)
        y, context = self.bilstm(x_feature)
        # print(y.shape, context[0][-1,:,:].unsqueeze(1).shape)
        y1, y2 = self.attn( y, context[0][-1,:,:].unsqueeze(1))
        # print(y1.shape, y2.shape)
        y_former = self.former( y1[:,-1,:].unsqueeze(1), self.transformer, y2[:,-1,:].unsqueeze(1))
        q = self.flatten(y_former)
        # print(q.shape)
        y = self.regressor(q)
        return y , q
