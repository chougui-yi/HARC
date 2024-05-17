from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import copy
from models.HRE.transformer import Transformer
import math
# from .ETF.ETFHead import ETFHead

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        try:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
        except AttributeError:
            pass

def generate_random_orthogonal_matrix(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    return orth_vec

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def eth( in_channels, num_classes, normal = False):
    orth_vec = generate_random_orthogonal_matrix(in_channels, num_classes)
    i_nc_nc = torch.eye(num_classes)
    one_nc_nc: torch.Tensor = torch.mul(torch.ones(num_classes, num_classes), (1 / num_classes))
    etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                        math.sqrt(num_classes / (num_classes - 1)))
    etf_rect = torch.ones((1, num_classes), dtype=torch.float32)
    if normal:
        etf_vec = (etf_vec / torch.sum(etf_vec, dim=0, keepdim=True))
    return etf_vec, etf_rect


class GDLT(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_head, n_encoder, n_decoder, n_query, dropout, activate_regular_restrictions = None):
        super(GDLT, self).__init__()
        self.in_proj = nn.Sequential(
            nn.Conv1d(kernel_size=1, in_channels=in_dim, out_channels=in_dim // 2),
            nn.BatchNorm1d(in_dim // 2),
            nn.ReLU(),
            nn.Conv1d(kernel_size=1, in_channels=in_dim // 2, out_channels=hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=n_encoder,
            num_decoder_layers=n_decoder,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout
        )
        self.eval_classes = n_query
        self.prototype = nn.Embedding(n_query, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, n_query)
        # torch.manual_seed(3407)
        self.weight = torch.linspace(0, 1, n_query, requires_grad=False).to(device)
    def forward(self, x, return_feat = False):
        # x (b, t, c)
        b, t, c = x.shape
        x = self.in_proj(x.transpose(1, 2)).transpose(1, 2)

        q = self.prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        encode_x = self.transformer.encoder(x)
        q1 = self.transformer.decoder(q, encode_x)

        s1 = self.regressor(q1)  # (b, n, n)
        s = torch.diagonal(s1, dim1=-2, dim2=-1)  # (b, n)
        norm_s = torch.sigmoid(s)
        norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        out = torch.sum(self.weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)
        if return_feat:
            return {'output': out, 'embed': q1, 'other':{"s":s1.clone().cpu().detach().numpy()}}
        return {'output': out, 'embed': q1, 's':s1}

def choose_activate(type_id, n_query, device):
    if type_id == 0:
        return torch.linspace(0, 1, n_query, requires_grad=False).to(device), torch.linspace(0, 1, n_query, requires_grad=False).to(device).flip(-1)
    elif type_id == 1:
        return torch.linspace(0, 1, n_query+1, requires_grad=False)[1:].to(device), torch.linspace(0, 1, n_query+1, requires_grad=False)[1:].to(device).flip(-1)
    elif type_id == 2:
        # sigmoid
        return torch.tensor([0.1, 0.2, 0.8, 1], requires_grad=False).to(device).to(torch.float32), torch.tensor([1, 1, -1, -1], requires_grad=False).to(device).to(torch.float32)
    else:
        # arcl1
        #return torch.tensor([-1, -0.8, 0.8, 1], requires_grad=False).to(device).to(torch.float32),torch.tensor([-1, -0.8, 0.8, 1], requires_grad=False).to(device).to(torch.float32).flip(-1)
        d = [-1, -0.8, 0.8, 1]
        return torch.tensor(d, requires_grad=False).to(device).to(torch.float32),torch.tensor( d, requires_grad=False).to(device).to(torch.float32).flip(-1)

class GDLTETH1(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_head, n_encoder, n_decoder, n_query, dropout, activate_regular_restrictions):
        super(GDLTETH1, self).__init__()
        self.in_proj = nn.Sequential(
            nn.Conv1d(kernel_size=1, in_channels=in_dim, out_channels=in_dim // 2),
            nn.BatchNorm1d(in_dim // 2),    
            nn.ReLU(),
            nn.Conv1d(kernel_size=1, in_channels=in_dim // 2, out_channels=hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=n_encoder,
            num_decoder_layers=n_decoder,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout
        )
        # old 
        self.eval_classes = n_query
        self.prototype = nn.Embedding(n_query, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, n_query)
        self.regressor_revert = nn.Linear(hidden_dim, n_query)
        we = choose_activate(activate_regular_restrictions, n_query, device)
        self.weight = we[0] # torch.tensor([-1, -0.8, 0.8, 1], requires_grad=False).to(device).to(torch.float32)
        self.weight_revert = we[1]# torch.tensor([1, 1, 0, 0], requires_grad=False).to(device).to(torch.float32)# we[1] # torch.tensor([1, 1, -1, -1], requires_grad=False).to(device).to(torch.float32) # we.clone().cpu().flip(-1)# torch.tensor([1, 0.8, -0.8, -1], requires_grad=False).to(device).to(torch.float32) # 
        # ball 固定 2 ： -1 -0.8， 0.8 ， 1 best
        

        # etf
        self.val = False
        self.score_len = 100        
        etf_vec1, etf_rect = eth( self.score_len, self.score_len) # 1000 score etf_rect[:,4]
        self.register_buffer('etf_vec1', etf_vec1)
        self.etf_rect = etf_rect
 

        self.key_len = 1
        self.regressori = nn.ModuleList()
        self.regi = nn.ModuleList()
        for i in range(self.key_len):
            self.regressori.append(
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(4 * hidden_dim, 256),
                    nn.Linear(hidden_dim, self.score_len),
                )
            )
            self.regi.append(
                torch.nn.MultiheadAttention(embed_dim = hidden_dim, num_heads = 4, dropout = dropout),
            )

    def forward(self, x, return_feat = False):
        # x (b, t, c)
        result = {
            "int": None,
            "int_revert": None,
            "dec": [],
        }
        
        b, t, c = x.shape
        x = self.in_proj(x.transpose(1, 2)).transpose(1, 2)

        q = self.prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        encode_x = self.transformer.encoder(x)   
        q1 = self.transformer.decoder(q, encode_x) # torch.Size([32, 4, 256])

        s = self.regressor(q1) 
        out = self.pre_old(s, self.weight, b)
        result['int'] = out

        s1 = self.regressor_revert(q1) 
        out = self.pre_old(s1, self.weight_revert, b)
        result['int_revert'] = out
        
        # q1 = q1.view(b, -1)
        # s = self.regressor1(q1) # - s.clone().cpu().detach() ( 32, 4 * 256)
        for i in range(self.key_len):
            s_dec, _ = self.regi[i](q1,q1,q1)
            s_dec = self.regressori[i](s_dec)
            norm_d = self.pre_logits(s_dec)
            result['dec'].append(norm_d)

        if return_feat:
            return {'output': result, 'embed': q1, "other":{
                "int":s.clone().cpu().detach().numpy() ,
                "int_revert":s1.clone().cpu().detach().numpy(),
                "dec":s_dec.clone().cpu().detach().numpy()}
            }

        return {'output': result, 'embed': q1}
    
    def pre_old(self, s, weight, b):
        s = torch.diagonal(s, dim1=-2, dim2=-1)  # (b, n) torch.Size([32, 4, 4])
        norm_s = torch.sigmoid(s)
        norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        out = torch.sum(weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)
        return out
    
    def pre_logits(self, x):
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        # x = x / torch.sum(x, dim=1, keepdim=True)
        return x
    
    def re_proj(self, x):
        return torch.argmax(x)

    def re_proj1(self, x, gt_label = None):
        x = torch.bmm( x.unsqueeze(1) , self.ref_poj[gt_label].unsqueeze(-1))
        return x

    def get_proj_class(self, gt_label1):
        gt_label = (gt_label1 * 10000).long()
        
        g_1 = gt_label//100 / 100
        g_2 = gt_label - gt_label//100 * 100
        # print(gt_label1, g_1, g_2)
        target_1 = self.etf_vec1[:, g_2].t()
        target = [ g_1, torch.ones_like(g_1) - g_1, target_1]
        # print(g_1, torch.ones_like(g_1) - g_1, g_2)
        return target

    def get_score(self, x):
        # print(x)
        cls_score1 = x['dec'][0] @ self.etf_vec1 # @ x[0].T
        c_1 = self.re_proj(cls_score1)
        score = (x['int'] + torch.ones_like(x['int_revert']) - x['int_revert'])/2  +  c_1/10000
        # print(x['int'] ,torch.ones_like(x['int_revert']) - x['int_revert'] , c_1, score)
        return score

    def eth_head(self, x = None, gt_label = None):
        target = 0
        if gt_label is None:
            target = self.get_score(x)
        if x is None:
            target = self.get_proj_class(gt_label)
        return target

class GDLTETH2(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_head, n_encoder, n_decoder, n_query, dropout, activate_regular_restrictions):
        super(GDLTETH2, self).__init__()
        self.in_proj = nn.Sequential(
            nn.Conv1d(kernel_size=1, in_channels=in_dim, out_channels=in_dim // 2),
            nn.BatchNorm1d(in_dim // 2),    
            nn.ReLU(),
            nn.Conv1d(kernel_size=1, in_channels=in_dim // 2, out_channels=hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=n_encoder,
            num_decoder_layers=n_decoder,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout
        )
        # old 
        self.eval_classes = n_query
        self.prototype = nn.Embedding(n_query, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, n_query)
        we = choose_activate(activate_regular_restrictions, n_query, device)
        self.weight = we[0] #torch.tensor([-1, -0.8, 0.8, 1], requires_grad=False).to(device).to(torch.float32)
        

        # etf
        self.val = False
        self.score_len = 100        
        etf_vec1, etf_rect = eth( self.score_len, self.score_len) # 1000 score etf_rect[:,4]
        self.register_buffer('etf_vec1', etf_vec1)
        self.etf_rect = etf_rect
 
        self.key_len = 1
        self.regressori = nn.ModuleList()
        self.regi = nn.ModuleList()
        for i in range(self.key_len):
            self.regressori.append(
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(4 * hidden_dim, 256),
                    nn.Linear(hidden_dim, self.score_len),
                )
            )
            self.regi.append(
                torch.nn.MultiheadAttention(embed_dim = hidden_dim, num_heads = 4, dropout = dropout),
            )

    def forward(self, x, return_feat = False):
        # x (b, t, c)
        result = {
            "int": None,
            "dec": [],
        }

        b, t, c = x.shape
        x = self.in_proj(x.transpose(1, 2)).transpose(1, 2)

        q = self.prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        encode_x = self.transformer.encoder(x)   
        q1 = self.transformer.decoder(q, encode_x) # torch.Size([32, 4, 256])

        s = self.regressor(q1) 
        out = self.pre_old(s, self.weight, b)
        result['int'] = out
        
        # q1 = q1.view(b, -1)
        # s = self.regressor1(q1) # - s.clone().cpu().detach() ( 32, 4 * 256)
        for i in range(self.key_len):
            s_dec,_ = self.regi[i](q1,q1,q1)
            s_dec = self.regressori[i](s_dec)
            norm_d = self.pre_logits(s_dec)
            result['dec'].append(norm_d)
        if return_feat:
            return {'output': result, 'embed': q1, "other":{ "int":s.clone().cpu().detach().numpy() , "dec":s_dec.clone().cpu().detach().numpy()}}
        return {'output': result, 'embed': q1}
    
    def pre_old(self, s, weight, b):
        s = torch.diagonal(s, dim1=-2, dim2=-1)  # (b, n) torch.Size([32, 4, 4])
        norm_s = torch.sigmoid(s)
        norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        out = torch.sum(weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)
        return out
    
    def pre_logits(self, x):
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        # x = x / torch.sum(x, dim=1, keepdim=True)
        return x
    
    def re_proj(self, x):
        return torch.argmax(x)

    def re_proj1(self, x, gt_label = None):
        x = torch.bmm( x.unsqueeze(1) , self.ref_poj[gt_label].unsqueeze(-1))
        return x

    def get_proj_class(self, gt_label1):
        gt_label = (gt_label1 * 10000).long()
        
        g_1 = gt_label//100 / 100
        g_2 = gt_label - gt_label//100 * 100
        # print(gt_label1, g_1, g_2)
        target_1 = self.etf_vec1[:, g_2].t()
        target = [ g_1, target_1]
        return target

    def get_score(self, x):
        # print(x)
        cls_score1 = x['dec'][0] @ self.etf_vec1 # @ x[0].T
        c_1 = self.re_proj(cls_score1)
        score = x['int']  +  c_1/10000
        # print(c_1, c_2, score)
        return score

    def eth_head(self, x = None, gt_label = None):
        target = 0
        if gt_label is None:
            target = self.get_score(x)
        if x is None:
            target = self.get_proj_class(gt_label)
        return target

class GDLTETH_ab(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_head, n_encoder, n_decoder, n_query, dropout, activate_regular_restrictions, method = 0):
        super(GDLTETH_ab, self).__init__()
        self.in_proj = nn.Sequential(
            nn.Conv1d(kernel_size=1, in_channels=in_dim, out_channels=in_dim // 2),
            nn.BatchNorm1d(in_dim // 2),    
            nn.ReLU(),
            nn.Conv1d(kernel_size=1, in_channels=in_dim // 2, out_channels=hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=n_encoder,
            num_decoder_layers=n_decoder,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout
        )

        self.ab = method

        self.replace_transformer_encode = nn.Sequential( #  torch.Size([32, 68, 256])
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.replace_transformer_decode = nn.Sequential( #   torch.Size([32, 4, 256])
            nn.Conv1d(68, 68, 1),
            nn.BatchNorm1d(68),    
            nn.ReLU(),
            nn.Conv1d(68, 68, 1),
            nn.BatchNorm1d(68),    
            nn.ReLU(),
            nn.Conv1d(68, n_query, 1),
            nn.BatchNorm1d(n_query),    
            nn.ReLU(),
        )

        # old 
        self.eval_classes = n_query
        self.prototype = nn.Embedding(n_query, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, n_query)
        we = choose_activate(activate_regular_restrictions, n_query, device)
        self.weight = we[0] #torch.tensor([-1, -0.8, 0.8, 1], requires_grad=False).to(device).to(torch.float32)
        

        # etf
        self.val = False
        self.score_len = 100        
        etf_vec1, etf_rect = eth( self.score_len, self.score_len) # 1000 score etf_rect[:,4]
        self.register_buffer('etf_vec1', etf_vec1)
        self.etf_rect = etf_rect
 
        self.key_len = 1
        self.regressori = nn.ModuleList()
        self.regi = nn.ModuleList()
        for i in range(self.key_len):
            self.regressori.append(
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(4 * hidden_dim, 256),
                    nn.Linear(hidden_dim, self.score_len),
                )
            )
            if self.ab == 3:
                print("ab is 3", self.ab)
                self.regi.append(
                        nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                )
            else:
                self.regi.append(
                    torch.nn.MultiheadAttention(embed_dim = hidden_dim, num_heads = 4, dropout = dropout),
                )

    def forward(self, x, return_feat = False):
        # x (b, t, c)
        result = {
            "int": None,
            "dec": [],
        }
        b, t, c = x.shape
        x = self.in_proj(x.transpose(1, 2)).transpose(1, 2) # x:  torch.Size([32, 68, 256])

        q = self.prototype.weight.unsqueeze(0).repeat(b, 1, 1) 

        if self.ab == 1 :
            encode_x = self.replace_transformer_encode(x)
        else:
            encode_x = self.transformer.encoder(x)     # encode_x : torch.Size([32, 68, 256])

        if self.ab == 2 :
            q1 = self.replace_transformer_decode(x)     # encode_x : torch.Size([32, 68, 256])
        else:
            q1 = self.transformer.decoder(q, encode_x) # torch.Size([32, 4, 256])

        s = self.regressor(q1) 
        out = self.pre_old(s, self.weight, b)
        result['int'] = out
        
        # q1 = q1.view(b, -1)
        # s = self.regressor1(q1) # - s.clone().detach() ( 32, 4 * 256)
        for i in range(self.key_len):
            if self.ab ==3:
                s_dec = self.regi[i](q1)
            else:
                s_dec,_ = self.regi[i](q1,q1,q1)
            s_dec = self.regressori[i](s_dec)
            norm_d = self.pre_logits(s_dec)
            result['dec'].append(norm_d)
        if return_feat:
            return {'output': result, 'embed': q1, "other":{ "int":s , "dec":s_dec}}
        return {'output': result, 'embed': q1}
    
    def pre_old(self, s, weight, b):
        s = torch.diagonal(s, dim1=-2, dim2=-1)  # (b, n) torch.Size([32, 4, 4])
        norm_s = torch.sigmoid(s)
        norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        out = torch.sum(weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)
        return out
    
    def pre_logits(self, x):
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        # x = x / torch.sum(x, dim=1, keepdim=True)
        return x
    
    def re_proj(self, x):
        return torch.argmax(x)

    def re_proj1(self, x, gt_label = None):
        x = torch.bmm( x.unsqueeze(1) , self.ref_poj[gt_label].unsqueeze(-1))
        return x

    def get_proj_class(self, gt_label1):
        gt_label = (gt_label1 * 10000).long()
        
        g_1 = gt_label//100 / 100
        g_2 = gt_label - gt_label//100 * 100
        # print(gt_label1, g_1, g_2)
        target_1 = self.etf_vec1[:, g_2].t()
        target = [ g_1, target_1]
        return target

    def get_score(self, x):
        # print(x)
        cls_score1 = x['dec'][0] @ self.etf_vec1 # @ x[0].T
        c_1 = self.re_proj(cls_score1)
        score = x['int']  +  c_1/10000
        # print(c_1, c_2, score)
        return score

    def eth_head(self, x = None, gt_label = None):
        target = 0
        if gt_label is None:
            target = self.get_score(x)
        if x is None:
            target = self.get_proj_class(gt_label)
        return target

class GDLTETH_1000(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_head, n_encoder, n_decoder, n_query, dropout):
        super(GDLTETH, self).__init__()
        print(in_dim, hidden_dim, n_head, n_encoder, n_decoder, n_query, dropout)
        self.in_proj = nn.Sequential(
            nn.Conv1d(kernel_size=1, in_channels=in_dim, out_channels=in_dim // 2),
            nn.BatchNorm1d(in_dim // 2),
            nn.ReLU(),
            nn.Conv1d(kernel_size=1, in_channels=in_dim // 2, out_channels=hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=n_encoder,
            num_decoder_layers=n_decoder,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout
        )
        # self.eval_classes = n_query
        self.prototype = nn.Embedding(n_query, hidden_dim)
        self.weight = torch.linspace(0, 1, n_query, requires_grad=False).to(device)
        self.weight_etf = torch.linspace(0, 1, 64 * n_query, requires_grad=False).to(device)
        # 
        
        self.score_len = 100
        d1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * hidden_dim, 256),
            nn.Linear(256, 128),
            nn.Linear(128, self.score_len),
        )
       
        self.key_len = 2
        self.regressor = nn.ModuleList(
            [ d1 for i in range(self.key_len)]
        )
        d = nn.Sequential(

            nn.Linear(n_query, 1, bias=False),
        )
        self.regressori = nn.ModuleList(
            [ d for i in range(self.key_len) ]
        )
        self.eval_classes = n_query
        
        etf_vec1, etf_rect = eth(self.score_len, self.score_len) # 1000 score etf_rect[:,4]
        # etf_vec1 = torch.eye(100)
        self.register_buffer('etf_vec1', etf_vec1) # 整数
        etf_vec2, etf_rect = eth( self.score_len, self.score_len) 
        # etf_vec2 = torch.eye(100)
        self.register_buffer('etf_vec2', etf_vec2) # 小数
        self.etf_rect = etf_rect
        self.val = False

    def forward(self, x):
        # x (b, t, c)
        b, t, c = x.shape
        #print("in:", x.shape) # in: torch.Size([32, 68, 1024])
        x = self.in_proj(x.transpose(1, 2)).transpose(1, 2)  
        # print("proj:", x.shape) # proj: torch.Size([32, 68, 256])
        q = self.prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        encode_x = self.transformer.encoder(x)
        # print("encoder:", encode_x.shape) encoder: torch.Size([32, 68, 256])
        q1 = self.transformer.decoder(q, encode_x)  
        #print("q1:", q1.shape) # q1: torch.Size([32, 4, 256])
        normd_list = []
        for i in range(self.key_len):
            s = self.regressor[i](q1)  # (b, n, n)  32, 1000 , 1000 
            # print("s:", s.shape)
            # s = torch.diagonal(s, dim1=-2, dim2=-1)  # (b, n) # 32 4
            # print("s1: ", s.shape)
            s = self.pre_logits(s)
            norm_d = self.pre_old( s, b)
            # print(norm_d.shape)
            normd_list.append(norm_d)
        return {'output': normd_list, 'embed': q1}

    def pre_old(self, norm_s, b):
        norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        # out = torch.sum(self.weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)
        return norm_s
    
    def pre_logits(self, x):
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x
    
    def re_proj(self, x):
        return torch.argmax(x,1)

    def re_proj1(self, x, gt_label = None):
        x = torch.bmm( x.unsqueeze(1) , self.ref_poj[gt_label].unsqueeze(-1))
        return x

    def get_proj_class(self, gt_label):
        gt_label = (gt_label * 10000).long()
        # print('gt_label:', gt_label, self.etf_vec.shape)
        g_1 = gt_label//100
        g_2 = gt_label - gt_label//100 * 100

        target_1 = self.etf_vec1[:, g_1].t()
        target_2 = self.etf_vec2[:, g_2].t()
       
        target = [target_1, target_2]
        # assert False, 'break here'
        return target

    def get_score(self, x):
        # print(x[0].shape, x[1].shape, self.etf_vec1.shape)
        cls_score1 =   x[0] @ self.etf_vec1 
        cls_score2 =   x[1] @ self.etf_vec2 
        # print(x[0].shape, x[1].shape, cls_score1.shape, cls_score2.shape)
        c_1 = self.re_proj(cls_score1)
        c_2 = self.re_proj(cls_score2)
        # print(c_1.shape, c_2.shape)
        score = (c_1 *100 + c_2)/100
        # print(score)
        # print(c_1, c_2, score)
        return score

    def eth_head(self, x = None, gt_label = None):
        target = 0
        if gt_label is None:
            target = self.get_score(x)
        if x is None:
            target = self.get_proj_class(gt_label)
        return target

class GDLTETH_10(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_head, n_encoder, n_decoder, n_query, dropout):
        super(GDLTETH, self).__init__()
        self.in_proj = nn.Sequential(
            nn.Conv1d(kernel_size=1, in_channels=in_dim, out_channels=in_dim // 2),
            nn.BatchNorm1d(in_dim // 2),
            nn.ReLU(),
            nn.Conv1d(kernel_size=1, in_channels=in_dim // 2, out_channels=hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=n_encoder,
            num_decoder_layers=n_decoder,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout
        )
        # self.eval_classes = n_query
        self.prototype = nn.Embedding(n_query, hidden_dim)
        self.weight = torch.linspace(0, 1, n_query+1, requires_grad=False).to(device)[1:] ## torch.linspace(0, 1, n_query, requires_grad=False).to(device)
        self.weight_etf = torch.linspace(0, 1, 64 * n_query, requires_grad=False).to(device)
        # 
        self.score_len = 10
        d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.Linear(64, self.score_len),
            #nn.Conv1d(n_query, self.score_len,1,1),
        )
        self.key_len = 3
        self.regressor = nn.ModuleList(
            [ d for i in range(self.key_len) ]
        )
        d = nn.Sequential(
            nn.Linear(n_query, 1, bias=False),
        )
        self.regressori = nn.ModuleList(
            [ d for i in range(self.key_len) ]
        )
        self.eval_classes = n_query
        print("welcome ot", self.score_len)
        etf_vec, etf_rect = eth( self.score_len, self.score_len) # 1000 score etf_rect[:,4]
        self.register_buffer('etf_vec1', etf_vec) # 百
        self.register_buffer('etf_vec2', etf_vec) # 个
        self.register_buffer('etf_vec3', etf_vec) # 小数位
        self.etf_rect = etf_rect
        self.val = False

    def forward(self, x):
        # x (b, t, c)
        b, t, c = x.shape
        #print("in:", x.shape) # in: torch.Size([32, 68, 1024])
        x = self.in_proj(x.transpose(1, 2)).transpose(1, 2)  
        # print("proj:", x.shape) # proj: torch.Size([32, 68, 256])
        q = self.prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        encode_x = self.transformer.encoder(x)
        # print("encoder:", encode_x.shape) encoder: torch.Size([32, 68, 256])
        q1 = self.transformer.decoder(q, encode_x)  
        #print("q1:", q1.shape) # q1: torch.Size([32, 4, 256])
        normd_list = []
        for i in range(self.key_len):
            s = self.regressor[i](q1)  # (b, 4, n)  32, 4 , 10 
            #print("s.shape:", s.shape)
            s = s.permute(0,2,1)
            #s = torch.diagonal(s, dim1=-2, dim2=-1)  # (b, n) # 32 4
            #print("s1: ", s.shape)
            s = self.regressori[i](s).squeeze(-1)
            norm_s = torch.sigmoid(s)
            norm_d = self.pre_logits(norm_s)
            normd_list.append(norm_d)
        return {'output': normd_list, 'embed': q1}

    def pre_old(self, norm_s, b):
        norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        out = torch.sum(self.weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)
        return out
    
    def pre_logits(self, x):
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x
    
    def re_proj(self, x):
        return torch.argmax(x,0)

    def re_proj1(self, x, gt_label = None):
        x = torch.bmm( x.unsqueeze(1) , self.ref_poj[gt_label].unsqueeze(-1))
        return x

    def get_proj_class(self, gt_label):
        gt_label = (gt_label * 1000).long()
        # print('gt_label:', gt_label, self.etf_vec.shape)
        g_1 = gt_label//100
        g_2 = gt_label//10 - g_1*10
        g_3 = gt_label - g_2 * 10 - g_1*100
        print("label:",gt_label[0], g_1[0], g_2[0], g_3[0])
        target_1 = self.etf_vec1[:, g_1].t()
        target_2 = self.etf_vec2[:, g_2].t()
        target_3 = self.etf_vec3[:, g_3].t()
        target = [target_1, target_2, target_3]
        # assert False, 'break here'
        return target

    def get_score(self, x):
        cls_score1 = self.etf_vec1[:] @ x[0].T
        cls_score2 = self.etf_vec2[:] @ x[1].T
        cls_score3 = self.etf_vec3[:] @ x[2].T
        c_1 = self.re_proj(cls_score1)
        c_2 = self.re_proj(cls_score2)
        c_3 = self.re_proj(cls_score3)
        # print("cls score :", cls_score.shape)
        score = c_1 * 100 + c_2 * 10 + c_3
        print("score:", score, c_1, c_2, c_3)
        return score # c_1 * 100 + c_2 * 10 + c_3

    def eth_head(self, x = None, gt_label = None):
        target = 0
        if gt_label is None:
            target = self.get_score(x)
        if x is None:
            target = self.get_proj_class(gt_label)
        return target
# 主力model
class GDLTETH_100(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_head, n_encoder, n_decoder, n_query, dropout):
        super(GDLTETH, self).__init__()
        self.in_proj = nn.Sequential(
            nn.Conv1d(kernel_size=1, in_channels=in_dim, out_channels=in_dim // 2),
            nn.BatchNorm1d(in_dim // 2),
            nn.ReLU(),
            nn.Conv1d(kernel_size=1, in_channels=in_dim // 2, out_channels=hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=n_encoder,
            num_decoder_layers=n_decoder,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout
        )
        # self.eval_classes = n_query
        self.prototype = nn.Embedding(n_query, hidden_dim)
        self.weight = torch.linspace(0, 1, n_query, requires_grad=False).to(device)
        self.weight_etf = torch.linspace(0, 1, 64 * n_query, requires_grad=False).to(device)
        # 
        self.score_len = 1000
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
             nn.Linear(hidden_dim, hidden_dim),
              nn.Linear(hidden_dim, hidden_dim),
               nn.Linear(hidden_dim, self.score_len),
            # nn.Conv1d(n_query,self.score_len,1,1),
        )
        self.regressor1 = nn.Sequential(
            nn.Linear(n_query, 1, bias = False),
            # nn.Conv1d(n_query,self.score_len,1,1),
        )
        self.eval_classes = n_query
        
        etf_vec, etf_rect = eth( self.score_len, self.score_len) # 1000 score etf_rect[:,4]
        self.register_buffer('etf_vec', etf_vec)
        self.etf_rect = etf_rect
        self.val = False

    def forward(self, x):
        # x (b, t, c)
        b, t, c = x.shape

        # print("in:", x.shape)
        x = self.in_proj(x.transpose(1, 2)).transpose(1, 2)
        # print("proj:", x.shape)
        q = self.prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        encode_x = self.transformer.encoder(x)
        # print("encoder:", encode_x.shape)
        q1 = self.transformer.decoder(q, encode_x)  # b 4 256 1000  1 1000
        # print("q1:", q1.shape)
      
        s = self.regressor(q1)  # (b, n, n) 4 , 4, #  32 4 1000

        s = s.permute(0,2,1)
        s = self.regressor1(s).squeeze(-1)
        # s = (s @ self.weight).squeeze(-1)
        # print("s:", s.shape)
        # s = torch.diagonal(s, dim1=-2, dim2=-1)  # (b, n)
        # print("s1: ", s.shape)
        norm_s = torch.sigmoid(s)
        norm_d = self.pre_logits(norm_s)
       
        return {'output': norm_d, 'embed': q1}

    def pre_old(self, norm_s, b):
        norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        out = torch.sum(self.weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)
        return out
    
    def pre_logits(self, x):
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x
    
    def re_proj(self, x):
        return torch.argmax(x,0)

    def re_proj1(self, x, gt_label = None):
        x = torch.bmm( x.unsqueeze(1) , self.ref_poj[gt_label].unsqueeze(-1))
        return x

    def get_proj_class(self, gt_label):
        gt_label = (gt_label * 1000).long()
        # print('gt_label:', gt_label, self.etf_vec.shape)
        target = self.etf_vec[:, gt_label].t()
        # assert False, 'break here'
        return target

    def get_score(self, x):
        cls_score = self.etf_vec[:] @ x.T
        # print("cls score :", cls_score.shape)
        return self.re_proj(cls_score)/self.score_len
    
    def eth_head(self, x = None, gt_label = None):
        target = 0
        if gt_label is None:
            target = self.get_score(x)
        if x is None:
            target = self.get_proj_class(gt_label)
        return target
