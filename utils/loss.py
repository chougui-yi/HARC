import torch
import torch.nn as nn

# class ETF_Loss(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#     def cal_orthogonal(self, matrix):  
#         identity = torch.eye(matrix.size(-1), device=matrix.device)  
#         return (matrix - identity).mean()
    
#     def forward(self, x):
#         return self.cal_orthogonal(x)
    
class ETF_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def cal_orthogonal(self, matrix, gt):  
        # print(matrix.shape, gt.shape)
        # product = torch.bmm( matrix, matrix.permute(0,2,1))  
        gt = torch.bmm( gt, gt.permute(0,2,1))  
        # print(product.shape, gt.shape)
        return ( matrix - gt).mean()
    
    def cal_orthogonal_eye(self, matrix, gt):  
        identity = torch.eye(matrix.size(-1), device=matrix.device)  
        return self.mse( matrix, identity) # ( matrix - identity).mean()
    
    def forward(self, x, gt):
        gt = gt.unsqueeze(-1)
        return self.cal_orthogonal( x, gt)