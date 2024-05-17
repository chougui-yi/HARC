import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights
    

class BiGRUA(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiGRUA, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.attn = Attention( hidden_size * 2 )
        
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # 双向GRU
        out, _ = self.gru(x, h0)
        # print(out.shape)
        out1, _ = self.attn(out, torch.ones_like(out))
        out = out1[:, -1, :]
        
        # 全连接层
        # out = self.fc(out)
        return out, out1

class BIGRU_Attention(nn.Module):
    def __init__(self, in_dim, class_number = 6):
        super(BIGRU_Attention, self).__init__()
        hidden_dim = in_dim
        self.in_proj = nn.Sequential(
            nn.Conv1d(kernel_size=1, in_channels=32, out_channels=hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv1d(kernel_size=1, in_channels=hidden_dim // 2, out_channels=hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        ) 
        
        self.in_proj_n = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 64)
            ) for i in range(4)
        ])
        
        self.bigrua = nn.Sequential(
            BiGRUA( hidden_dim, hidden_dim, 4, 6)
        )
        self.fc = nn.Linear(in_dim * 2, in_dim)
        
        self.regressor = nn.Sequential(
            nn.Linear(768 , 512),
            nn.Linear(512, 256),
            nn.Linear(256, 64),
            nn.Linear(64, class_number),
            nn.Sigmoid(),
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x_arg, x_code, x_ast, x_node_type, x_deep, x_node_total, x_width, x_entropy = x
        x_arg =  self.in_proj(x_arg.permute(0,2,1)).permute(0,2,1)
      
        # print( x_arg.shape, x_code.shape, x_ast.shape, x_node_type.shape  )
        x_deep = self.in_proj_n[0](x_deep).unsqueeze(1)
        x_node_total = self.in_proj_n[1](x_node_total).unsqueeze(1)
        x_width = self.in_proj_n[2](x_width).unsqueeze(1)
        x_entropy = self.in_proj_n[3](x_entropy).unsqueeze(1) # (1, 64)
        
        x_feature_1 = torch.cat( [ x_arg, x_code, x_ast, x_node_type], 1)
        x_feature_2 = torch.cat( [ x_deep, x_node_total, x_width, x_entropy], 1)
        # print(x_feature_1.shape, x_feature_2.shape)
        y, q = self.bigrua(x_feature_1)
        # print(y.shape, q.shape)
        q = self.fc(y)
        y = self.regressor( torch.cat([q, self.flatten(x_feature_2)], -1) )
        return y, q



class BIGRU_Attention_(nn.Module):
    def __init__(self, in_dim, class_number = 6):
        super().__init__()
        hidden_dim = in_dim
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
        
        self.bigrua = nn.Sequential(
            BiGRUA( hidden_dim, hidden_dim, 4, 6)
        )
        self.fc = nn.Linear(in_dim * 2, in_dim)
        
        self.regressor = nn.Sequential(
            # nn.Linear(768 , 512),
            nn.Linear(512, 256),
            nn.Linear(256, 64),
            nn.Linear(64, class_number),
            nn.Sigmoid(),
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x[0]
        b,c,t = x.shape
     
        if t == 32:
            x_feature =  self.in_proj_head_32(x.permute(0,2,1)).permute(0,2,1)
        else:
            x_feature =  self.in_proj_head_512(x.permute(0,2,1)).permute(0,2,1)
    
        y, q = self.bigrua(x_feature)
        # print(y.shape, q.shape)
        q = self.fc(y)
        # print(q.shape)
        y = self.regressor( q )
        return y, q
