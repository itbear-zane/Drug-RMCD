import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class TransformerEncoderModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, nhead=8, dim_feedforward=2048, dropout=0.1, \
        activation='relu', max_len=5000, num_directions=2):
        super(TransformerEncoderModel, self).__init__()
        
        # 参数初始化
        self.input_size = input_size  # 输入特征维度
        self.hidden_size = hidden_size // num_directions  # 单向隐藏层大小
        self.num_layers = num_layers
        self.num_directions = num_directions  # 双向
        
        # Transformer 参数
        self.d_model = hidden_size  # Transformer 的隐藏层大小
        self.nhead = nhead  # 多头注意力头数
        self.num_encoder_layers = num_layers  # 编码器层数
        self.dim_feedforward = dim_feedforward  # 前馈网络维度
        
        # 输入线性变换层（将 input_size 映射到 d_model）
        self.input_projection = nn.Linear(self.input_size, self.d_model)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False  # 注意：PyTorch 默认 batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)
        
        # 输出线性变换层（将 d_model 映射到 hidden_size * num_directions）
        self.output_projection = nn.Linear(self.d_model, self.hidden_size * self.num_directions)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(self.d_model, dropout, max_len)

    def forward(self, src):
        """
        src: (batch_size, sequence_length, input_size)
        """
        # 1. 输入线性变换
        src = self.input_projection(src)  # (batch_size, sequence_length, d_model)
        
        # 2. 调整维度以适应 Transformer 输入
        src = src.permute(1, 0, 2)  # (sequence_length, batch_size, d_model)
        
        # 3. 添加位置编码
        src = self.positional_encoding(src)  # (sequence_length, batch_size, d_model)
        
        # 4. Transformer 编码器
        output = self.transformer_encoder(src)  # (sequence_length, batch_size, d_model)
        
        # 5. 调整维度回原始顺序
        output = output.permute(1, 0, 2)  # (batch_size, sequence_length, d_model)
        
        # 6. 输出线性变换
        output = self.output_projection(output)  # (batch_size, sequence_length, hidden_size * num_directions)
        
        # 7. 获取最后一个时间步的隐藏状态 h_n
        h_n = output[:, -1, :]  # (batch_size, hidden_size * num_directions)
        h_n = h_n.unsqueeze(0).repeat(self.num_layers * self.num_directions, 1, 1)  # (num_layers * num_directions, batch_size, hidden_size)
        
        return output, h_n

class TransformerDecoderModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, nhead=8, dim_feedforward=2048, dropout=0.1, \
        activation='relu', max_len=5000, num_directions=2):
        super(TransformerDecoderModel, self).__init__()
        
        # 参数初始化
        self.input_size = input_size  # 输入特征维度
        self.hidden_size = hidden_size // num_directions  # 单向隐藏层大小
        self.num_layers = num_layers
        self.num_directions = num_directions  # 双向
        
        # Transformer 参数
        self.d_model = hidden_size  # Transformer 的隐藏层大小
        self.nhead = nhead  # 多头注意力头数
        self.num_decoder_layers = num_layers  # 解码器层数
        self.dim_feedforward = dim_feedforward  # 前馈网络维度
        
        # 输入线性变换层（将 input_size 映射到 d_model）
        self.input_projection = nn.Linear(self.input_size, self.d_model)
        
        # Transformer 解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False  # 注意：PyTorch 默认 batch_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_decoder_layers)
        
        # 输出线性变换层（将 d_model 映射到 hidden_size * num_directions）
        self.output_projection = nn.Linear(self.d_model, self.hidden_size * self.num_directions)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(self.d_model, dropout, max_len)

    def forward(self, src):
        """
        src: (batch_size, sequence_length, input_size)
        """
        batch_size, sequence_length, _ = src.size()
        
        # 1. 输入线性变换
        src = self.input_projection(src)  # (batch_size, sequence_length, d_model)
        
        # 2. 调整维度以适应 Transformer 输入
        src = src.permute(1, 0, 2)  # (sequence_length, batch_size, d_model)
        
        # 3. 添加位置编码
        src = self.positional_encoding(src)  # (sequence_length, batch_size, d_model)
        
        # 4. 创建因果掩码
        tgt_mask = self.generate_square_subsequent_mask(sequence_length).to(src.device)
        
        # 5. Transformer 解码器
        memory = src  # 使用输入序列作为记忆张量
        output = self.transformer_decoder(tgt=src, memory=memory, tgt_mask=tgt_mask)  # (sequence_length, batch_size, d_model)
        
        # 6. 调整维度回原始顺序
        output = output.permute(1, 0, 2)  # (batch_size, sequence_length, d_model)
        
        # 7. 输出线性变换
        output = self.output_projection(output)  # (batch_size, sequence_length, hidden_size * num_directions)
        
        # 8. 获取最后一个时间步的隐藏状态 h_n
        h_n = output[:, -1, :]  # (batch_size, hidden_size * num_directions)
        h_n = h_n.unsqueeze(0).repeat(self.num_layers * self.num_directions, 1, 1)  # (num_layers * num_directions, batch_size, hidden_size)
        
        return output, h_n

    @staticmethod
    def generate_square_subsequent_mask(sz):
        """生成因果掩码"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# 位置编码模块
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1, cfg=None):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = True
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        """_summary_

        Args:
            v (_type_): bs, n_img, dim
            l (_type_): bs, n_text, dim
            attention_mask_v (_type_, optional): _description_. bs, n_img
            attention_mask_l (_type_, optional): _description_. bs, n_text

        Returns:
            _type_: _description_
        """
        # if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
        #     import ipdb; ipdb.set_trace()
        bsz, tgt_len, _ = v.size()

        query_states = self.v_proj(v) * self.scale
        key_states = self._shape(self.l_proj(l), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))  # bs*nhead, nimg, ntxt

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(
                attn_weights, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(
                attn_weights, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[0]
        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(
                attn_weights_l, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(
                attn_weights_l, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        # mask vison for language
        if attention_mask_v is not None:
            attention_mask_v = (
                attention_mask_v[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn_weights_l.masked_fill_(attention_mask_v, float("-inf"))

        attn_weights_l = attn_weights_l.softmax(dim=-1)

        # mask language for vision
        if attention_mask_l is not None:
            attention_mask_l = (
                attention_mask_l[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn_weights.masked_fill_(attention_mask_l, float("-inf"))
        attn_weights_v = attn_weights.softmax(dim=-1)

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)

        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
            )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_v, attn_output_l

# Bi-Direction MHA (text->image, image->text)
class BiAttentionBlock(nn.Module):
    def __init__(
        self,
        v_dim,
        l_dim,
        embed_dim,
        num_heads,
        dropout=0.1,
        drop_path=0.0,
        init_values=1e-4,
        cfg=None,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlock, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(
            v_dim=v_dim, l_dim=l_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True)

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(
            v, l, attention_mask_v=attention_mask_v, attention_mask_l=attention_mask_l
        )
        # v, l = v + delta_v, l + delta_l
        v = v + self.drop_path(self.gamma_v * delta_v)
        l = l + self.drop_path(self.gamma_l * delta_l)
        return v, l

class Projector(nn.Module):
    """
    A neural network module used for simple classification tasks. It consists of a two-layered linear network
    with a nonlinear activation function in between.

    Attributes:
        - linear1: The first linear layer.
        - linear2: The second linear layer that outputs to the desired dimensions.
        - activation_fn: The nonlinear activation function.
    """

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
    ):
        """
        Initializes the Projector module.

        :param input_dim: Dimension of the input features.
        :param out_dim: Dimension of the output.
        :param activation_fn: The activation function to use.
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.activation_fn = get_activation_fn(activation_fn)

    def forward(self, x):
        """
        Forward pass of the Projector.

        :param x: Input tensor to the module.

        :return: Tensor after passing through the network.
        """
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x
    
def get_activation_fn(activation):
    """Returns the activation function corresponding to `activation`"""

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

class Embedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim, pretrained_embedding=None):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embedding is not None:
            self.embedding.weight.data = torch.from_numpy(pretrained_embedding)
        self.embedding.weight.requires_grad = False

    def forward(self, x):
        """
        Inputs:
        x -- (batch_size, seq_length)
        Outputs
        shape -- (batch_size, seq_length, embedding_dim)
        """
        return self.embedding(x)


class DrugEmbedding(nn.Module):
    def __init__(
        self,
        if_biattn = True,
        drug_pretrained_dim = 768, 
        prot_pretrained_dim = 1024, 
        embed_dim = 512, 
        num_heads = 8, 
    ):
        super().__init__()
        """Constructor for the model."""
        
        self.drug_project = Projector(
            input_dim = drug_pretrained_dim,
            out_dim = embed_dim,
            activation_fn = 'relu'
        )

        self.prot_project = Projector(
            input_dim = prot_pretrained_dim,
            out_dim = embed_dim,
            activation_fn = 'relu'
        )

        self.if_biattn = if_biattn
        if if_biattn:
            self.bi_attention = BiAttentionBlock(
                v_dim=512,
                l_dim=512,
                embed_dim=embed_dim,
                num_heads=num_heads,
            )
    
    def forward(self, inputs, masks):
        drug_embed, prot_embed, drug_mask, prot_mask = inputs[0], inputs[1], masks[0], masks[1]
        
        drug_embed = self.drug_project(drug_embed)
        prot_embed = self.prot_project(prot_embed)

        if self.if_biattn:
            drug_embed, prot_embed = self.bi_attention(drug_embed, prot_embed, ~drug_mask.bool(), ~prot_mask.bool())
        
        # print(torch.any(torch.isnan(drug_embed)).item(), torch.any(torch.isnan(prot_embed)).item())
        # print("-" * 20)

        embedding = torch.cat([drug_embed, prot_embed], dim=1)
        mask = torch.cat([drug_mask, prot_mask], dim=1)

        return embedding, mask

class Rnn(nn.Module):

    def __init__(self, cell_type, embedding_dim, hidden_dim, num_layers):
        super(Rnn, self).__init__()
        if cell_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=embedding_dim,
                               hidden_size=hidden_dim // 2,
                               num_layers=num_layers,
                               batch_first=True,
                               bidirectional=True)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(input_size=embedding_dim,
                              hidden_size=hidden_dim // 2,
                              num_layers=num_layers,
                              batch_first=True,
                              bidirectional=True)
        else:
            raise NotImplementedError('cell_type {} is not implemented'.format(cell_type))

    def forward(self, x):
        """
        Inputs:
        x - - (batch_size, seq_length, input_dim)
        Outputs:
        h - - bidirectional(batch_size, seq_length, hidden_dim)
        """
        h = self.rnn(x)
        return h
    

# class Classifier(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
#         super(Classifier, self).__init__()
#         self.fc1 = nn.Linear(in_dim, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.bn2 = nn.BatchNorm1d(hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, out_dim)
#         self.bn3 = nn.BatchNorm1d(out_dim)
#         self.fc4 = nn.Linear(out_dim, binary)

#     def forward(self, x):
#         x = self.bn1(F.relu(self.fc1(x)))
#         x = self.bn2(F.relu(self.fc2(x)))
#         x = self.bn3(F.relu(self.fc3(x)))
#         x = self.fc4(x)
#         return x


class Sp_norm_model(nn.Module):
    def __init__(self, args):
        super(Sp_norm_model, self).__init__()
        self.args = args
        # self.embedding_layer = Embedding(args.vocab_size,
        #                                  args.embedding_dim,
        #                                  args.pretrained_embedding)
        self.embedding_layer = DrugEmbedding(if_biattn=args.if_biattn, embed_dim=args.embedding_dim)
        if args.cell_type == 'GRU':
            self.gen = nn.GRU(input_size=args.embedding_dim,
                                    hidden_size=args.hidden_dim // 2,
                                    num_layers=args.num_layers,
                                    batch_first=True,
                                    bidirectional=True)
            self.cls = nn.GRU(input_size=args.embedding_dim,
                            hidden_size=args.hidden_dim // 2,
                            num_layers=args.num_layers,
                            batch_first=True,
                            bidirectional=True)
        elif args.cell_type == 'TransformerDecoder':
            self.gen = TransformerDecoderModel(args.embedding_dim, args.hidden_dim, args.num_layers)
            self.cls = TransformerDecoderModel(args.embedding_dim, args.hidden_dim, args.num_layers)
        elif args.cell_type == 'TransformerEncoder':
            self.gen = TransformerEncoderModel(args.embedding_dim, args.hidden_dim, args.num_layers)
            self.cls = TransformerEncoderModel(args.embedding_dim, args.hidden_dim, args.num_layers)
        else:
            raise ValueError(f"{args.cell_type} is not supported!")

        self.cls_fc = nn.Linear(args.hidden_dim, args.num_class)

        self.z_dim = 2
        self.gen_fc = nn.Linear(args.hidden_dim, self.z_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm1 = nn.LayerNorm(args.hidden_dim)
        self.generator=nn.Sequential(self.gen,
                                     SelectItem(0),
                                     self.layernorm1,
                                     self.dropout,
                                     self.gen_fc)


    def _independent_soft_sampling(self, rationale_logits):
        """
        Use the hidden states at all time to sample whether each word is a rationale or not.
        No dependency between actions. Return the sampled (soft) rationale mask.
        Outputs:
                z -- (batch_size, sequence_length, 2)
        """
        z = torch.softmax(rationale_logits, dim=-1)

        return z

    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    def forward(self, inputs, masks):
        embedding, masks = self.embedding_layer(inputs, masks)
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * embedding# (batch_size, seq_length, embedding_dim)
        gen_logits=self.generator(embedding)
        ########## Sample ##########
        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        ########## Classifier ##########
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits


    def train_one_step(self, inputs, masks):    #input x directly to predictor
        embedding, masks = self.embedding_layer(inputs, masks)
        masks_ = masks.unsqueeze(-1)
        # (batch_size, seq_length, embedding_dim)
        embedding = masks_ * embedding
        outputs, _ = self.cls(embedding)  # (batch_size, seq_length, hidden_dim)
        outputs = outputs * masks_ + (1. -
                                      masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        # shape -- (batch_size, num_classes)
        logits = self.cls_fc(self.dropout(outputs))
        return logits


    def get_rationale(self, inputs, masks):
        embedding, masks = self.embedding_layer(inputs, masks)
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * embedding  # (batch_size, seq_length, embedding_dim)
        gen_logits = self.generator(embedding) # (batch_size, seq_length, 2)
        ########## Sample ##########
        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        return z, masks
    
    def pred_forward_logit(self, inputs, masks,z):
        embedding, masks = self.embedding_layer(inputs, masks)
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * embedding # (batch_size, seq_length, embedding_dim)

        ########## Classifier ##########
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return cls_logits


    def g_skew(self,inputs, masks):
        embedding, masks = self.embedding_layer(inputs, masks)
        #  masks_ (batch_size, seq_length, 1)
        masks_ = masks.unsqueeze(-1)
        ########## Genetator ##########
        embedding = masks_ * embedding  # (batch_size, seq_length, embedding_dim)
        gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        gen_output = self.layernorm1(gen_output)
        gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        soft_log=self._independent_soft_sampling(gen_logits)
        return soft_log



class GenEncShareModel(nn.Module):

    def __init__(self, args):
        super(GenEncShareModel, self).__init__()
        self.args = args
        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)
        self.enc = Rnn(args.cell_type,
                       args.embedding_dim,
                       args.hidden_dim,
                       args.num_layers)
        self.z_dim = 2
        self.gen_fc = nn.Linear(args.hidden_dim, self.z_dim)
        self.cls_fc = nn.Linear(args.hidden_dim, args.num_class)
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm = nn.LayerNorm(args.hidden_dim)

    def _independent_soft_sampling(self, rationale_logits):
        """
        Use the hidden states at all time to sample whether each word is a rationale or not.
        No dependency between actions. Return the sampled (soft) rationale mask.
        Outputs:
                z -- (batch_size, sequence_length, 2)
        """
        z = torch.softmax(rationale_logits, dim=-1)

        return z

    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        z = self._independent_soft_sampling(rationale_logits)
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    # inputs (batch_size, seq_length)
    # masks (batch_size, seq_length)
    def forward(self, inputs, masks):
        #  masks_ (batch_size, seq_length, 1)
        masks_ = masks.unsqueeze(-1)
        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        gen_output, _ = self.enc(embedding)  # (batch_size, seq_length, hidden_dim)
        gen_output = self.layernorm(gen_output)
        gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        ########## Sample ##########
        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        ########## Classifier ##########
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.enc(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = self.layernorm(cls_outputs)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))

        # LSTM
        return z, cls_logits

    def train_one_step(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)
        # (batch_size, seq_length, embedding_dim)
        embedding = masks_ * self.embedding_layer(inputs)
        outputs, _ = self.enc(embedding)  # (batch_size, seq_length, hidden_dim)
        outputs = outputs * masks_ + (1. -
                                      masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        # shape -- (batch_size, num_classes)
        logits = self.cls_fc(self.dropout(outputs))
        return logits









