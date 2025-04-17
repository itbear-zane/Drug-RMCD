import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm



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


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

class Generator(nn.Module):
    def __init__(self, gen, layernorm, dropout, gen_fc):
        super().__init__()
        self.gen = gen
        self.layernorm = layernorm
        self.dropout = dropout
        self.gen_fc = gen_fc
    
    def forward(self, x, mask):
        gen_output = self.gen(x, src_key_padding_mask=~mask.bool())
        layer_output = self.layernorm(gen_output)
        drop_output = self.dropout(layer_output)
        res = self.gen_fc(drop_output)
        return res


class DrugEmbedding(nn.Module):
    def __init__(self, drug_pretrained_dim = 768, prot_pretrained_dim = 1024, embedding_dim = 512, num_heads = 4):
        super().__init__()
        """Constructor for the model."""
        
        self.drug_project = Projector(
            input_dim = drug_pretrained_dim,
            out_dim = embedding_dim,
            activation_fn = 'relu'
        )

        self.prot_project = Projector(
            input_dim = prot_pretrained_dim,
            out_dim = embedding_dim,
            activation_fn = 'relu'
        )

        self.bi_attention = BiAttentionBlock(
            v_dim=embedding_dim,
            l_dim=embedding_dim,
            embed_dim=embedding_dim,
            num_heads=num_heads,
        )
    
    def forward(self, inputs, masks):
        drug_embed, prot_embed, drug_mask, prot_mask = inputs[0], inputs[1], masks[0], masks[1]
        
        drug_embed = self.drug_project(drug_embed)
        prot_embed = self.prot_project(prot_embed)

        
        drug_embed, prot_embed = self.bi_attention(drug_embed, prot_embed, ~drug_mask.bool(), ~prot_mask.bool())
        
        # print(torch.any(torch.isnan(drug_embed)).item(), torch.any(torch.isnan(prot_embed)).item())
        # print("-" * 20)

        embedding = torch.cat([drug_embed, prot_embed], dim=1)
        mask = torch.cat([drug_mask, prot_mask], dim=1)

        return embedding, mask

class DrugRMCD(nn.Module):
    def __init__(self, args):
        super(DrugRMCD, self).__init__()
        self.args = args
        self.embedding_layer = DrugEmbedding(embedding_dim=args.embedding_dim, num_heads=args.num_heads)

        self.generator = self._init_generator(args)
        self.encoder = self._init_encoder(args)

        self.cls_fc = Classifier(in_dim=args.mlp_in_dim, hidden_dim=args.mlp_hidden_dim, out_dim=args.mlp_in_dim, binary=args.num_class)
        
    def _init_generator(self, args):
        """
        Initialize the generator.
        """
        gen = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=args.embedding_dim,
                                    nhead=args.num_heads, 
                                    dim_feedforward=args.dim_feedforward,
                                    dropout=args.dropout, 
                                    batch_first=True),
            num_layers=args.num_layers
        )
        layernorm = nn.LayerNorm(args.embedding_dim)
        dropout = nn.Dropout(args.dropout)
        gen_fc = nn.Linear(args.embedding_dim, 2)
        generator = Generator(gen,
                            layernorm,
                            dropout,
                            gen_fc)
        return generator
    
    def _init_encoder(self, args):
        """
        Initialize the encoder.
        """
        encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=args.embedding_dim,
                                    nhead=args.num_heads, 
                                    dim_feedforward=args.dim_feedforward,
                                    dropout=args.dropout, 
                                    batch_first=True),
            num_layers=args.num_layers
        )
        return encoder
        
    def independent_straight_through_sampling(self, rationale_logits):
        """
        Use the hidden states at all time to sample whether each word is a rationale or not.
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
        embedding = embedding * masks_
        gen_logits=self.generator(embedding, masks.bool()) # (batch_size, seq_length, 2)
        ########## Sample ##########
        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        ########## Classifier ##########
        embedding = embedding * z[:, :, 1].unsqueeze(-1)
        cls_outputs = self.encoder(embedding, src_key_padding_mask=~z[:, :, 1].bool())  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. - masks_) * (-1e6)

        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(cls_outputs)
        # shape -- (batch_size, num_classes)
        return z, cls_logits

    def train_one_step(self, inputs, masks):    #input x directly to predictor
        embedding, masks = self.embedding_layer(inputs, masks)
        masks_ = masks.unsqueeze(-1)

        embedding = embedding * masks_
        cls_outputs = self.encoder(embedding, src_key_padding_mask=~masks.bool())  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. - masks_) * (-1e6)

        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(cls_outputs)
        return cls_logits


    def get_rationale(self, inputs, masks):
        ########## Genetator ##########
        embedding, masks = self.embedding_layer(inputs, masks)  # (batch_size, seq_length, embedding_dim)
        mask_ = masks.unsqueeze(-1)
        embedding = embedding * mask_
        gen_logits = self.generator(embedding, masks.bool()) # (batch_size, seq_length, 2)    
        ########## Sample ##########
        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)

        return z, masks
    
    def pred_forward_logit(self, inputs, masks, z):
        embedding, masks = self.embedding_layer(inputs, masks) # (batch_size, seq_length, embedding_dim)
        masks_ = masks.unsqueeze(-1)
        ########## Classifier ##########
        embedding = embedding * masks_
        cls_outputs = self.encoder(embedding, src_key_padding_mask=~z[:,:,1].bool())  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. - masks_) * (-1e6)

        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(cls_outputs)
        return cls_logits









