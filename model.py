import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm



class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)
        return logits, att_maps


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BCNet(nn.Module):
    """Simple class for non-linear bilinear connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/bc.py
    """

    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=[.2, .5], k=3):
        super(BCNet, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim;
        self.q_dim = q_dim
        self.h_dim = h_dim;
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1])  # attention
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if None == h_out:
            pass
        elif h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

    def forward(self, v, q):
        if None == self.h_out:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            logits = torch.einsum('bvk,bqk->bvqk', (v_, q_))
            return logits

        # low-rank bilinear pooling using einsum
        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v))
            q_ = self.q_net(q)
            logits = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
            return logits  # b x h_out x v x q

        # batch outer product, linear projection
        # memory efficient but slow computation
        else:
            v_ = self.dropout(self.v_net(v)).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            logits = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            return logits.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v)  # b x v x d
        q_ = self.q_net(q)  # b x q x d
        logits = torch.einsum('bvk,bvq,bqk->bk', (v_, w, q_))
        if 1 < self.k:
            logits = logits.unsqueeze(1)  # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k  # sum-pooling
        return logits


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


class Sp_norm_model(nn.Module):
    def __init__(self, args):
        super(Sp_norm_model, self).__init__()
        self.args = args
        self.drug_embedding_layer = Projector(input_dim=768,
                                            out_dim=args.embedding_dim, 
                                            activation_fn='relu')
        self.prot_embedding_layer = Projector(input_dim=1024, 
                                            out_dim=args.embedding_dim,
                                            activation_fn='relu')
        
        self.drug_generator = self._init_generator(args)
        self.prot_generator = self._init_generator(args)

        self.drug_encoder = self._init_encoder(args)
        self.prot_encoder = self._init_encoder(args)

        self.bi_attention = weight_norm(
            BANLayer(v_dim=args.embedding_dim, q_dim=args.embedding_dim, h_dim=args.mlp_in_dim, h_out=args.ban_heads), 
            name='h_mat', dim=None)

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
        drug_embedding = self.drug_embedding_layer(inputs[0])
        prot_embedding = self.prot_embedding_layer(inputs[1])

        drug_masks = masks[0]
        prot_masks = masks[1]

        ########## Genetator ##########
        drug_gen_logits=self.drug_generator(drug_embedding, drug_masks) # (batch_size, seq_length, 2)
        prot_gen_logits=self.prot_generator(prot_embedding, prot_masks) # (batch_size, seq_length, 2)
        ########## Sample ##########
        drug_z = self.independent_straight_through_sampling(drug_gen_logits)  # (batch_size, seq_length, 2)
        prot_z = self.independent_straight_through_sampling(prot_gen_logits)  # (batch_size, seq_length, 2)
        ########## Classifier ##########
        drug_enc_output = self.drug_encoder(drug_embedding, src_key_padding_mask=~drug_z[:, :, 1].bool())  # (batch_size, seq_length, hidden_dim)
        prot_enc_output = self.prot_encoder(prot_embedding, src_key_padding_mask=~prot_z[:, :, 1].bool())  # (batch_size, seq_length, hidden_dim)

        fusion_output, _ = self.bi_attention(drug_enc_output, prot_enc_output) # (batch_size, hidden_dim)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(fusion_output)
        return drug_z, prot_z, cls_logits

    def train_one_step(self, inputs, masks):    #input x directly to predictor
        drug_embedding = self.drug_embedding_layer(inputs[0])
        prot_embedding = self.prot_embedding_layer(inputs[1])

        drug_masks= masks[0]
        prot_masks = masks[1]

        drug_enc_output = self.drug_encoder(drug_embedding, src_key_padding_mask=~drug_masks.bool())  # (batch_size, seq_length, hidden_dim)
        prot_enc_output = self.prot_encoder(prot_embedding, src_key_padding_mask=~prot_masks.bool())  # (batch_size, seq_length, hidden_dim)

        fusion_output, _ = self.bi_attention(drug_enc_output, prot_enc_output) # (batch_size, hidden_dim)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(fusion_output)
        return cls_logits


    def get_rationale(self, inputs, masks, type=None):
        ########## Genetator ##########
        if type == 'drug':
            embedding = self.drug_embedding_layer(inputs[0])  # (batch_size, seq_length, embedding_dim)
            mask = masks[0] == 1
            gen_logits = self.drug_generator(embedding, mask) # (batch_size, seq_length, 2)    
        elif type == 'prot':
            embedding = self.prot_embedding_layer(inputs[1])  # (batch_size, seq_length, embedding_dim)
            mask = masks[1] == 1
            gen_logits = self.prot_generator(embedding, mask)  # (batch_size, seq_length, 2)
        else:
            raise NotImplementedError('type {} is not implemented'.format(type))
        ########## Sample ##########
        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)

        mask = mask.unsqueeze(-1)
        return z, mask            
    
    def pred_forward_logit(self, inputs, drug_z, prot_z):
        drug_embedding = self.drug_embedding_layer(inputs[0]) # (batch_size, seq_length, embedding_dim)
        prot_embedding = self.prot_embedding_layer(inputs[1]) # (batch_size, seq_length, embedding_dim)

        ########## Classifier ##########
        drug_enc_output = self.drug_encoder(drug_embedding, src_key_padding_mask=drug_z[:, :, 1].bool())  # (batch_size, seq_length, hidden_dim)
        prot_enc_output = self.prot_encoder(prot_embedding, src_key_padding_mask=prot_z[:, :, 1].bool())  # (batch_size, seq_length, hidden_dim)
        fusion_output, _ = self.bi_attention(drug_enc_output, prot_enc_output) # (batch_size, hidden_dim)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(fusion_output)
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









