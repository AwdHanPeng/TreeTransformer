"""
Implementation of "Attention is All You Need"
"""
import math
from tkinter import N
from tkinter.messagebox import NO
from regex import F
import torch.nn as nn
import torch
from trlib.modules.util_class import LayerNorm
from trlib.modules.multi_head_attn import MultiHeadedAttention
from trlib.modules.position_ffn import PositionwiseFeedForward
from trlib.encoders.encoder import EncoderBase
from trlib.utils.misc import sequence_mask, get_rel_mask
from trlib.modules.ggnn_module import GGNN


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.
    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, args,
                 d_model,
                 heads,
                 d_ff,
                 d_k,
                 d_v,
                 dropout,
                 max_relative_positions=0,
                 use_neg_dist=True,
                 use_tree_relative_attn=False,
                 tree_rel_vocab_size=0,
                 use_td_tree_pos_enc=False,
                 use_local_relation=False):
        super(TransformerEncoderLayer, self).__init__()

        self.attention = MultiHeadedAttention(args, heads,
                                              d_model,
                                              d_k,
                                              d_v,
                                              dropout=dropout,
                                              max_relative_positions=max_relative_positions,
                                              use_neg_dist=use_neg_dist,
                                              use_tree_relative_attn=use_tree_relative_attn,
                                              tree_rel_vocab_size=tree_rel_vocab_size,
                                              use_td_tree_pos_enc=use_td_tree_pos_enc,
                                              use_local_relation=use_local_relation)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout, args)
        self.pre_norm = args.pre_norm

    def forward(self, inputs, mask, rel_matrix=None, rel_mask=None, input_tokens=None, td_scores=None, LR=None):
        """
        Transformer Encoder Layer definition.
        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`
        Returns:
            (`FloatTensor`):
            * outputs `[batch_size x src_len x model_dim]`
        """

        if self.pre_norm:
            '''
            if self.pre_norm:
            return x + self.dropout(sublayer(self.norm(x)))
            else:
            return self.norm(x + self.dropout(sublayer(x)))
            '''
            inputs = self.layer_norm(inputs)
            context, attn_per_head, _ = self.attention(inputs, inputs, inputs,
                                                       mask=mask, rel_matrix=rel_matrix,
                                                       rel_mask=rel_mask,
                                                       input_tokens=input_tokens,
                                                       attn_type="self",
                                                       td_scores=td_scores, LR=LR)
            out = self.dropout(context) + inputs
        else:
            context, attn_per_head, _ = self.attention(inputs, inputs, inputs,
                                                       mask=mask, rel_matrix=rel_matrix,
                                                       rel_mask=rel_mask,
                                                       input_tokens=input_tokens,
                                                       attn_type="self",
                                                       td_scores=td_scores, LR=LR)
            out = self.layer_norm(self.dropout(context) + inputs)
        return self.feed_forward(out), attn_per_head


class LocalRelationEmbedding(nn.Module):
    def __init__(self, args, d_model, heads):
        super().__init__()
        self.hidden = d_model
        self.attn_heads = heads
        self.d_k = self.hidden // self.attn_heads
        self.args = args
        self.max_ary = self.args.td_max_ary
        self.max_depth = self.args.td_max_depth
        self.action_size = self.args.action_size
        self.tree_TD = self.args.tree_TD
        self.td_add = args.td_add
        self.td_cat = args.td_cat
        self.bi_linear = args.bi_linear
        self.no_PN = args.no_PN
        self.only_1, self.only_2 = args.only_1, args.only_2
        if self.tree_TD:
            self.length = self.max_ary * \
                          (self.max_ary - 1) // 2 + self.max_ary + 1
            if self.td_add or self.td_cat:
                self.embedding_a = nn.Embedding(
                    self.max_ary + 1, self.action_size, padding_idx=0)  # for child
                nn.init.xavier_normal_(self.embedding_a.weight)
                self.embedding_b = nn.Embedding(
                    self.max_ary + 1, self.action_size, padding_idx=0)  # for father
                nn.init.xavier_normal_(self.embedding_b.weight)

                self.a_idx, self.b_idx = [0], [0]
                for i in range(1, self.max_ary + 1):
                    self.a_idx += [j for j in range(1, i + 1)]
                    self.b_idx += [i for j in range(1, i + 1)]
                assert len(self.a_idx) == self.length
            if self.td_cat:
                self.td_cat_linear = nn.Linear(2 * self.action_size, self.action_size)
            else:
                self.embedding = nn.Embedding(
                    self.length, self.action_size, padding_idx=0)
                nn.init.xavier_normal_(self.embedding.weight)

        else:
            self.length = self.max_ary + 1
            self.embedding = nn.Embedding(
                self.length, self.action_size, padding_idx=0)
            nn.init.xavier_normal_(self.embedding.weight)
        if self.bi_linear:
            self.w_1 = nn.Linear(self.action_size, self.hidden)
            self.w_2 = nn.Linear(self.action_size, self.hidden)
        else:
            self.w = nn.Linear(self.action_size, self.hidden)
        self.linear = nn.ModuleList(
            [nn.Linear(self.hidden, self.hidden) for _ in range(2)])
        self.norm = nn.LayerNorm(self.hidden)

    def forward(self, relations):
        '''
        relations : bs,n,n,3
        actually we dont need this input
        we can convert the relation to n*n gather map here.
        '''
        # stage 1, cat the pos and neg
        if self.tree_TD:
            if self.td_add or self.td_cat:
                pos_a = self.embedding_a(torch.tensor(self.a_idx, dtype=torch.int, device=relations.device))  # child
                pos_b = self.embedding_b(torch.tensor(self.b_idx, dtype=torch.int, device=relations.device))  # father
                # (0,0) (1,1) (1,2) (2,2) (1,3) (2,3) (3,3) ...
                # [0,1,1,2,1,2,3,1,2,3,4] totalNums = self.length
                # [0,1,2,2,3,3,3,4,4,4,4] totalNums = self.length
                if self.td_add:
                    pos = pos_a + pos_b
                elif self.td_cat:
                    pos = self.td_cat_linear(torch.cat([pos_a, pos_b], dim=-1))
                else:
                    raise Exception('No Valid TD WAY')
            else:
                pos = self.embedding(torch.arange(0, self.length, device=relations.device))  # n,dim
        else:
            pos = self.embedding(torch.arange(0, self.length, device=relations.device))  # n,dim
        if self.bi_linear:
            POS = self.w_1(pos)  # n,hidden
            NEG = self.w_2(pos)  # n,hidden
            final = torch.cat([POS, NEG], dim=0)  # 2*n,hidden
        else:
            if self.no_PN:
                neg = pos
            else:
                neg = pos * (-1)
            final = self.w(torch.cat([pos, neg], dim=0))  # 2*n,hidden
        norm_embedding = self.norm(final)  # N,hidden

        R_Q, R_K = [l(x).view(-1, self.attn_heads, self.d_k).transpose(0, 1) for l, x in
                    zip(self.linear, [norm_embedding, norm_embedding])]  # h,i,k; h,j,k

        # stage 2, convert the gather map
        bs, l, l, _ = relations.shape
        actions = relations[:, :, :, 0]
        fathers = relations[:, :, :, 1]
        if self.only_1:
            actions.fill_(0)
        if self.only_2:
            fathers.fill_(0)
        direction = relations[:, :, :, 2]
        if self.tree_TD:
            idx = actions + (fathers * (fathers - 1) / 2).to(torch.int)
        else:
            idx = actions
        final_idx = idx + direction * self.length  # n,l,l
        return (R_Q, R_K, final_idx.to(torch.long))


class TDTreePositionalEmbedding(nn.Module):
    def __init__(self, args, d_model, heads, inner_TF=False):
        super().__init__()
        self.hidden = d_model
        self.attn_heads = heads
        self.d_k = self.hidden // self.attn_heads
        self.args = args
        self.max_ary = self.args.td_max_ary
        self.max_depth = self.args.td_max_depth
        self.action_size = self.args.action_size
        self.tree_TD = self.args.tree_TD
        self.td_add = args.td_add
        self.td_cat = args.td_cat

        self.only_1, self.only_2 = args.only_1, args.only_2

        if self.tree_TD:
            self.length = self.max_ary * \
                          (self.max_ary - 1) // 2 + self.max_ary + 1
            if self.td_add or self.td_cat:
                self.embedding_a = nn.Embedding(
                    self.max_ary + 1, self.action_size, padding_idx=0)  # for child
                nn.init.xavier_normal_(self.embedding_a.weight)
                self.embedding_b = nn.Embedding(
                    self.max_ary + 1, self.action_size, padding_idx=0)  # for father
                nn.init.xavier_normal_(self.embedding_b.weight)
            if self.td_cat:
                self.td_cat_linear = nn.Linear(2 * self.action_size, self.action_size)
            else:
                self.embedding = nn.Embedding(
                    self.length, self.action_size, padding_idx=0)
                nn.init.xavier_normal_(self.embedding.weight)

        else:
            self.length = self.max_ary + 1
            self.embedding = nn.Embedding(
                self.length, self.action_size, padding_idx=0)
            nn.init.xavier_normal_(self.embedding.weight)

        self.inner_TF = inner_TF
        if self.inner_TF:
            TF = nn.TransformerEncoderLayer(d_model=self.action_size, nhead=4,
                                            dim_feedforward=self.action_size * 4, batch_first=True)
            self.TF = nn.TransformerEncoder(TF, num_layers=2)

            self.PE = nn.Embedding(
                self.max_depth, self.action_size, padding_idx=None)
            nn.init.xavier_normal_(self.PE.weight)
            self.out_linear = nn.Linear(self.action_size, self.hidden)
        else:
            self.tree_cat = nn.Linear(
                self.action_size * self.max_depth, self.hidden)
        self.norm = nn.LayerNorm(self.hidden)
        self.linear = nn.ModuleList(
            [nn.Linear(self.hidden, self.hidden) for _ in range(2)])

    def forward(self, td_actions):
        # td_actions: [batch_size x src_len x depth x 2]
        bs, l, depth, _ = td_actions.shape
        actions = td_actions[:, :, :, 0]
        fathers = td_actions[:, :, :, 1]

        if self.only_1:
            actions.fill_(0)
        if self.only_2:
            fathers.fill_(0)

        if self.tree_TD:
            if self.td_add:
                embedding = self.embedding_a(actions) + self.embedding_b(fathers)  # bs,l,depth,hidden
            elif self.td_cat:
                embedding = self.td_cat_linear(
                    torch.cat([self.embedding_a(actions), self.embedding_b(fathers)], dim=-1))
            else:
                idx = actions + (fathers * (fathers - 1) / 2).to(torch.int)
                embedding = self.embedding(idx)  # bs,l,depth,hidden
        else:
            idx = actions  # bs,l,depth
            embedding = self.embedding(idx)  # bs,l,depth,hidden

        if self.inner_TF:
            PE = self.PE(torch.arange(0, self.max_depth,
                                      device=idx.device))
            embedding = embedding * \
                        math.sqrt(self.action_size) + PE.unsqueeze(0).unsqueeze(0)
            pre_embedding = embedding.view(bs * l, depth, -1)
            idx_ = idx.detach().clone()
            idx_[:, :, 0] = 1
            mask = (idx_ == 0).view(bs * l, depth)  # bs*l,depth
            after_embedding = self.TF(pre_embedding, src_key_padding_mask=mask).view(bs, l, depth, -1). \
                masked_fill(mask.view(
                bs, l, depth).unsqueeze(-1).expand(-1, -1, -1, self.action_size), 0.0)
            pre_final = torch.sum(after_embedding, dim=2)  # bs,l,hidden
            rnn_length = torch.count_nonzero(idx_, dim=-1)  # bs,l
            final = pre_final / rnn_length.unsqueeze(-1)  # bs,l,hidden
            final = self.out_linear(final)  # bs,l,hidden
        else:
            final = self.tree_cat(embedding.view(bs, l, -1))
        norm_embedding = self.norm(final)
        U_Q, U_K = [l(x).view(bs, -1, self.attn_heads, self.d_k).transpose(1, 2) for l, x in
                    zip(self.linear, [norm_embedding, norm_embedding])]
        score = torch.einsum('bhik,bhjk->bhij', U_Q, U_K) / math.sqrt(self.d_k)
        return score


class TransformerEncoder(EncoderBase):
    """
    The Transformer encoder from "Attention is All You Need".
    .. mermaid::
       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O
    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
    Returns:
        (`FloatTensor`, `FloatTensor`):
        * embeddings `[src_len x batch_size x model_dim]`
        * memory_bank `[src_len x batch_size x model_dim]`
    """

    def __init__(self,
                 args,
                 num_layers,
                 d_model=512,
                 heads=8,
                 d_k=64,
                 d_v=64,
                 d_ff=2048,
                 dropout=0.2,
                 max_relative_positions=0,
                 use_neg_dist=True,
                 use_tree_relative_attn=False,
                 tree_rel_vocab_size=0,
                 ggnn_layers_info={},
                 type_vocab_size=0,
                 type_vocab_size2=0,
                 use_td_tree_pos_enc=False,
                 inner_TF=False,
                 use_local_relation=False):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        if isinstance(max_relative_positions, int):
            max_relative_positions = [max_relative_positions] * self.num_layers
        assert len(max_relative_positions) == self.num_layers

        self.layer = nn.ModuleList(
            [TransformerEncoderLayer(args, d_model,
                                     heads,
                                     d_ff,
                                     d_k,
                                     d_v,
                                     dropout,
                                     max_relative_positions=max_relative_positions[i],
                                     use_neg_dist=use_neg_dist,
                                     use_tree_relative_attn=use_tree_relative_attn,
                                     tree_rel_vocab_size=tree_rel_vocab_size,
                                     use_td_tree_pos_enc=use_td_tree_pos_enc,
                                     use_local_relation=use_local_relation)
             for i in range(num_layers)])

        if ggnn_layers_info != {}:
            self.use_ggnn_layers = True
            self.ggnn_layers = nn.ModuleList([
                GGNN(state_dim=d_model,
                     n_edge_types=ggnn_layers_info["n_edge_types"],
                     n_steps=ggnn_layers_info["n_steps_ggnn"])
                for i in range(num_layers)])
            self.ggnn_first = ggnn_layers_info["ggnn_first"]
        else:
            self.use_ggnn_layers = False
        # here
        self.use_td_tree_pos_enc = use_td_tree_pos_enc
        if self.use_td_tree_pos_enc:
            self.td_tree_pos_layer = TDTreePositionalEmbedding(
                args, d_model, heads, inner_TF)
        # HERE
        self.use_local_relation = use_local_relation
        if self.use_local_relation:
            self.local_relation_layer = LocalRelationEmbedding(args, d_model, heads)
            if self.use_td_tree_pos_enc:
                if args.tree_TD and (args.td_add or args.td_cat):
                    self.local_relation_layer.embedding_a.weight = self.td_tree_pos_layer.embedding_a.weight
                    self.local_relation_layer.embedding_b.weight = self.td_tree_pos_layer.embedding_b.weight
                else:
                    self.local_relation_layer.embedding.weight = self.td_tree_pos_layer.embedding.weight
                if args.tree_TD and args.td_cat:
                    self.local_relation_layer.td_cat_linear.weight = self.td_tree_pos_layer.td_cat_linear.weight
                    self.local_relation_layer.td_cat_linear.bias = self.td_tree_pos_layer.td_cat_linear.bias
                # here we tied the two embedding weight

    def count_parameters(self):
        params = list(self.layer.parameters())
        return sum(p.numel() for p in params if p.requires_grad)

    def forward(self, src, lengths=None, rel_matrix=None, src_type=None, src_type2=None, src_tokens=None,
                adj_matrices=None, code_td_paths_rep=None, code_local_relations_rep=None):
        """
        Args:
            src (`FloatTensor`): `[batch_size x src_len x model_dim]`
            lengths (`LongTensor`): length of each sequence `[batch]`
            rel_matrix (`LongTensor`): [batch_size x src_len x src_len]`
            src_type (`LongTensor`): [batch_size x src_len]`
            code_td_paths_rep: [batch_size x src_len x depth x 2]
        Returns:
            (`FloatTensor`):
            * outputs `[batch_size x src_len x model_dim]`
        """
        self._check_args(src, lengths)

        out = src
        mask = None if lengths is None else \
            ~sequence_mask(lengths, out.shape[1]).unsqueeze(1)
        rel_mask = None if rel_matrix is None else \
            get_rel_mask(lengths, out.shape[1])
        # Run the forward pass of every layer of the tranformer.
        representations = []
        attention_scores = []

        # HERE
        if self.use_td_tree_pos_enc:
            td_scores = self.td_tree_pos_layer(code_td_paths_rep)
            # bhij
        else:
            td_scores = None
        # HERE
        if self.use_local_relation:
            LR = self.local_relation_layer(code_local_relations_rep)
        else:
            LR = None

        for i in range(self.num_layers):
            if self.use_ggnn_layers and self.ggnn_first:
                out = self.ggnn_layers[i](out, adj_matrices)
                representations.append(out)
            out, attn_per_head = self.layer[i](
                out, mask, rel_matrix, rel_mask, src_tokens, td_scores, LR)
            representations.append(out)
            attention_scores.append(attn_per_head)
            if self.use_ggnn_layers and not self.ggnn_first:
                out = self.ggnn_layers[i](out, adj_matrices)
                representations.append(out)

        return representations, attention_scores
