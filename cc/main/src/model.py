#!/usr/bin/env python3
# Copyright (c) 2019 OpenAI, HugginFace Inc. team. and TaeHwan Jung
# Copyright (c) Facebook, Inc. and its affiliates.
# ----------------------------------------------------------------------------
# MIT LICENSE
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------
"""
    Transformer model is adapted from: https://github.com/graykode/gpt-2-Pytorch
        (Commit: 46ae886391a94c6683be438269252c4afd5ba762)
    Original Paper and repository here: https://github.com/openai/gpt-2
"""
# added Attention modifications

import copy
import math

import torch
import torch.nn as nn

from src.utils.encodings_utils import PositionalEncodings

INF = 1e10


def gelu(x):
    return (
            0.5
            * x
            * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, std_eps=1e-6):
        """Construct a layernorm module in the TF style.
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.std_eps = std_eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).std(-1, keepdim=True)
        x = (x - u) / (s + self.std_eps)
        return self.weight * x + self.bias


class Attention(nn.Module):
    """
        Radford et al. 2019. Language Models are Unsupervised Multitask Learners.
    """

    def __init__(
            self, nx, n_ctx, n_head, scale=False, dropout=None,
    ):
        super(Attention, self).__init__()
        n_state = nx
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % n_head == 0
        self.register_buffer(
            "bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx)
        )
        self.n_head = n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = nn.Linear(nx, n_state * 3)
        self.c_proj = nn.Linear(nx, n_state)
        self.dropout = nn.Dropout(dropout)

    def _attn(self, q, k, v):
        raise NotImplementedError

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def get_q_k_v(self, x):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        return query, key, value

    def self_attention(self, query, key, value):
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a

    def forward(self, x):
        query, key, value = self.get_q_k_v(x)
        return self.self_attention(query, key, value)


class RelativeAttention(Attention):
    def __init__(
            self, nx, n_ctx, n_head, scale=False, dropout=None, additive=False,
            use_tree=False, use_seq=False, rel_vocab_size=None, rel_kmax=None, args=None
    ):
        super(RelativeAttention, self).__init__(nx, n_ctx, n_head, scale, dropout)
        self.additive = additive
        self.use_seq = use_seq
        self.use_tree = use_tree

        if use_tree:
            self.rel_weights = nn.Embedding(rel_vocab_size, n_head)

        if use_seq:
            # sequential relative attention
            n_state = nx
            self.rel_kmax = min(rel_kmax, n_ctx) if rel_kmax is not None else n_ctx
            x = torch.LongTensor(range(n_ctx))
            bias = torch.clamp(-x[:, None] + x[None, :], min=-self.rel_kmax, max=self.rel_kmax) + self.rel_kmax
            self.register_buffer(
                "rel_ids", bias
            )
            self.rel_keys = nn.Embedding(2 * self.rel_kmax + 1, n_state // n_head)
            self.rel_values = nn.Embedding(2 * self.rel_kmax + 1, n_state // n_head)
        # HERE
        self.use_td_tree_pos_enc = args.use_td_tree_pos_enc
        self.use_local_relation = args.local_relation
        self.sqrt = args.sqrt
        self.only_local_relation = args.only_local_relation

    def matmul_with_relative_representations(self, q, rel, transpose_rel=False):
        # sequential relative attention helper function
        # q: [b, h, n, dh] -> [n, b, h, dh] -> [n, b*h, dh]
        # rel: [n, n, dh]
        # return: [b, h, n, n]
        nb, nh, nt, _ = q.size()
        q = q.permute(2, 0, 1, 3).contiguous()
        q = q.reshape(q.size(0), nb * nh, q.size(-1))
        if not transpose_rel:
            rel = rel.permute(0, 2, 1)
        x = torch.matmul(q, rel)
        x = x.reshape(nt, nb, nh, -1)
        x = x.permute(1, 2, 0, 3).contiguous()
        return x

    def _attn(self, q, k, v, rel=None, tds=None, lr=None):
        w = torch.matmul(q, k)
        nd, ns = w.size(-2), w.size(-1)

        if self.use_seq:
            rel_k = self.rel_keys(self.rel_ids[ns - nd: ns, :ns])
            w = w + self.matmul_with_relative_representations(q, rel_k)

        if self.scale:
            w = w / math.sqrt(v.size(-1))

        b = self.bias[:, :, ns - nd: ns, :ns]

        if self.use_tree:
            assert rel is not None
            # tree relative attention mask
            # additive is better than multiplicative
            w = w + rel * b if self.additive else w * (rel * b)

        # HERE
        if self.use_td_tree_pos_enc and not self.only_local_relation:
            w = w + tds

        if self.use_local_relation:
            (LR_Q, LR_K, LR_map) = lr
            # LR_Q:[hnk] LR_K[hnk],LR_map[bij]
            LR_map = LR_map.unsqueeze(1).expand(-1, LR_Q.shape[0], -1, -1)
            LR_1 = torch.einsum('bhik,bhnk->bhin', q,
                                LR_K.unsqueeze(0).expand(q.shape[0], -1, -1,
                                                         -1)) / math.sqrt(
                v.size(-1))  # query already div self.d_k before!
            LR_1 = torch.gather(LR_1, -1, LR_map)  # bhin->bhij

            LR_2 = torch.einsum('bhnk,bhkj->bhnj', LR_Q.unsqueeze(0).expand(k.shape[0], -1, -1, -1),
                                k) / math.sqrt(v.size(-1))
            LR_2 = torch.gather(LR_2, -2, LR_map)  # bhnj->bhij
            w = w + LR_1 + LR_2

        if self.use_td_tree_pos_enc or self.use_local_relation:
            if self.sqrt == -1:
                raise Exception('Not Set Sqrt!')
            w = w / math.sqrt(self.sqrt)

        w = w * b - INF * (1 - b)

        w_normed = nn.Softmax(dim=-1)(w)  # calc attention scores
        if self.dropout is not None:
            w_normed = self.dropout(w_normed)

        ret = torch.matmul(w_normed, v)

        if self.use_seq:
            rel_v = self.rel_values(self.rel_ids[ns - nd: ns, :ns])
            ret += self.matmul_with_relative_representations(w_normed, rel_v, transpose_rel=True)

        return ret

    def self_attention(self, query, key, value, rel, tds, lr):
        a = self._attn(query, key, value, rel, tds, lr)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a

    def forward(self, x, rel=None, tds=None, lr=None):
        query, key, value = self.get_q_k_v(x)
        if self.use_tree:
            rel = self.rel_weights(rel)
            rel = rel.permute(0, 3, 1, 2)
        return self.self_attention(query, key, value, rel, tds, lr)


class MLP(nn.Module):
    def __init__(self, n_state, n_embd):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(n_embd, n_state)
        self.c_proj = nn.Linear(n_state, n_embd)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2


class Block(nn.Module):
    def __init__(
            self,
            args,
            n_ctx,
            n_head,
            n_embd,
            layer_norm_epsilon,
            scale=False,
            rel_vocab_size=None,  # tree relative attention
            use_tree=False,  # tree relative attention
            use_seq=False,  # sequential relative attention
            rel_kmax=None,  # sequential relative attention
            additive=False,  # tree relative attention
            residual_dropout=0.1,
            atten_dropout=0.1,
    ):
        super(Block, self).__init__()
        self.ln_1 = LayerNorm(n_embd, std_eps=layer_norm_epsilon)
        self.args = args
        args = [n_embd, n_ctx, n_head, scale, atten_dropout]

        self.attn = RelativeAttention(
            *args, rel_kmax=rel_kmax, rel_vocab_size=rel_vocab_size,
            additive=additive,
            use_tree=use_tree,
            use_seq=use_seq,
            args=self.args

        )

        self.ln_2 = LayerNorm(n_embd, std_eps=layer_norm_epsilon)
        self.mlp = MLP(4 * n_embd, n_embd)
        self.drop1 = nn.Dropout(residual_dropout)
        self.drop2 = nn.Dropout(residual_dropout)

    def forward(self, x, **att_kwargs):
        a = self.drop1(self.attn(self.ln_1(x), **att_kwargs))
        x = x + a
        m = self.drop2(self.mlp(self.ln_2(x)))
        x = x + m
        return x


class GPT2Model(nn.Module):
    def __init__(
            self, args,
            types_vocab_size,
            values_vocab_size,
            n_layer,
            n_embd,
            n_ctx,
            n_head,
            layer_norm_epsilon,
            rel_vocab_size=None,
            use_tree=False,
            use_seq=False,
            rel_kmax=None,
            root_paths=False,
            tree_pos_enc_meta=None,
            use_sin_pos_enc=False,
            use_pos_embed=False,
            additive=False,
            residual_dropout=0.1,
            embed_dropout=0.1,
            atten_dropout=0.1
    ):
        super(GPT2Model, self).__init__()
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.wte_types = nn.Embedding(types_vocab_size, n_embd)
        self.wte_values = nn.Embedding(values_vocab_size, n_embd)

        self.use_tree = use_tree
        self.use_seq = use_seq

        self.positional_encodings = PositionalEncodings(
            n_ctx,
            n_embd,
            use_sin_pos_enc,
            use_pos_embed,
            tree_pos_enc_meta,
            None,
            embed_dropout
        )

        block = Block(
            args,
            n_ctx,
            n_head,
            n_embd,
            layer_norm_epsilon,
            scale=True,
            rel_vocab_size=rel_vocab_size,
            use_tree=use_tree,
            use_seq=use_seq,
            rel_kmax=rel_kmax,
            additive=additive,
            residual_dropout=residual_dropout,
            atten_dropout=atten_dropout,
        )
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(n_layer)])
        self.ln_f = LayerNorm(n_embd, std_eps=layer_norm_epsilon)

        # HERE:
        self.use_td_tree_pos_enc = args.use_td_tree_pos_enc
        if self.use_td_tree_pos_enc:
            self.td_tree_pos_layer = TDTreePositionalEmbedding(
                args, n_embd, n_head, False)

        self.use_local_relation = args.local_relation
        if self.use_local_relation:
            self.local_relation_layer = LocalRelationEmbedding(args, n_embd, n_head)
            if self.use_td_tree_pos_enc:
                if args.tree_TD and (args.td_add or args.td_cat):
                    self.local_relation_layer.embedding_a.weight = self.td_tree_pos_layer.embedding_a.weight
                    self.local_relation_layer.embedding_b.weight = self.td_tree_pos_layer.embedding_b.weight
                else:
                    self.local_relation_layer.embedding.weight = self.td_tree_pos_layer.embedding.weight
                if args.tree_TD and args.td_cat:
                    self.local_relation_layer.td_cat_linear.weight = self.td_tree_pos_layer.td_cat_linear.weight
                    self.local_relation_layer.td_cat_linear.bias = self.td_tree_pos_layer.td_cat_linear.bias

    def forward(self, types, values, rel=None, positions=None, paths=None, tds=None, lr=None):
        # HERE
        if self.use_td_tree_pos_enc:
            td_scores = self.td_tree_pos_layer(tds)
        else:
            td_scores = None
        if self.use_local_relation:
            LR = self.local_relation_layer(lr)
        else:
            LR = None
            # HERE

        #  prepare
        input_shape = values.size()
        values = values.view(-1, values.size(-1))
        types_embeds = 0.
        if len(types) > 0:  # if we do not using types data (Text mode)
            types = types.view(-1, types.size(-1))
            types_embeds = self.wte_types(types)
        values_embeds = self.wte_values(values)
        inputs_embeds = types_embeds + values_embeds

        hidden_states = inputs_embeds

        # augment with positinal information
        hidden_states = self.positional_encodings(hidden_states, paths=paths, positions=positions)

        att_kwargs = {}  # attention modifications: additional inputs
        if self.use_tree:
            att_kwargs.update({"rel": rel})

        # HERE:
        if self.use_td_tree_pos_enc:
            att_kwargs.update({"tds": td_scores})
        if self.use_local_relation:
            att_kwargs.update({"lr": LR})

        # apply Transformer block
        for block in self.h:
            hidden_states = block(hidden_states, **att_kwargs)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape)


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
                pos = self.embedding(torch.arange(0, self.length, device=relations.device))  # n,dim
        else:
            pos = self.embedding(torch.arange(0, self.length, device=relations.device))  # n,dim
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


class GPT2LMHead(nn.Module):
    def __init__(self, model_embeddings_weights, n_embd):
        super(GPT2LMHead, self).__init__()
        self.n_embd = n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        lm_logits = self.decoder(hidden_state)
        return lm_logits


class TransformerModel(nn.Module):
    def __init__(
            self, args, types_vocab_size,
            values_vocab_size, n_layer, n_embd,
            n_ctx, n_head, layer_norm_epsilon, **kwargs
    ):
        super(TransformerModel, self).__init__()
        self.transformer = GPT2Model(args, types_vocab_size, values_vocab_size,
                                     n_layer, n_embd, n_ctx, n_head, layer_norm_epsilon, **kwargs)

        self.types_head = GPT2LMHead(self.transformer.wte_types.weight, n_embd)
        self.values_head = GPT2LMHead(self.transformer.wte_values.weight, n_embd)
        self.args = args

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, rel=None, positions=None, tds=None, lr=None, paths=None):
        hidden_states = self.transformer(x["types"], x["values"], rel, positions, paths, tds, lr)
        types = self.types_head(hidden_states) if len(x["types"]) > 0 else []
        values = self.values_head(hidden_states)
        return types, values


class MaskedLoss(nn.CrossEntropyLoss):
    def __init__(self, pad_idx, oov_idx, empty_idx):
        super(MaskedLoss, self).__init__()
        self.pad_idx = pad_idx
        self.oov_idx = oov_idx
        self.empty_idx = empty_idx

    def forward(self, *inputs, return_len=False):
        y_pred, y, ext = inputs
        assert len(y.size()) == 2
        # we do not calculate loss on the history part of the sequence 
        # from ast splitting
        ext_r = ext.unsqueeze(-1).repeat(1, y.size(-1))
        ext_ids = torch.arange(y.size(-1), device=ext_r.device).view(1, -1).repeat(*(y.size()[:-1] + (1,)))
        where = ext_ids >= ext_r  # skip the memory from the previous code snippet
        where &= y != self.pad_idx  # calc loss only on known tokens and filter padding and empty values
        where &= y != self.oov_idx
        where &= y != self.empty_idx
        where = where.view(-1)

        y_pred = y_pred.view(-1, y_pred.size(-1))
        y = y.view(-1)
        if where.sum() == 0:
            return y_pred.new_ones(1, requires_grad=True) * 1e-8  # in case the seq is empty
        loss = super(MaskedLoss, self).forward(y_pred[where], y[where])
        return loss
