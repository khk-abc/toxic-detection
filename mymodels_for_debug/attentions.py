import math

import torch.nn as nn
import torch
from transformers.modeling_utils import find_pruneable_heads_and_indices, prune_linear_layer

import numpy as np



def my_create_extended_attention_mask_for_decoder(attention_mask):

    if attention_mask.dim()==1:
        lenth = attention_mask.shape[0]
        til = torch.tril(torch.ones((lenth, lenth)), diagonal=0)
        attention_mask = attention_mask.reshape(1, lenth)
        attention_mask=(attention_mask* attention_mask.permute(1,0))
    else:
        bz,lenth = attention_mask.shape
        til = torch.tril(torch.ones((bz,lenth, lenth)), diagonal=0)
        attention_mask = attention_mask.unsqueeze(1).repeat(1, lenth, 1)

        attention_mask=(attention_mask* attention_mask.permute(0,2,1))

    return til*attention_mask



def create_cross_pad_mask(query_mask, kv_mask,query_length=None,kv_length=None,role_query=True):
    # if role_query:
    #     key_value_attention_mask = key_value_attention_mask.unsqueeze(-1).repeat(1, 1, kv_length)
    # else:
    #     key_value_attention_mask = key_value_attention_mask.unsqueeze(1).repeat(1, query_length, 1)

    # print(query_mask.shape)
    # print(kv_mask.shape)
    assert query_mask.dim()==2 and kv_mask.dim()==2
    query_mask=query_mask.unsqueeze(-1).repeat(1,1,kv_length)
    kv_mask=kv_mask.unsqueeze(1).repeat(1,query_length,1)

    cross_attention_mask=query_mask*kv_mask

    return cross_attention_mask

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, query_mask=None, kv_mask=None,role_query=True):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads


        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)


        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            query_mask,
            kv_mask,
            role_query
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn



class BertResidualOutput(nn.Module):
    def __init__(self, hidden_size,layer_norm_eps=1e-10,hidden_dropout_prob=0.3):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1, output_attention=True):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values,query_mask=None, kv_mask=None,role_query=True):

        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # print('attenc scores:', scores)
        # print('attenc scores shape:', scores.shape)
        # print('query:',queries)
        # print('key:',keys)

        # print('pad_mask:',pad_mask)
        if kv_mask is not None and query_mask is not None:
            # bz, length
            pad_mask = create_cross_pad_mask(query_mask=query_mask,kv_mask=kv_mask,
                                             query_length=L,kv_length=S,role_query=role_query)

            pad_mask = pad_mask.unsqueeze(1).repeat(1, H, 1, 1)

            # print('pad mask: ', pad_mask[:,0].sum(dim=-1))

            scores.masked_fill_(pad_mask.to(scores.device) == 0, -np.inf)

            A = torch.softmax(scale * scores, dim=-1)
            A = A.masked_fill(pad_mask.to(scores.device) == 0, 0.)

        else:
            A = torch.softmax(scale * scores, dim=-1)

        A = self.dropout(A)

        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class BertSelfAttention(nn.Module):
    def __init__(self, config=None,hidden_size=768,num_attention_heads=8,attention_probs_dropout_prob=0.3):
        super().__init__()

        if config is not None:
            if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
                raise ValueError(
                    "The hidden size (%d) is not a multiple of the number of attention "
                    "heads (%d)" % (config.hidden_size, config.num_attention_heads)
                )
            hidden_size = config.hidden_size
            num_attention_heads = config.num_attention_heads
            attention_probs_dropout_prob = config.attention_probs_dropout_prob

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)

        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=True,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

            # raw version
            # attention_scores = attention_scores + attention_mask

            # modified version
            attention_mask = attention_mask.repeat(1, attention_scores.shape[1], 1, 1)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -np.inf)


        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if attention_mask is not None:
            attention_probs = attention_probs.masked_fill(attention_mask == 0, 0.)  # 将nan值填充为0

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config=None,hidden_size=768,layer_norm_eps=1e-12,hidden_dropout_prob=0.3):
        super().__init__()
        if config is not None:
            hidden_size = config.hidden_size
            layer_norm_eps = config.layer_norm_eps
            hidden_dropout_prob = config.hidden_dropout_prob

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config=None,hidden_size=768,num_attention_heads=8,
                 attention_probs_dropout_prob=0.3,layer_norm_eps=1e-12,
                 hidden_dropout_prob=0.3):
        super().__init__()
        if config is not None:
            hidden_size = config.hidden_size
            num_attention_heads = config.num_attention_heads
            attention_probs_dropout_prob = config.attention_probs_dropout_prob
            layer_norm_eps=config.layer_norm_eps
            hidden_dropout_prob=config.hidden_dropout_prob


        self.self = BertSelfAttention(hidden_size=hidden_size,
                                      num_attention_heads=num_attention_heads,
                                      attention_probs_dropout_prob=attention_probs_dropout_prob)
        self.output = BertSelfOutput(hidden_size=hidden_size,
                                     layer_norm_eps=layer_norm_eps,hidden_dropout_prob=hidden_dropout_prob)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=True,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs
