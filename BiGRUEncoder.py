import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Pure Bi-GRU Encoder for sequence
# input: 由word2vec得到的单词级表示组成的矩阵
# input_size:[batch, seq_len, features(embedding_size)] seq_len指单词个数
# outputs: (output, hidden)
# output_size: [batch, seq_len, hidden_size] 每个单词对应的时间步都输出一个output
# hidden_size: [batch, hidden_size]
class BiGRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiGRUEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rnn = nn.GRU(input_size, hidden_size//2, bidirectional=True, batch_first=True) #因为双向gru，最后生成的句子级表示由gru头和尾部hidden拼接组成，所以令hidden_size//2
        self.output_size = hidden_size

    def forward(self, inputs):
        self.rnn.flatten_parameters() #适应多gpu训练
        outputs, hidden = self.rnn(inputs) #nn.GRU在bidirectional=True情况下自动输出gru头和尾部的hidden，因此输出的hidden_size为[2, 1, hidden_size//2]
        hidden = hidden.view(inputs.size(0), -1) #将其整理为[1, 2*hidden_size//2] 1指batch_size
        return outputs, hidden


# Bi_GRU + Self_attention 将output视为单词的综合表示，其中蕴含了上下单词间的信息，因此对n*m形矩阵做attention
#（n*m中n是单词的个数，m是ouput的维度，经过linear变为n*1，过softmax得到n*1的注意力权重系数，将权重系数乘到output上，再把结果加到句子级特征表示上）
# input_size: [batch, seq_len, features]
# output_size: [batch, hidden_size]
class BiGRU_selfatt(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiGRU_selfatt, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rnn = nn.GRU(input_size, hidden_size//2, bidirectional=True, batch_first=True)
        self.token_scorer = nn.Linear(hidden_size, 1)
        self.output_size = hidden_size

    def forward(self, inputs):
        self.rnn.flatten_parameters()
        outputs, hidden = self.rnn(inputs)
        hidden = hidden.view(inputs.size(0), -1)
        
        token_score = self.token_scorer(outputs).squeeze(-1) # [batch, seq_len, hidden_size] * [hidden_size, 1] = [batch, seq_len, 1] --> [batch, seq_len]
        token_score = token_score.softmax(dim=-1) #得到注意力权重系数
        
        # [batch, seq_len, hidden_size] * [batch, seqlen, 1] = [batch, seq_len, hidden_size]
        # 对最后一维向量乘上权重系数后求和--->[batch, hidden_size]
        weighted_sum = (outputs * token_score.unsqueeze(-1)).sum(-2)
        return hidden + weighted_sum


# BiGRU + context_attention
# input_size: [batch, seq_len, features]
# output_size: [batch, hidden_size] 计算全文注意力的上下文信息
class BiGRU_Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiGRU_Attention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size // 2, bidirectional=True, batch_first=True)
    
    def attention_net(self, output, hidden):
        hidden = hidden.unsqueeze(2)
        
        # [batch, seq_len, hidden_size] * [batch, hidden_size, 1] = [batch, seq_len, 1] --> [batch, seq_len]
        # 每个单词对应的output中已经含有上下单词的信息，现将句子级的全局信息加入到每个单词的表示中
        attn_weights = torch.bmm(output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        
        # [batch, hidden_size, seqlen] * [batch, seqlen, 1] = [batch, hidden_size, 1] --> [batch, hidden_size]
        context = torch.bmm(output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights # 输出的context即为句子级表示
        
    def forward(self, inputs):
        # input_size : [batch_size, seq_len, features(embedding_dim)]
        # output_size : [batch_size, seq_len, hidden_size]
        # hidden : [D(directions = 2) * num_layers(rnn中可以设置 = 1), N(batch_size), D_out = hidden_size//2]---->[batch_size, hidden_size]
        output, hidden = self.rnn(inputs)
        hidden = hidden.view(inputs.size(0), -1)
        
        context, attention = self.attention_net(output, hidden) #计算context_attention
        return context, attention


# MultiheadAttention for sequence encoder
# Using with pure RNN.
# input: q, k, v
# q : [batch, t_q, hidden] - RNN前后向序列语意表征[batch_size, hidden]
# k : [batch, t_k, hidden] - RNN序列时间步输出[batch, seq_len, hidden]
# v : [batch, t_v, hidden]

# output: y
# y : [batch_size, t_q, hidden_size]

class MultiheadAttention(nn.Module):
    def __init__(self, input_size, num_head, dropout_rate=0):
        super(MultiheadAttention, self).__init__()
        self.input_size = input_size
        self.num_head = num_head
        self.dropout_rate = dropout_rate

        self.Q_proj = nn.Sequential(nn.Linear(self.input_size, self.input_size), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.input_size, self.input_size), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.input_size, self.input_size), nn.ReLU())
        self.O_proj = nn.Sequential(nn.Linear(self.input_size, self.input_size), nn.ReLU())

        self.output_dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, queries, keys, values, mask=None):

        q = self.Q_proj(queries)  # (N, T_q, C)
        k = self.K_proj(keys)  # (N, T_k, C)
        v = self.V_proj(values)  # (N, T_k, C)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)

        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y = self.ScaledDotProductAttention(q, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.O_proj(y)

        return y

    def ScaledDotProductAttention(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.num_head
        out_dim = in_feature * self.num_head
        return x.reshape(batch_size, self.num_head, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.num_head
        return x.reshape(batch_size, seq_len, self.num_head, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.num_head, seq_len, sub_dim)

        
# TextCNN for Sequence words
# inputs: [batch, seq_len, features]
# output: [unigram_num + bigram_num + trigram_num] --> sum(cnn_filters)

class TextCNN_Encoder(nn.Module):
    def __init__(self, input_width, cnn_filters):
        super(TextCNN_Encoder, self).__init__()
        self.input_size = input_width
        self.unigram_filter_num, self.bigram_filter_num, self.trigram_filter_num = cnn_filters
        
        self.edu_unigram_cnn = nn.Conv2d(1, self.unigram_filter_num, (1, input_width), padding=(0, 0))
        self.edu_bigram_cnn = nn.Conv2d(1, self.bigram_filter_num, (2, input_width), padding=(1, 0))
        self.edu_trigram_cnn = nn.Conv2d(1, self.trigram_filter_num, (3, input_width), padding=(2, 0))
        
    def forward(self, cnn_input):
        cnn_input = cnn_input.view(1, 1, cnn_input.size(0), cnn_input.size(1))
        
        # [N, C_in, H, W] --> [N, C_out, H, 1] --> [N, C_out, H]
        # [N, C_out, H] --> [N * C_out]
        unigram_output = F.relu(self.edu_unigram_cnn(cnn_input)).squeeze(-1)
        unigram_feats = F.max_pool1d(unigram_output, kernel_size=unigram_output.size(2)).view(-1)
        
        # [N, C_in, H, W] --> [N, C_out, H+1, 1] --> [N, C_out, H+1],  padding = (1, 0)
        # [N, C_out, H+1] --> [N * C_out]
        bigram_output = F.relu(self.edu_bigram_cnn(cnn_input)).squeeze(-1)
        bigram_feats = F.max_pool1d(bigram_output, kernel_size=bigram_output.size(2)).view(-1)
        
        # [N, C_in, H, W] --> [N, C_out, H+2, 1] --> [N, C_out, H+2],  padding = (2, 0)
        # [N, C_out, H+2] --> [N * C_out]
        trigram_output = F.relu(self.edu_trigram_cnn(cnn_input)).squeeze(-1)
        trigram_feats = F.max_pool1d(trigram_output, kernel_size=trigram_output.size(2)).view(-1)
        
        # [unigram_num + bigram_num + trigram_num]
        cnn_feats = torch.cat([unigram_feats, bigram_feats, trigram_feats], dim=0)
        
        return cnn_feats


if __name__ == '__main__':
    # hyper-para
    input_size = 100
    seq_len = 30
    batch_size = 1
    hidden_size = 256

    inputs = torch.rand(batch_size, seq_len, input_size)

    pure_gru = BiGRUEncoder(input_size, hidden_size)
    outputs, hidden = pure_gru(inputs)
    print(outputs.shape) #torch.Size([1, 30, 256])
    print(hidden.shape) #torch.Size([1, 256])
    
    multi_attn = MultiheadAttention(hidden_size, 8)
    # gru_selfattn = BiGRU_selfatt(input_size, hidden_size)
    # gru_attention = BiGRU_Attention(input_size, hidden_size)
    # text_cnn = TextCNN_Encoder(input_size, [60, 30, 10])

    

    q = hidden.view(1, 1, -1)
    k, v = outputs, outputs

    output = multi_attn(q, k, v)
    print(output.shape)



    # output_selfattn = gru_selfattn(inputs)
    # print(output_selfattn.shape)

    # output_attention, attention = gru_attention(inputs)
    # print(output_attention.shape)

    # textcnn_output = text_cnn(inputs.view(inputs.size(1), -1))
    # print(textcnn_output.shape)
