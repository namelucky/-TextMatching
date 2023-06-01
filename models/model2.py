from fastNLP.modules import ConditionalRandomField, allowed_transitions
from modules.transformer import TransformerEncoder
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d,self).__init__()
    def forward(self, x):
        return torch.max_pool1d(x,kernel_size=x.shape[2])
class AvgPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        # x: batch, seq_len, dim
        # mask: batch, seq_len, 1
        t=torch.sum(mask.int(), dim=1)#为了反正一行的元素全部为空
        a=torch.full_like(t,1e10)
        t2=torch.where(t<=0,a,t)
        return torch.sum(x.masked_fill_(~mask.bool(), 0), dim=1) /t2

class Pooling(nn.Module):
    def forward(self, x, mask):
        return x.masked_fill_(~mask.bool(),-100).max(dim=1)[0]
class LMATCH(nn.Module):
    def __init__(self, vocab_size, num_layers, d_model, n_head, feedforward_dim, dropout, embedding_pretrained,
                 after_norm=True, attn_type='adatrans',
                 fc_dropout=0.3, pos_embed=None, scale=False, dropout_attn=None):
        """

        :param tag_vocab: fastNLP Vocabulary
        :param embed: fastNLP TokenEmbedding
        :param num_layers: number of self-attention layers
        :param d_model: input size
        :param n_head: number of head
        :param feedforward_dim: the dimension of ffn
        :param dropout: dropout in self-attention
        :param after_norm: normalization place
        :param attn_type: adatrans, naive
        :param rel_pos_embed: position embedding的类型，支持sin, fix, None. relative时可为None
        :param bi_embed: Used in Chinese scenerio
        :param fc_dropout: dropout rate before the fc layer
        """
        super().__init__()
        self.embedding_dim = 300
        self.vocab_size = vocab_size
        # 词向量矩阵
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim, 0)
        # 使用预训练词向量初始化
        if embedding_pretrained:
            self.embedding.weight.data.copy_(torch.from_numpy(np.asarray(embedding_pretrained)))

        self.in_fc = nn.Linear(self.embedding_dim, d_model)

        self.transformer = TransformerEncoder(num_layers, d_model, n_head, feedforward_dim, dropout,
                                              after_norm=after_norm, attn_type=attn_type,
                                              scale=scale, dropout_attn=dropout_attn,
                                              pos_embed=pos_embed)
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc2=nn.Linear(256,100)#降维用的
        self.maxPool=nn.AdaptiveMaxPool1d(1)
        self.linear_stack = nn.Sequential(
            nn.BatchNorm1d(16448),#17408
            nn.Linear(16448, 300),#19456
            nn.ReLU(),
            nn.BatchNorm1d(300),
            nn.Linear(300, 300),  #
            nn.ReLU(),
            nn.BatchNorm1d(300),
            nn.Linear(300, 1),
            nn.Sigmoid()
        )
        self.pooling = Pooling()
        self.avg_pooling = AvgPooling()
        self.fc = nn.Linear(12800, d_model)
        self.out_fc = nn.Linear(d_model, 1)

    def attn_align(self, x, y, y_mask=None):
        # 与Self-Attn非常相似，区别在于x=Q, y=K, y=V

        # x: batch, max_len1, dim
        # y: batch, max_len2, dim
        # y_mask: batch, max_len2

        # x@y^T :计算x[i]与y[j]的相似度
        # attn_score: batch, max_len1, max_len2
        attn_score = x.matmul(y.transpose(1, 2))

        # 对attn_score进行mask
        if y_mask != None:
            #  y_mask:(batch, 1, max_len2),用于后续的board casting
            # y_mask = torch.unsqueeze(y_mask, dim=1)
            y_mask = y_mask.transpose(1, 2)
            # 将mask=0的位置使用-1e9来进行替换
            # print(attn_score.size(), y_mask.size())
            attn_score = attn_score.masked_fill(y_mask.int() == 0, -1e9)

        # 进行softmax操作
        # attn_score: batch, max_len1, max_len2
        attn_score = torch.softmax(attn_score, dim=1)

        # 注意力权重与y进行加权: attn_score@y
        # z: batch, max_len1, dim
        z = attn_score.matmul(y)

        return z

    def _forward(self, titles,ans,lattice_matrix=None):
    #ans表示答案，chars表示title
        # 生成mask,0的位置就是用来mask
        title = titles[0]
        title1 = titles[1]

        ans1 = ans[0]
        ans2 = ans[1]

        maskt1 = title.ne(1)#([48, 20])
        maskt2 = title1.ne(1)

        maska1=ans1.ne(1)
        maska2=ans2.ne(1)

        title = self.embedding(title)
        title1=self.embedding(title1)

        ans1 = self.embedding(ans1)
        ans2 = self.embedding(ans2)

        title=self.in_fc(title)
        title1=self.in_fc(title1)

        ans1 = self.in_fc(ans1)
        ans2 = self.in_fc(ans2)

        print('maskt1{}'.format(maskt1.shape))#([48, 32])

        title = self.transformer(title, maskt1, None)
        title1=self.transformer(title1,maskt2,None)

        ans1 = self.transformer(ans1, maska1, None)
        ans2 = self.transformer(ans2, maska2, None)

        title=self.fc_dropout(title)
        title1=self.fc_dropout(title1)

        ans1 = self.fc_dropout(ans1)
        ans2 = self.fc_dropout(ans2)

        ans1=ans1.clone()
        ans2 = ans2.clone()

        a1_aligned = self.attn_align(ans1, ans2)
        a2_aligned = self.attn_align(ans2, ans1)

        title=title.clone()
        title1=title1.clone()

        t1_aligned=self.attn_align(title,title1)
        a_max = self.maxPool(t1_aligned)
        a_max = torch.squeeze(a_max)
        a_pool = t1_aligned.mean(dim=-1)
        t2_aligned = self.attn_align(title1, title)


        Tx1_combined = torch.cat([title, t1_aligned, title - t1_aligned, title * t1_aligned], dim=-1)#([48, 20, 1024])
        print('Tx1_combined{}'.format(Tx1_combined.shape))#([48, 32, 2048])
        Tx2_combined = torch.cat([title1, t2_aligned, title1 - t2_aligned, title1 * t2_aligned], dim=-1)
        Ax1_combined = torch.cat([ans1, a1_aligned, ans1 - a1_aligned, ans1 * a1_aligned], dim=-1)
        Ax2_combined = torch.cat([ans2, a2_aligned, ans2 - a2_aligned, ans2 * a2_aligned], dim=-1)

        maskt1 = maskt1.unsqueeze(maskt1.dim())
        maskt2 = maskt2.unsqueeze(maskt2.dim())
        maska1 = maska1.unsqueeze(maska1.dim())
        maska2 = maska2.unsqueeze(maska2.dim())

        Ta_max = self.pooling(Tx1_combined.clone(), maskt1)
        Tb_max = self.pooling(Tx2_combined.clone(), maskt2)
        Ta_avg = self.avg_pooling(Tx1_combined.clone(), maskt1)
        Tb_avg = self.avg_pooling(Tx2_combined.clone(), maskt2)

        Aa_max = self.pooling(Ax1_combined.clone(), maska1)
        Ab_max = self.pooling(Ax2_combined.clone(), maska2)
        Aa_avg = self.avg_pooling(Ax1_combined.clone(), maska1)
        Ab_avg = self.avg_pooling(Ax2_combined.clone(), maska2)

        s_concat = torch.cat( [Ta_max, Tb_max, Ta_avg, Tb_avg,
         Aa_max, Ab_max, Aa_avg, Ab_avg,a_max, a_pool], dim=1)

        s_concat = self.fc_dropout(s_concat)
        out = self.linear_stack(s_concat)
        return out

    def forward(self, titles ,ans, lattice_matrix=None):

        return self._forward(titles,ans, lattice_matrix)

    def predict(self, titles,ans, lattice_matrix=None):
        output = self._forward(titles,ans, lattice_matrix=lattice_matrix)
        _, pre = output.max(dim=1)
        return pre
