'''
'''

from torchtext.data.utils import get_tokenizer
import torch
from torch.utils.data import Dataset
from gensim.models import word2vec

from tqdm import tqdm
import pandas as pd
from torchtext.vocab import build_vocab_from_iterator
import numpy as np
from gensim.models import KeyedVectors
import re

from nltk.stem import WordNetLemmatizer
class SNLIDataset(Dataset):
    '''
    注：词汇表的创建只能通过训练集来创建
    '''

    def __init__(self, mode, length=24,mask=False):
        # 获取分词器
        self.tokenizer = get_tokenizer('basic_english')
        # 'train', 'test', 'dev'
        self.mode = mode
        # 是否需要返回输入word的mask
        self.mask = mask
        # 标签-数值映射
        # 获取分词后的会话文本和标签
        self.s1_list, self.s2_list, self.label_list,self.ans_list = self.load_raw_data(mode)
        # S1最大长度
        self.S1_MAX_LEN = 32
        # S2最大长度
        self.S2_MAX_LEN = 32
        self.Ans_MAX_LEN=100
        # 文本词汇表
        self.train_text_vocab = self.build_vocab()
        # 文本词汇映射
        self.train_text_vocab_dict = self.train_text_vocab.get_stoi()
        # <pad>,<unk>的idx
        self.pad_idx = self.train_text_vocab_dict.get('<pad>')
        self.unk_idx = self.train_text_vocab_dict.get('<unk>')
        self.dim=300## 词向量维度
        self.lattice_dict = {}


    def word_lemmatizer(self,list_words):
        wordnet_lemmatizer = WordNetLemmatizer()
        list_words_lemmatizer = []
        for word in list_words:
            list_words_lemmatizer.append(wordnet_lemmatizer.lemmatize(word))
        return list_words_lemmatizer


    def get_file_path(self, mode):
        '''获取数据路径'''
        if mode == 'train':
            return r'data/quora-a/train.tsv'
        elif mode == 'dev':
            return r'data/quora-a/dev.tsv'
        elif mode == 'test':
            return r'data/quora-a/test.tsv'
        else:
            raise ValueError('The value of mode can only be: train, dev or test!')

    def load_raw_data(self, mode):
        '''
        加载原始文本数据集,只完成分词的部分
        :param mode:

        :return:
        '''

        df = pd.read_csv(self.get_file_path(mode), sep='\t', keep_default_na=False,header=None,index_col=False)
        df.columns = ["s1", "s2", "label", "ans"]
        df = self.preprocessing(df)

        s1_list, s2_list, label_list,ans_list= [], [], [],[]
        text=[]

        for data_pair in tqdm(list(zip(df['s1'], df['s2'], df['label'],df['ans']))):
            s1, s2, label,ans = data_pair
            split_s1=self.word_cut(s1)
            split_s2=self.word_cut(s2)
            split_ans=self.word_cut(ans)


            # 过滤掉标注不一致标签
            # 进行分词，添加分词结果
            t1=self.word_lemmatizer(split_s1)
            t1=[x for x in t1 if x != 'nan']
            t2 = self.word_lemmatizer(split_s2)
            t2 = [x for x in t2 if x != 'nan']
            a2=self.word_lemmatizer(split_ans)
            a2 = [x for x in a2 if x != 'nan']

            s1_list.append(t1)
            s2_list.append(t2)
            label_list.append(label)
            ans_list.append(a2)

        text.extend(s1_list)
        text.extend(s2_list)
        text.extend(ans_list)

        self.pretrain_embedding(text,'./vec/w2v_stack.txt')

        return s1_list, s2_list, label_list,ans_list
    def word_cut(self,text):
        return self.tokenizer(text)

    def build_vocab(self):
        '''
        创建词汇表
        :return:
        '''
        # 只能使用训练集的文本来创建词汇表
        if self.mode == 'train':
            s1_list, s2_list,ans_list = self.s1_list, self.s2_list,self.ans_list
        else:
            s1_list, s2_list, _, ans_list= self.load_raw_data(mode='train')

        # -----------------
        # 创建文本词汇表
        # -----------------
        # 合并s1_list, s2_list
        s1_list.extend(s2_list)
        s1_list.extend(ans_list)

        # # 转为迭代器
        iter_text = iter(s1_list)
        # 创建词汇表
        train_text_vocab = build_vocab_from_iterator(iter_text, specials=["<unk>", "<pad>"])
        # 设置<unk>,用于oov的情况
        train_text_vocab.set_default_index(train_text_vocab["<unk>"])

        print('Finish building text vocab...')
        torch.save(train_text_vocab, 'train_text_vocab.pth')
        train_text_vocab=torch.load('train_text_vocab.pth')
        print('Finish loading text vocab...')

        return train_text_vocab

    def __len__(self):
        # 获取数据集数目
        return len(self.label_list)

    def __getitem__(self, idx):
        '''
        对文本进行数字化的过程
        :param idx:
        :return:
        '''
        s1, s2, label ,ans= self.s1_list[idx], self.s2_list[idx], self.label_list[idx],self.ans_list[idx]

        lattice1=None
        lattice2=None

        # 文本数字化
        s1_token = self.train_text_vocab(s1)
        s2_token = self.train_text_vocab(s2)
        ans_token=self.train_text_vocab(ans)


        # s1,s2的mask
        s1_mask = [1] * len(s1_token)
        s2_mask = [1] * len(s2_token)
        ans_mask=[1]*len(ans_token)

        # padding填充
        s1_padding = [self.pad_idx] * (self.S1_MAX_LEN - len(s1_token))
        s1_token.extend(s1_padding)

        s2_padding = [self.pad_idx] * (self.S2_MAX_LEN - len(s2_token))
        s2_token.extend(s2_padding)

        ans_padding=[self.pad_idx]*(self.Ans_MAX_LEN-len(ans_token))
        ans_token.extend(ans_padding)

        s1_mask_padding = [0] * (self.S1_MAX_LEN - len(s1_mask))
        s1_mask.extend(s1_mask_padding)

        s2_mask_padding = [0] * (self.S2_MAX_LEN - len(s2_mask))
        s2_mask.extend(s2_mask_padding)

        ans_mask_padding=[0]*(self.Ans_MAX_LEN-len(ans_mask))
        ans_mask.extend(ans_mask_padding)

        pad_token=[self.pad_idx]*25


        # 长度截断
        s1_token = s1_token[:self.S1_MAX_LEN]
        s2_token = s2_token[:self.S2_MAX_LEN]
        ans_token = ans_token[:self.Ans_MAX_LEN]

        s1_mask = s1_mask[:self.S1_MAX_LEN]
        s2_mask = s2_mask[:self.S2_MAX_LEN]
        ans_mask=ans_mask[:self.Ans_MAX_LEN]


        # 转tensor
        s1_token = torch.tensor(s1_token)
        s2_token = torch.tensor(s2_token)
        pad_token=torch.tensor(pad_token)
        # print('pad_token{}'.format(pad_token.shape))
        # print('s1_token{}'.format(s1_token.shape))
        title_token=torch.cat([s1_token,s2_token],0)
        title_token = torch.cat([title_token, pad_token], 0)
        # print('title_token{}'.format(title_token.shape))


        ans_token=torch.tensor(ans_token)

        ans_token1=torch.cat([s1_token,ans_token],0)
        ans_token2=torch.cat([s2_token,ans_token],0)

        # s1_mask = torch.tensor(s1_mask)
        # s2_mask = torch.tensor(s2_mask)
        # ans_mask=torch.tensor(ans_mask)

        label = torch.tensor(label)

        if self.mask:
            return s1_token, s2_token, s1_mask, s2_mask, label,ans_token,ans_mask
        else:
            return ans_token1,ans_token2,label

    def pretrain_embedding(self,texts,f_name,binary = False):
        model = word2vec.Word2Vec(sentences=texts,size=300,window=2,min_count=1,workers=2)
        model.wv.save_word2vec_format(fname=f_name, binary=binary, fvocab=None)
    def get_pretrain_embedding(self):
        train_words_list = self.train_text_vocab.get_itos()
        embedding_matrix = 1 * np.random.randn(len(train_words_list) + 1, self.dim)
        print('keras的embedding矩阵维度为[{},{}]'.format(len(train_words_list) + 1, self.dim))
        embedding_matrix[0] = np.random.randn(self.dim)
        w2v_path = 'vec/w2v_stack.txt'
        w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=False)
        count = 0
        nocount = 0
        index = 0

        vectors=[]

        for word in train_words_list:  # index是从1开始的
            if word in w2v_model.vocab:
                embedding_matrix[index] = w2v_model.word_vec(word)
                vectors.append(w2v_model.word_vec(word))
                count = count + 1
            else:
                #随机初始化
                embedding_matrix[index]=np.random.uniform(-0.01, 0.01, self.dim)
                vectors.append(np.random.uniform(-0.01, 0.01, self.dim))
                print('{}没有进行向量化'.format(word))
                nocount = nocount + 1;
            index = index + 1
        print('total {}, word in model have {}'.format(len(train_words_list),count))
        print('有{}个词没有进行embedding向量化'.format(nocount))
        return vectors

    def clean_text(self,text):
        """
        Clean text
        :param text: the string of text
        :return: text string after cleaning
        """
        # unit
        text = re.sub(r"(\d+)kgs ", lambda m: m.group(1) + ' kg ', text)  # e.g. 4kgs => 4 kg
        text = re.sub(r"(\d+)kg ", lambda m: m.group(1) + ' kg ', text)  # e.g. 4kg => 4 kg
        text = re.sub(r"(\d+)k ", lambda m: m.group(1) + '000 ', text)  # e.g. 4k => 4000
        text = re.sub(r"\$(\d+)", lambda m: m.group(1) + ' dollar ', text)
        text = re.sub(r"(\d+)\$", lambda m: m.group(1) + ' dollar ', text)

        # acronym
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"cannot", "can not ", text)
        text = re.sub(r"what\'s", "what is", text)
        text = re.sub(r"What\'s", "what is", text)
        text = re.sub(r"\'ve ", " have ", text)
        text = re.sub(r"n\'t", " not ", text)
        text = re.sub(r"i\'m", "i am ", text)
        text = re.sub(r"I\'m", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"c\+\+", "cplusplus", text)
        text = re.sub(r"c \+\+", "cplusplus", text)
        text = re.sub(r"c \+ \+", "cplusplus", text)
        text = re.sub(r"c#", "csharp", text)
        text = re.sub(r"f#", "fsharp", text)
        text = re.sub(r"g#", "gsharp", text)
        text = re.sub(r" e mail ", " email ", text)
        text = re.sub(r" e \- mail ", " email ", text)
        text = re.sub(r" e\-mail ", " email ", text)
        text = re.sub(r",000", '000', text)
        text = re.sub(r"\'s", " ", text)

        # spelling correction
        text = re.sub(r"ph\.d", "phd", text)
        text = re.sub(r"PhD", "phd", text)
        text = re.sub(r"pokemons", "pokemon", text)
        text = re.sub(r"pokémon", "pokemon", text)
        text = re.sub(r"pokemon go ", "pokemon-go ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" 9 11 ", " 911 ", text)
        text = re.sub(r" j k ", " jk ", text)
        text = re.sub(r" fb ", " facebook ", text)
        text = re.sub(r"facebooks", " facebook ", text)
        text = re.sub(r"facebooking", " facebook ", text)
        text = re.sub(r"insidefacebook", "inside facebook", text)
        text = re.sub(r"donald trump", "trump", text)
        text = re.sub(r"the big bang", "big-bang", text)
        text = re.sub(r"the european union", "eu", text)
        text = re.sub(r" usa ", " america ", text)
        text = re.sub(r" us ", " america ", text)
        text = re.sub(r" u s ", " america ", text)
        text = re.sub(r" U\.S\. ", " america ", text)
        text = re.sub(r" US ", " america ", text)
        text = re.sub(r" American ", " america ", text)
        text = re.sub(r" America ", " america ", text)
        text = re.sub(r" quaro ", " quora ", text)
        text = re.sub(r" mbp ", " macbook-pro ", text)
        text = re.sub(r" mac ", " macbook ", text)
        text = re.sub(r"macbook pro", "macbook-pro", text)
        text = re.sub(r"macbook-pros", "macbook-pro", text)
        text = re.sub(r" 1 ", " one ", text)
        text = re.sub(r" 2 ", " two ", text)
        text = re.sub(r" 3 ", " three ", text)
        text = re.sub(r" 4 ", " four ", text)
        text = re.sub(r" 5 ", " five ", text)
        text = re.sub(r" 6 ", " six ", text)
        text = re.sub(r" 7 ", " seven ", text)
        text = re.sub(r" 8 ", " eight ", text)
        text = re.sub(r" 9 ", " nine ", text)
        text = re.sub(r"googling", " google ", text)
        text = re.sub(r"googled", " google ", text)
        text = re.sub(r"googleable", " google ", text)
        text = re.sub(r"googles", " google ", text)
        text = re.sub(r" rs(\d+)", lambda m: ' rs ' + m.group(1), text)
        text = re.sub(r"(\d+)rs", lambda m: ' rs ' + m.group(1), text)
        text = re.sub(r"the european union", " eu ", text)
        text = re.sub(r"dollars", " dollar ", text)

        # punctuation
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"-", " - ", text)
        text = re.sub(r"/", " / ", text)
        text = re.sub(r"\\", " \ ", text)
        text = re.sub(r"=", " = ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r"\.", " . ", text)
        text = re.sub(r",", " , ", text)
        text = re.sub(r"\?", " ? ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\"", " \" ", text)
        text = re.sub(r"&", " & ", text)
        text = re.sub(r"\|", " | ", text)
        text = re.sub(r";", " ; ", text)
        text = re.sub(r"\(", " ( ", text)
        text = re.sub(r"\)", " ( ", text)

        # symbol replacement
        text = re.sub(r"&", " and ", text)
        text = re.sub(r"\|", " or ", text)
        text = re.sub(r"=", " equal ", text)
        text = re.sub(r"\+", " plus ", text)
        text = re.sub(r"₹", " rs ", text)  # 测试！
        text = re.sub(r"\$", " dollar ", text)

        # remove extra space
        text = ' '.join(text.split())

        return text

    def preprocessing(self,data_df):
        """
        :param data_df:需要处理的数据集
        """
        for index, row in data_df.iterrows():
            # 分别遍历每行的两个句子，并进行分词处理
            for col_name in ["s1", "s2","ans"]:
                clean_str = re.sub(r"[^a-zA-Z0-9 ']", "", str(row[col_name]))  ##只保留a-z A-Z 0-9 其余用空格代替
                out_str = clean_str.strip()  # 去除开头和结尾的空格
                out_str = out_str.lower()  ##将文本统一转换大小写，统一小写
                out_str = self.clean_text(out_str)
                # print(type(out_str))
                # #去除停用词
                # out_str=[word for word in out_str if word not in stopwords.words("english")]
                # 词形还原Lemmatization 例如将started还原成start
                # out_str=[WordNetLemmatizer().lemmatize(word) for word in out_str]

                data_df.at[index, col_name] = out_str
        data_df.fillna(value='nothing', inplace=True)
        return data_df

if __name__ == '__main__':

    train_data = SNLIDataset('train')
    train_data.get_pretrain_embedding()

    #可以先运行data_prepare_ans.py产生运行的词向量和单词表，然后将108行、133-140行注释掉，避免二次产生词向量和词表

