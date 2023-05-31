import pandas as pd
import numpy as np
import json,time
from  tqdm import tqdm
from sklearn.metrics import accuracy_score,classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader,RandomSampler,SequentialSampler
from transformers import BertModel,BertConfig,BertTokenizer,AdamW,get_cosine_schedule_with_warmup

#参数
bert_path = 'bert_model/'  #预训练模型的位置
tokenizer = BertTokenizer.from_pretrained(bert_path)   #初始化分词器
max_len = 100     #数据阻断长度
BATCH_SIZE = 32
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
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

#1.1处理数据成input_ids,token_type_ids,attention_mask,label
class dataSet(Dataset):
    def __init__(self,data_path):
        self.data_path=data_path
        self.max_len=100
        self.input_ids,self.token_type_ids,self.attention_mask,self.input_ids2,self.token_type_ids2,self.attention_mask2,self.labels=self.load_raw_data(data_path)
    def __len__(self):
        return len(self.labels)

    def load_raw_data(self,data_path):
        input_ids,token_type_ids,attention_mask,input_ids2,token_type_ids2,attention_mask2 = [],[],[],[],[],[]
        labels = []
        flag=True
        with open(data_path,encoding='utf-8') as f:
            for i,line in tqdm(enumerate(f)):
                id1,id2,title1,title2,body1,body2,ans,y = line.strip().split('\t')   #删除所有的空格，用\t分割数据集和标签
                s1=title1+body1
                s2=title2+body2

                #调用tokenizer转换成bert需要的数据格式
                encode_dict = tokenizer.encode_plus(text=s1,max_length=self.max_len,padding='max_length',truncation=True)
                #分别获取三个值  目前值的类型为list
                input_ids.append(encode_dict['input_ids'])
                token_type_ids.append(encode_dict['token_type_ids'])
                attention_mask.append(encode_dict['attention_mask'])

                encode_dict2 = tokenizer.encode_plus(text=s2,max_length=self.max_len,padding='max_length',truncation=True)
                #分别获取三个值  目前值的类型为list
                input_ids2.append(encode_dict2['input_ids'])
                token_type_ids2.append(encode_dict2['token_type_ids'])
                attention_mask2.append(encode_dict2['attention_mask'])
                labels.append(int(y))

        return input_ids,token_type_ids,attention_mask,input_ids2,token_type_ids2,attention_mask2,labels


    def __getitem__(self, idx):

        #list转化成tensor格式
        input_ids,token_type_ids,attention_mask = torch.tensor(self.input_ids),torch.tensor(self.token_type_ids),torch.tensor(self.attention_mask)
        input_ids2,token_type_ids2,attention_mask2 = torch.tensor(self.input_ids2),torch.tensor(self.token_type_ids2),torch.tensor(self.attention_mask2)
        labels=torch.tensor(self.labels)
        return input_ids[idx],input_ids2[idx],token_type_ids[idx],token_type_ids2[idx],attention_mask[idx],attention_mask2[idx],labels[idx]


#1.3实例化函数
#训练集带label
train_dataset = dataSet('data/stack4/train.tsv')
print('finish')
train_loader = DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)
#验证集带label
dev_dataset = dataSet('data/stack4/dev.tsv')
dev_loader =DataLoader(dataset=dev_dataset,batch_size=BATCH_SIZE,shuffle=True)
#测试集
test_dataset = dataSet('data/stack4/test.tsv')
test_loader = DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE,shuffle=True)
#得到后续用的数据为train_loader,dev_loader,test_loader

class Bert_Model(nn.Module):
    def __init__(self,bert_path,classes=2):
        super(Bert_Model,self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        self.avg_pooling = AvgPooling()
        for param in self.bert.parameters():
            param.requires_grad=True
        self.fc = nn.Linear(2304,classes)  #直接分类
    def forward(self,input_ids,token_type_ids,attention_mask,input_ids2,token_type_ids2,attention_mask2):
        s1_output = self.bert(input_ids,token_type_ids,attention_mask)[1]  #池化后的输出,是向量
        print("s1_output shape{}".format(s1_output.shape))
        s2_output = self.bert(input_ids2, token_type_ids2, attention_mask2)[1]
        # s1_output_avg=self.avg_pooling(s1_output,attention_mask)
        # s2_output_avg = self.avg_pooling(s2_output, attention_mask2)
        out=torch.cat([torch.abs(s1_output-s2_output),s1_output,s2_output],dim=1)
        print("out shape{}".format(out.shape))
        logit = self.fc(out)    #全连接层,概率矩阵
        print("logit shape{}".format(logit.shape))
        return logit

#实例化bert模型
model = Bert_Model(bert_path).to(DEVICE)

optimizer = AdamW(model.parameters(),lr=2e-5,weight_decay=1e-4)  #使用Adam优化器
#设置学习率
schedule = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=len(train_loader),num_training_steps=EPOCHS*len(test_loader))


# 在验证集上评估模型性能的函数
def evaluate(model, data_loader, device):
    model.eval()  # 防止模型训练改变权值
    val_true, val_pred = [], []
    with torch.no_grad():  # 计算的结构在计算图中,可以进行梯度反转等操作
        for idx, (ids, tpe, att,ids2,tpe2,att2, y) in enumerate(data_loader):  # 得到的y要转换一下数据格式
            y_pred = model(ids.to(device), tpe.to(device), att.to(device),ids2.to(device), tpe2.to(device), att2.to(device))  # 此时得到的是概率矩阵
            y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()  # 将概率矩阵转换成标签并变成list类型
            val_pred.extend(y_pred)  # 将标签值放入列表
            val_true.extend(y.cpu().numpy().tolist())  # 将真实标签转换成list放在列表中

    return accuracy_score(val_true, val_pred)


# 如果是比赛没有labels_test，那么这个函数for里面没有y，输出没有test_true，处理数据的时候没有labels_test放到dataloader里
def predict(model, data_loader, device):
    model.eval()
    test_pred, test_true = [], []
    with torch.no_grad():
        for idx, (ids, tpe, att,ids2,tpe2,att2, y) in enumerate(data_loader):
            y_pred = model(ids.to(device), tpe.to(device), att.to(device),ids2.to(device), tpe2.to(device), att2.to(device))  # 将概率矩阵转化成标签值
            test_pred.extend(y_pred)
            test_true.extend(y.cpu().numpy().tolist())
    return test_pred, test_true


# 训练函数
def train_and_eval(model, train_loader, valid_loader, optimizer, schedule, device, epoch):
    best_acc = 0.0
    patience = 0
    criterion = nn.CrossEntropyLoss()  # 损失函数
    for i in range(epoch):
        start = time.time()
        model.train()  # 开始训练
        print("***************Running training epoch{}************".format(i + 1))
        train_loss_sum = 0.0
        for idx, (ids, tpe, att,ids2,tpe2,att2, y) in enumerate(train_loader):
            # y=y.unsqueeze(-1)
            print('y shape{}'.format(y.shape))

            ids, tpe, att,ids2,tpe2,att2, y= ids.to(device), tpe.to(device), att.to(device),ids2.to(device), tpe2.to(device), att2.to(device),y.to(device,dtype=torch.long)
            y_pred = model(ids, tpe, att,ids2,tpe2,att2)  # 加载模型获得概率矩阵
            loss = criterion(y_pred, y)  # 计算损失
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新优化参数
            schedule.step()  # 更新学习率
            train_loss_sum += loss.item()
            # 只打印五次结果
            if (idx + 1) % (len(train_loader) // 5) == 0:
                print("Epoch {:04d} | Step {:04d}/{:04d} | Loss {:.4f} | Time {:.4f}".format(
                    i + 1, idx + 1, len(train_loader), train_loss_sum / (idx + 1), time.time() - start))
            # 每一次epoch输出一个准确率
        model.eval()
        acc = evaluate(model, valid_loader, device)  # 验证模型的性能
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_bert_model.pth")  # 保存最好的模型
        print("current acc is {:.4f},best acc is {:.4f}".format(acc, best_acc))
        print("time costed = {}s \n".format(round(time.time() - start, 5)))

train_and_eval(model,train_loader,dev_loader,optimizer,schedule,DEVICE,EPOCHS)
model.load_state_dict(torch.load("best_bert_model.pth"))
#得到预测标签和真实标签
test_pred,test_true= predict(model,test_loader,DEVICE)
#输出测试机的准确率
print("\n Test Accuracy = {} \n ".format(accuracy_score(test_true,test_pred)))
#打印各项验证指标
print(classification_report(test_true,test_pred,digits=4))
print(test_pred[:10])
print('------------------')
print(test_true[:10])
#nohup python3 bert_cqa.py  >./result/bert_cqa 2>&1 &