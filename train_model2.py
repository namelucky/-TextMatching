import os
from torch.utils.data import DataLoader
from loadlog import configure_logging
from data_prepare_ans import SNLIDataset
from data_prepare_title import SNLIDatasetTitle


from models.model2 import LMATCH
import torch
from seed_util import seed_torch
from transformers import get_linear_schedule_with_warmup
import time
import seaborn as sns
import matplotlib.pyplot as plt
configure_logging("logging_config.json")

begin_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
device = torch.device("cuda:0")
seed_torch(512)
attn_type = 'adatrans'  # 'adatrans',transformer
n_epochs = 60
batch_size = 48
warmup_steps = 0.01
after_norm = 1
model_type = 'adatrans'
normalize_embed = True
dropout = 0.15
fc_dropout = 0.4
sentence_max_length = 25  # 句子的截断长度

num_layers = 2
n_heads = 4
head_dims = 128
lr = 0.0001
warm_up_proportion = 0.01

d_model = n_heads * head_dims
dim_feedforward = int(2 * d_model)

train_dataset = SNLIDataset('train', length=sentence_max_length, mask=False)
val_dataset = SNLIDataset('dev', length=sentence_max_length, mask=False)
test_dataset = SNLIDataset('test', length=sentence_max_length,mask=False)

train_dataset_title = SNLIDatasetTitle('train', length=24, mask=False)
val_dataset_title = SNLIDatasetTitle('dev', length=24, mask=False)
test_dataset_title = SNLIDatasetTitle('test', length=24,mask=False)


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

train_loader_title = DataLoader(dataset=train_dataset_title, batch_size=batch_size, shuffle=False, num_workers=1)
test_loader_title = DataLoader(dataset=test_dataset_title, batch_size=batch_size, shuffle=False, num_workers=1)
val_loader_title = DataLoader(dataset=val_dataset_title, batch_size=batch_size, shuffle=False, num_workers=1)

embedding_pretrained = train_dataset.get_pretrain_embedding()

model = LMATCH(vocab_size=len(train_dataset.train_text_vocab.get_itos()), num_layers=num_layers,
               d_model=d_model, n_head=n_heads,
               feedforward_dim=dim_feedforward, dropout=dropout,
               embedding_pretrained=embedding_pretrained,
               after_norm=after_norm, attn_type=attn_type,
               fc_dropout=fc_dropout,
               pos_embed=None,
               scale=False)
model.to(device)

# construct loss and optimizer
criterion = torch.nn.BCELoss(reduction='mean')
criterion.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=lr,weight_decay=0.000001)

# 更新次数
total_step = len(train_dataset) * n_epochs // batch_size
# 学习率计划
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_step * warm_up_proportion,
                                            num_training_steps=total_step)


def train(epoch):
    train_loss = 0.0
    count = 0
    tp = 0
    fp = 0
    fn = 0
    correct=0
    total_num = len(train_loader.dataset)
    for i , (data1,data2) in enumerate(zip(train_loader,train_loader_title)):

        ans1,ans2,labels = data1
        t1,t2,title,title2=data2

        t1 = t1.to(device)
        t2 = t2.to(device)
        ans1 = ans1.to(device)
        ans2 = ans2.to(device)

        y_pred = model([t1, t2],  [ans1, ans2])  # 前向传播
        pred_label = y_pred.squeeze()
        pred_label = pred_label.to(torch.float32)
        labels = labels.to(torch.float32)
        labels = labels.to(device)
        pred_label = pred_label.to(device)
        # print('labels{}'.format(labels.shape))
        loss = criterion(pred_label, labels)  # 计算损失
        optimizer.zero_grad()  # 清0梯度
        loss.backward()  ##反向传播
        optimizer.step()  ##更新梯度参数
        scheduler.step()
        train_loss += loss.item()  ##一个epoch内所有batch的损失和
        count = i
        pred_label = pred_label.cpu()
        labels = labels.cpu()
        pred = torch.where(pred_label >= 0.5, torch.tensor([1.0]), torch.tensor([0.0]))
        # acc = torch.eq(pred, labels).sum().item() / labels.size(0)
        # print("train acc:", acc)
        tp += torch.sum((pred == labels) & (pred == 1))
        fp += torch.sum((pred != labels) & (pred == 1))
        fn += torch.sum((pred != labels) & (pred == 0))
        correct += torch.sum(pred == labels)
    tp = tp.data.item()
    fp = fp.data.item()
    fn = fn.data.item()
    precision = tp / (tp + fp+1e-6)
    recall = tp / (tp + fn+1e-6)
    F1 = 2 * (precision * recall) / ((precision + recall+1e-6))
    acc = correct / total_num
    print("ACC {} ,train F1: {} ".format(acc, F1))
    print("第{}epoch的 train loss:{}".format(epoch, train_loss / count))
    return acc,F1,train_loss / count

def test():
    tp = 0
    fp = 0
    fn = 0
    total_num = len(test_loader.dataset)
    correct = 0
    with torch.no_grad():
        for i, (data1, data2) in enumerate(zip(test_loader, test_loader_title), 0):
            ans1,ans2,labels = data1
            t1, t2, title, title2 = data2
            t1 = t1.to(device)
            t2 = t2.to(device)
            ans1 = ans1.to(device)
            ans2 = ans2.to(device)
            y_pred = model([t1,t2], [ans1, ans2])   # 前向传播
            pred_label = y_pred.squeeze()
            pred_label = pred_label.cpu()
            labels = labels.cpu()
            # pred=pred_label
            pred = torch.where(pred_label >= 0.5, torch.tensor([1.0]), torch.tensor([0.0]))
            # acc = torch.eq(pred, labels).sum().item() / labels.size(0)
            tp += torch.sum((pred == labels) & (pred == 1))
            fp += torch.sum((pred != labels) & (pred == 1))
            fn += torch.sum((pred != labels) & (pred == 0))
            correct += torch.sum(pred == labels)

        tp = tp.data.item()
        fp = fp.data.item()
        fn = fn.data.item()
        precision = tp / (tp + fp+1e-6)
        recall = tp / (tp + fn+1e-6)
        F1 = 2 * (precision * recall) / ((precision + recall+1e-6))
        acc = correct / total_num
        print("test acc:{} test F1{}".format(acc,F1))
    return acc, F1


def val():
    model.eval()
    val_loss = 0
    correct = 0
    tp = 0
    fp = 0
    fn = 0
    total_num = len(val_loader.dataset)
    print('Number of validate dataset:{}'.format(total_num))
    with torch.no_grad():
        for i, (data1, data2) in enumerate(zip(val_loader, val_loader_title), 0):
            ans1,ans2,labels = data1
            t1, t2, title, title2 = data2
            t1 = t1.to(device)
            t2 = t2.to(device)
            ans1 = ans1.to(device)
            ans2 = ans2.to(device)
            y_pred = model([t1,t2], [ans1, ans2])  # 前向传播
            pred_label = y_pred.squeeze()
            # print(pred_label.shape)
            pred_label = pred_label.to(torch.float32)
            labels = labels.to(torch.float32)
            labels = labels.to(device)
            pred_label = pred_label.to(device)

            loss = criterion(pred_label, labels)  # 计算损失
            pred_label = pred_label.cpu()
            labels = labels.cpu()
            pred = torch.where(pred_label >= 0.5, torch.tensor([1.0]), torch.tensor([0.0]))
            correct += torch.sum(pred == labels)
            tp += torch.sum((pred == labels) & (pred == 1))
            fp += torch.sum((pred != labels) & (pred == 1))
            fn += torch.sum((pred != labels) & (pred == 0))
            print_loss = loss.data.item()
            val_loss += print_loss
        correct = correct.data.item()
        tp = tp.data.item()
        fp = fp.data.item()
        fn = fn.data.item()
        precision = tp / (tp + fp+1e-6)
        recall = tp / (tp + fn+1e-6)
        F1 = 2 * (precision * recall) / ((precision + recall+1e-6))
        acc = correct / total_num
        avg_loss = val_loss / len(val_loader)
        print('\nVal set:Average loss:{:.4f}, Accuracy:{}/{} ({:.4f}%)\n'.format(
            avg_loss, correct, len(val_loader.dataset), 100 * acc
        ))
        print("ACC {} ,val F1: {} ".format(acc, F1))

    return acc, F1,avg_loss


def load_model():
    model.load_state_dict(torch.load('./model/train_stack_xception_title_body_ans_relv4.tar'))
    test()

if __name__ == '__main__':

    best_val_F1 = 0
    best_val_acc = 0
    best_test_F1=0 #这个是根据测试集的数据看的
    best_test_acc=0
    test_test_F1=0 #这个是根据测试集自己的数据看的
    test_test_acc = 0  # 这个是根据测试集自己的数据看的

    train_acc_list=[]
    train_F1_list=[]
    train_loss_list=[]
    val_acc_list=[]
    val_F1_list=[]
    val_loss_list=[]
    test_acc_list = []
    test_F1_list = []

    epoch_list=[]

    for epoch in range(n_epochs):
        train_acc,train_F1,train_loss=train(epoch)
        val_acc, val_F1,val_loss = val()
        test_acc, test_F1 = test()#0.5 0.7 0.6 0.4 0.6

        train_acc_list.append(train_acc.item())

        train_F1_list.append(train_F1)

        train_loss_list.append(train_loss)

        val_acc_list.append(val_acc)

        val_F1_list.append(val_F1)

        val_loss_list.append(val_loss)

        test_acc_list.append(test_acc)

        test_F1_list.append(test_F1)

        epoch_list.append(epoch+1)

        if val_F1 > best_val_F1:
            best_val_F1 = val_F1
            best_val_acc = val_acc
            # test_acc, test_F1 = test()

            if(test_F1>best_test_F1):
                best_test_F1=test_F1
                best_test_acc=test_acc

            model_name = 'train_stack_xception_title_body_ans_relv4_2.tar'
            torch.save({
                'state_dict': model.state_dict()
            }, os.path.join('./model', model_name))

        if(test_F1>test_test_F1):
            test_test_F1=test_F1
            test_test_acc=test_acc

        print('lattice levelbest_val_acc:{}  Val F1:{} val_loss:{} epoch{}'.format(val_acc,val_F1,val_loss,epoch))
        print('lattice level best Test acc:{} Test F1{} epoch{}'.format(test_acc, test_F1, epoch))
        print('lattice level 只看测试集 best Test acc:{} Test F1 {} '.format(best_test_acc, best_test_F1))
        print('lattice level 只看测试集 best Test acc:{} Test F1 {} '.format(test_test_acc, test_test_F1))
    # load_model()
    end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print('begin',begin_time)
    print('end',end_time)

    #绘图
    fig, (ax1, ax2,ax3) = plt.subplots(3, 3, figsize=(50, 24))
    sns.lineplot(x=epoch_list,y=train_acc_list, ax=ax1[0],label="train acc")
    sns.lineplot(x=epoch_list, y=train_F1_list, ax=ax1[1],label="train_F1_list")
    sns.lineplot(x=epoch_list, y=train_loss_list, ax=ax1[2],label="train_loss_list")


    sns.lineplot(x=epoch_list, y=val_acc_list, ax=ax2[0],label="val_acc_list")
    sns.lineplot(x=epoch_list, y=val_F1_list, ax=ax2[1],label="val_F1_list")
    sns.lineplot(x=epoch_list,y=val_loss_list, ax=ax2[2],label="val_loss_list")

    # sns.lineplot(x=epoch_list, y=test_acc_list, ax=ax3[0],label="test_acc_list")
    sns.lineplot(x=epoch_list, y=test_F1_list,  ax=ax3[1],label="test_F1_list")
    plt.savefig('train_model2.jpg')
    plt.show()


# nohup python3 train_model2.py  >./result/train_model2 2>&1 &