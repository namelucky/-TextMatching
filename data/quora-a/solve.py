import json
import pandas as pd

def convert(file):
    'file train test dev'
    num=0
    data=[]
    with open('./json/'+file+'.jsonl', 'r', encoding='utf-8') as fp:
        # load()函数将fp(一个支持.read()的文件类对象，包含一个JSON文档)反序列化为一个Python对象
        for line in fp.readlines():
            list1 = []
            dic=json.loads(line)
            list1.append(dic['qu'])
            list1.append(dic['qa'])
            list1.append(dic['is_duplicate'])
            list1.append(dic['ans'])
            data.append(list1)
            num=num+1
    fp.close()
    df=pd.DataFrame(data,columns=['qu','qa','label','ans'])
    df.to_csv(file+'.tsv',index=None,header=None,sep='\t')
    print('{} 数量{}'.format(file,num))


convert('train')
convert('test')
convert('dev')