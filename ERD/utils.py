import json
import os
import time
import datetime
import numpy as np
import gensim
import random
import math
import re
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
with open("/Users/chenguanjin/Downloads/word2vec.pkl", "rb") as handle:
    word2vec = pickle.load(handle)
print("load word2vec finished")

FLAGS={"post_fn":2,
       "time_limit":48,
       "batch_size":16,
       "hidden_dim": 100,
       "embedding_dim": 300,
       "max_seq_len": 100,
       "max_sent_len": 50,
       "class_num": 2,
       "action_num": 2,
       "random_rate": 0.01,
       "OBSERVE": 1000,
       "max_memory": 80000,
       "reward_rate": 0.2,
       }

files = []
data = {}
data_ID = []
data_len = []
data_y = []

valid_data_ID = []
valid_data_y = []
valid_data_len = []
reward_counter = 0
eval_flag = 0

#get_data_ID
def get_data_ID():
    global data_ID
    return data_ID

#get_data_len
def get_data_len():
    global data_len
    return data_len

#get_curtime
def get_curtime():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

#get all files in []
def list_files(data_path):
    global data, files
    fs = os.listdir(data_path)
    for f1 in fs:
        tmp_path = os.path.join(data_path, f1)
        if not os.path.isdir(tmp_path):
            if tmp_path.split('.')[-1] == 'json':
                files.append(tmp_path)
        else:
            list_files(tmp_path)

#trans the time
def str2timestamp(str_time):
    month = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
             'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
             'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    ss = str_time.split(' ')
    m_time = ss[5] + "-" + month[ss[1]] + '-' + ss[2] + ' ' + ss[3]
    d = datetime.datetime.strptime(m_time, "%Y-%m-%d %H:%M:%S")
    t = d.timetuple()
    timeStamp = int(time.mktime(t))
    return timeStamp

#get josn data
def data_process(file_path):
    ret = {}
    ss = file_path.split("/")
    data = json.load(open(file_path, mode="r", encoding="utf-8"))
    ret[ss[5]] = {'label': [ss[6]], 'text': [data['text'].lower()], 'created_at': [str2timestamp(data['created_at'])]}
    return ret

#re trans
def transIrregularWord(line):
    if not line:
        return ''
    line.lower()
    line = re.sub("@[^ ]*", "", line)
    line = re.sub("#[^ ]*", "", line)
    line = re.sub("http(.?)://[^ ]*", "", line)
    return line 

#sort by time
def sortTempList(temp_list):
    time = np.array([item[0] for item in temp_list])
    posts = np.array([item[1] for item in temp_list])
    labels = np.array([item[2] for item in temp_list])
    idxs = time.argsort().tolist()
    rst = [[t, p ,q] for (t, p, q) in zip(time[idxs], posts[idxs],labels[idxs])]
    del time, posts, labels
    return rst

#load data ,get information in global variables
def load_data(data_path):
    # get data files path
    global data, files, data_ID, data_len, eval_flag
    data = {}
    files = []
    data_ID = []
    data_len = []
    list_files(data_path) #load all filepath to files
    max_sent = 0
    # load data to json
    for file in files:
        td = data_process(file) 
        """
        {'ottawashooting': {'label': 'rumours', 
                     'text': ['extended: dramatic video of gunfire inside hallways of parliament hill. (the globe and mail) http://t.co/sbou4rap96 http://t.co/71rnixs3eb'], 
                     'created_at': [1413964222]
                    }
        }
        """
        for key in td.keys(): 
            if key in data:
                data[key]['text'].append(td[key]['text'][0])
                data[key]['created_at'].append(td[key]['created_at'][0])
                data[key]['label'].append(td[key]['label'][0])
            else:
                data[key] = td[key]
    # convert to my data style
    for key, value in data.items():
        temp_list = []
        for i in range(len(data[key]['text'])):
            temp_list.append([data[key]['created_at'][i], data[key]['text'][i],data[key]['label'][i]])
        temp_list = sortTempList(temp_list)
        data[key]['text'] = []
        data[key]['created_at'] = []
        data[key]['label'] = []
        ttext = ""
        last = 0
        for i in range(len(temp_list)):
            if i % FLAGS["post_fn"] == 0: 
                if len(ttext) > 0: 
                    words = transIrregularWord(ttext)
                    data[key]['text'].append(words)
                    data[key]['created_at'].append(temp_list[i][0])
                    data[key]['label'].append(temp_list[i][2])
                ttext = temp_list[i][1]
            else:
                ttext += " " + temp_list[i][1]
            last = i

        if len(ttext) > 0:
            words = transIrregularWord(ttext)
            data[key]['text'].append(words)
            data[key]['created_at'].append(temp_list[last][0])
            data[key]['label'].append(temp_list[last][2])
            
    for key in data.keys():
        data_ID.append(key)
    data_ID = random.sample(data_ID, len(data_ID)) 
    for i in range(len(data_ID)):
        data_len.append(len(data[data_ID[i]]['text']))
        if data[data_ID[i]]['label'] == "rumours":
            data_y.append([1.0, 0.0])
        else:
            data_y.append([0.0, 1.0])

    eval_flag = int(len(data_ID) / 4) * 3

    #print("{} data loaded".format(len(data)))

file_name="/Users/chenguanjin/Downloads/pheme-rnr-dataset"
load_data(file_name)

class my_dataset(Dataset):
    def __init__(self):
        load_data(file_name)
        self.data=data["charliehebdo"]
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        self.text=self.data["text"]
        self.date=self.data["created_at"]
        self.label=self.data["label"]
        return self.text[idx],self.date[idx],self.label[idx]

def collate_fn(data):
    b=len(data)
    #x=np.zeros((b,100,300))
    text=[i[0] for i in data]
    date=[i[1] for i in data]
    label=[i[2] for i in data]
    #print(text)
    seq=[]
    for i in range(b):
        sent=[]
        if len(text)<FLAGS["max_sent_len"]:
            for k in range(len(text)):
                try:
                    sent.append([word2vec[text[k]]])
                except KeyError:
                    sent.append([word2vec['an'] +  word2vec['unknown'] + word2vec['word']]) 
            for k in range(len(text),FLAGS["max_sent_len"]):
                sent.append([word2vec['an'] +  word2vec['unknown'] + word2vec['word']])
        else:
            for k in range(FLAGS["max_sent_len"]):
                try:
                    sent.append([word2vec[text[k]]])
                except KeyError:
                    sent.append([word2vec['an'] +  word2vec['unknown'] + word2vec['word']])
        seq.append(sent)
        if label[i]=="rumours":
            label[i]=1
        else:
            label[i]=0
    seq=torch.tensor(seq)
    seq=seq.reshape(b,FLAGS["max_sent_len"],-1)
    date=torch.tensor(date).reshape(b,-1)
    label=torch.tensor(label).reshape(b,-1)
    return seq,date,label




train_dataset=my_dataset()
loader=DataLoader(dataset=train_dataset,batch_size=16,shuffle=True,collate_fn=collate_fn,drop_last=True)
#for i,(a,b,c) in enumerate(loader):
    #break
#print(a.shape,b.shape,c.shape)
