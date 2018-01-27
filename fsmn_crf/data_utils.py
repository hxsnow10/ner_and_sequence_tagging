# encoding=utf-8
import numpy as np

class NerDataset():

    def __init__(self, data_path, char2id, label2id, batch_size, seq_len):
        self.data_path=data_path
        self.char2id=char2id
        self.label2id=label2id
        self.batch_size=batch_size
        self.seq_len=seq_len

    def __iter__(self):
        X_sample=[]
        y_sample=[]
        data=[[],[]]
        for line in open(self.data_path):
            line=line.strip()
            if not line:
                X_sample=X_sample[:self.seq_len]
                y_sample=y_sample[:self.seq_len]
                if len(X_sample)<self.seq_len:
                    X_sample+= [0,]*(self.seq_len-len(X_sample))
                    y_sample+= [0,]*(self.seq_len-len(y_sample))
                data[0].append(X_sample)
                data[1].append(y_sample)
                X_sample=[]
                y_sample=[]
                if len(data[0])>=10*self.batch_size:
                    while len(data[0])>=self.batch_size:
                        s=[np.array(x[:self.batch_size]) for x in data]
                        yield s
                        data=[x[self.batch_size:] for x in data]
            else:
                char,label=line.split()
                charid=self.char2id.get(char.decode('utf-8'),self.char2id["UNT"])
                labelid=self.label2id.get(label.decode('utf-8'),0)
                X_sample.append(charid)
                y_sample.append(labelid)
        while len(data[0])>=self.batch_size:
            s=[np.array(x[:self.batch_size]) for x in data]
            yield s
            data=[x[self.batch_size:] for x in data]

