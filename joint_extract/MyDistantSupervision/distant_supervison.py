# encoding=utf-8
from collections import defaultdict
from byteify import byteify
import ahocorasick
import re

all=True
Target_R=['P1376','P131','P150']

def defs():
    return defaultdict(int)

def load_relation():
    rval=defaultdict(defs)
    def unk():
        return 'UNK'
    entityid2name=defaultdict(unk)
    ii1=open('entityid2name.txt','r')
    for line in ii1:
        name,id_=line.strip().split('\t')
        entityid2name[id_]=name

    ii2=open('relationbyid.txt','r')
    for line in ii2:
        name,r_id,entity2,is_entity =line.strip().split('\t')
        if eval(is_entity):
            entity2=entityid2name[entity2]
        rval[name][entity2]=r_id
    return rval

def buildA(keys):
    A = ahocorasick.Automaton()
    for key in keys:
        if re.match('\w+',key):continue 
        A.add_word(key,key)
    A.make_automaton()
    return A

fuhao=re.compile(u'。|？|“|”| |；|;')
def data():
    ii=open('wiki_abstract.txt','r')
    for line in ii:
        name,text=line.split('\t')
        sents=fuhao.split(text.decode('utf-8'))
        for sent in sents:
            if sent:
                yield sent.encode('utf-8')

def distant_supvervised():
    # relation2name={}
    relation=load_relation()#relation[a][b]=r_id
    names=set(relation.keys())
    for name in relation:
        names.update(relation[name].keys())
    A=buildA(names)
    rval=[]
    oo=open('result.txt','w')
    for sent in data():
        matchs=A.iter(sent)#应该走分词
        for enda,a in matchs:
            for endb,b in matchs:
                if a==b:continue
                if a in b:continue
                if b in a:continue
                start=min(enda+1-len(a),endb+1-len(b))
                end=max(enda+1,endb+1)
                if a in relation and b in relation[a]:
                    if relation[a][b] in Target_R or all:
                        oo.write(a+'\t'+b+'\t'+relation[a][b]+'\t'+sent[start:end]+'\n')                
                    #rval.append((a,b,relation[a][b]))
                if b in relation and a in relation[b]:
                    if relation[b][a] in Target_R or all:
                        oo.write(b+'\t'+a+'\t'+relation[b][a]+'\t'+sent[start:end]+'\n')                
                    #rval.append((b,a,relation[b][a]))
    return 

if __name__=='__main__':
   distant_supvervised() 
