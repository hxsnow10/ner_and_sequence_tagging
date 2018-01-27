#!/usr/bin/env python
#coding:utf-8

__author__ = 'YangXiao'

from polyglot.text import Text
from sklearn import metrics
import pickle
import sys

"""python test_ner.py output"""

def format_entity(s):
    s = s.lower()
    if 'per' in s:
        return 'PER'
    elif 'loc' in s:
        return 'LOC'
    elif 'org' in s:
        return 'ORG'
    elif 'time' in s:
        return 'TIME'
    elif 'date' in s:
        return 'TIME'
    else:
        return 'O'

def flatmap(li):
    a = []
    for l in li:
        if l != '':
            a.append(Text(l).words)
    #a = map(lambda x:Text(x).words, li)
        
    return [it for i in a for it in i]


pred = sys.argv[1]
all_corpus, words, label = pickle.load(open('test_param.pkl'))
label_flat = [i for it in label for i in it]
out_sete = []
# with open('output', 'r') as fr:
with open(pred, 'r') as fr:
    ss = []
    for i in fr:
        line = i.strip()
        if line != '':
            ss.append(line)
        else:
            out_sete.append(ss)
            ss = []
assert len(words) == len(out_sete)


pred_output = []
for i in range(len(all_corpus)):    
    
    pred_output_ent = dict(zip(iter(words[i]), iter(['O']*len(words[i]))))
    ent_output_p = []
    for line in out_sete[i]:
        ll = line.strip().split(' ')
        if len(ll) >1:
            tag = format_entity(ll[1])
            ws = flatmap([ll[0]])
            for j in ws:
                pred_output_ent[j] = tag

            
    for k in words[i]:
        ent_output_p.append(pred_output_ent[k]) 

    pred_output.append(ent_output_p)

pred_output_flat = [i for it in pred_output for i in it]

diff_entity = ['PER', 'LOC', 'ORG', 'TIME']
if 'TIME' not in set(pred_output_flat):
	diff_entity.remove('TIME')

for I in diff_entity:
	print I
	label_p = map(lambda x:int(x==I),label_flat)
	pred_p = map(lambda x:int(x==I),pred_output_flat)
	print 'precision:', metrics.precision_score(label_p, pred_p)
	print 'recall:', metrics.recall_score(label_p, pred_p)
	print 'f1:', metrics.f1_score(label_p, pred_p)
	print '--'*20+'\n'

