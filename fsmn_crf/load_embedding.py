#coding:utf-8

import gzip
import pickle
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
# ar_w2v_path = os.path.join(cur_path, "ar_wiki_w2v_sg_text2.txt.gz")
ar_w2v_path = os.path.join(cur_path, "w2v_extract_social_tweet_status_1_ar_UNT.txt.gz")
import numpy as np

def loadWord2vec(w2v_file=ar_w2v_path, embed_file="pretrain_embd.pkl"):
    if os.path.isfile(embed_file):
        with open(embed_file, 'r') as fr:
            vocab, embd = pickle.load(fr)
        print "loaded word2vec"
    else:
        vocab = []
        embd = []
        fr = open(w2v_file, 'r')
        line = fr.readline().decode('utf-8').strip()
        # print line
        word_dim = int(line.split(' ')[1])
        vocab.append("<PAD>")
        embd.append([0.0] * word_dim)
        for line in fr:
            if len(line.strip().split(' '))!=word_dim+1:continue
            row = line.strip().decode('utf-8').split(' ')
            vocab.append(row[0])
            embd.append(map(float, row[1:]))
        #tweet word2vec has word "UNT"
        if "UNT" not in vocab:
            vocab.append("UNT")
            embd.append([0.0] * word_dim)
        print "builded word2vec"
        fr.close()
        print "len(vocab):{0}, len(embd):{1}".format(len(vocab), len(embd))
        saveEmbed(vocab, embd, embed_file)
    print np.array(embd, dtype=np.float32)
    return vocab, embd

def saveEmbed(vocab, embd, savename):
    print "saving embedding to:", savename
    with open(savename, 'w') as fw:
        pickle.dump([vocab, embd], fw)


if __name__ == '__main__':
    vocab, embd = loadWord2vec()
    saveEmbed(vocab, embd, 'pretrain_embd.pkl')
    print "saved pretrain embedding."
