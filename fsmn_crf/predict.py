#coding:utf-8
import tensorflow as tf
from ner import NerModel, NerModelDev
import numpy as np
import sys
from sklearn import metrics
from load_embedding import loadWord2vec
# from ner_train_embd import NerModelTrainEmbd
import helper
import time
import math
import re
import json
import os,sys

cur_dir = os.path.dirname( os.path.abspath(__file__)) or os.getcwd()
splits=re.compile(u"(。|；|！|？|；|\n| |[|]|\(|\))", re.UNICODE)
MODEL_PATH=os.path.join(cur_dir, 'model_1_Bifsmn_None_300','model')
model_path=MODEL_PATH
batch_s = 5
seq_len = 200
savemap_file_path = os.path.join(cur_dir,"map.pkl")
vec_path=os.path.join(cur_dir,"xhw_char_300.vec")
pretrain_embd_path = os.path.join(cur_dir, "xhw_char_300.pkl")
emb_size=300
use_pretrain_embd = True

class Model(object):
    def __init__(self, vocab_size, path_model=MODEL_PATH, embedding=None, batch_size=1, seq_len=200, emb_size=128):
        self.num_steps = seq_len
        self.batch_size = batch_size
        # self.embedding = embedding
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            # self.ner_model = NerModel(vocab_size, embedding, False, batch_size=self.batch_size)
            self.ner_model = NerModelDev(vocab_size, embedding, False, batch_size=self.batch_size, seq_len=seq_len,\
                    emb_size=emb_size)
            # self.ner_model = NerModelTrainEmbd(vocab_size, False, batch_size=self.batch_size)
        config = tf.ConfigProto(device_count={'GPU': 0})
        self.sess = tf.Session(config=config)
        saver = tf.train.Saver()
        try:
            saver.restore(self.sess, path_model)
            print "restore tensorflow model "
        except:
            self.sess.run(tf.global_variables_initializer())
            print "init model "
        # char2id, self.id2char, label2id, self.id2label = helper.buildMap(None)
        self.char2id, self.id2char, self.label2id, self.id2label = \
            helper.buildMap(None, savemap_file=savemap_file_path,
                use_pretrain_embd=use_pretrain_embd)
        self.letter2id = helper.buildCharMap()

    def extarct_sents(self, text):
        rval=[]
        sents=splits.split(text)
        for sent in sents:
            if rval and len(rval[-1]+sent)<=self.num_steps:
                rval[-1]+=sent
            else:
                while len(sent)>=self.num_steps:
                    rval.append(sent[:self.num_steps])
                    sent=sent[self.num_steps:]
                if sent:
                    rval.append(sent[:self.num_steps])
                    
        return rval
    #@profile
    def get_ner(self, text):
        text=text.decode('utf-8')
        sents=self.extarct_sents(text)
        assert ''.join(sents)==text
        X=[]
        start=0
        rval=[]
        labels=[]
        for k,sent in enumerate(sents):
            tmp=[self.char2id.get(s,self.char2id["UNT"]) for s in sent][:self.num_steps]
            X.append(tmp)
            if len(X)==self.batch_size or k+1==len(sents):
                X+=[[] for _ in range(self.batch_size-len(X))]
                for i in range(len(X)):
                    X[i]+=[0,]*(self.num_steps-len(X[i]))
                seq_label, seq_len = self.predict(X)
                X=[]
                labels.extend(seq_label)
        assert len(text)==len(labels)
        state,word,start=None,'',0
        i=0
        for k,(ch,tag) in enumerate(zip(text, labels)):
            if tag=='O' or tag=='<PAD>' or ch in ['\n']:
                if state and word:
                    rval.append((state,word,(start,start+len(word.decode('utf-8')))))
                state,word=None,''
            else:
                l,state_ = tag.split('-')
                if state:
                    if state_==state and l!='E':
                        word+=ch
                    elif l=='E':
                        word+=ch
                        word=word.strip()
                        rval.append((state,word,(start,start+len(word.decode('utf-8')))))
                        state,word=None,''
                    else:
                        rval.append((state,word,(start,start+len(word.decode('utf-8')))))
                        state,word,start=state_,ch,i
                else: 
                    state,word,start=state_,ch,i
            i+=len(ch.decode('utf-8'))
        if state and word:
            rval.append((state,word,(start,start+len(word.decode('utf-8')))))
        rval=[(state,word.strip(),loc) for state,word,loc in rval if word.strip()]
        return rval
                        
    #@profile           
    def predict(self, X, X_words=None):
        viterbi_sequences = []
        # logits, sequence_actual_length, transition_params = self.ner_model.sess.run(
        #     [self.ner_model.logits, self.ner_model.length, self.ner_model.transition_params],
        #     feed_dict={self.ner_model.input_data:X})
        # X_char = helper.wordBatch2charBatch(X, self.id2char, self.letter2id)

        '''
        from tensorflow.python.client import timeline
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('timeline_01.json', 'w') as f:
                f.write(chrome_trace)
        '''
        # 连起来再一次性vd会变快吗
        # beam_search 会变快吗
        logits, sequence_actual_length, transition_params = self.sess.run(
            [self.ner_model.logits, self.ner_model.length, self.ner_model.transition_params],
            feed_dict={self.ner_model.input_data: X})
        for logit, seq_len in zip(logits, sequence_actual_length):
            if seq_len==0:continue
            logit_actual = logit[:seq_len]
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                logit_actual, transition_params)
            viterbi_sequences.extend(viterbi_sequence)

        # print('共标记句子数: %d' % len(X))
        seq_label = map(lambda x: self.id2label[x], viterbi_sequences)
        return seq_label, seq_len
    #@profile
    def evaluate(self, data):
        eval_y_true = []
        eval_y_pred = []
        eval_x = []
        y_pred, y_true = [], []
        st = time.time()
        for i,(X,y) in enumerate(data):
            try:
                y_pred_seq,seq_len = self.predict(X)
                y_pred.extend(y_pred_seq)
                y=y[:seq_len]
                y_true.extend(y)
                if y_pred_seq != y:
                    eval_y_pred.append(y_pred_seq)
                    eval_y_true.append(y)
                    eval_x.append(X)

            except Exception, e:
                import traceback
                traceback.print_exc()
                print Exception, ':', e
        et = time.time()
        print "predicted count {0}, cost time: {1}, {2}lines/sec".format(len(X), et - st,
                                                                         len(X) / float(et - st))

        y_true_flat = [it for item in y_true for it in item]
        assert len(y_pred) == len(y_true_flat)
        diff_entity = self.label2id.keys()


        for I in diff_entity:
            print I
            label_p = map(lambda x: int(x == I), y_true_flat)
            pred_p = map(lambda x: int(x == I), y_pred)
            print 'precision:', metrics.precision_score(label_p, pred_p)
            print 'recall:', metrics.recall_score(label_p, pred_p)
            print 'f1:', metrics.f1_score(label_p, pred_p)
            print '--' * 20 + '\n'

        self.saveBadCase(eval_y_true, eval_y_pred, eval_x)

    def saveBadCase(self, y_true, y_pred, X):
        save_file = 'badcase.txt'
        bad_count = len(y_true)
        print "bad case count:", bad_count
        assert bad_count == len(y_pred)
        assert bad_count == len(X)
        with open(save_file, 'w') as fw:
            fw.write('label_true\tlabel_pred\tword\n')
            for i in range(bad_count):
                sent_len = len(y_true[i])
                x_no_pad = np.where(X[i]==0)[0][0]#filter(lambda x:x!=0,X[i])
                assert sent_len == len(y_pred[i])
                assert sent_len == x_no_pad
                for j in range(sent_len):
                    word = self.id2char[X[i][j]]
                    sent = y_true[i][j]+'\t'+y_pred[i][j]+'\t'+word+'\n'
                    fw.write(sent.encode('utf-8'))
                fw.write('\n')

def load_model():
    # w2v_path = "ar_tweet_w2v_sg.txt.gz"
    use_pretrain_embd = True
    char2id, id2char, label2id, id2label = helper.buildMap(None, savemap_file=savemap_file_path,
                                                           use_pretrain_embd=use_pretrain_embd)
    # char2id, id2char, label2id, id2label = helper.buildMap(None)
    vocab, embd = loadWord2vec(w2v_file=vec_path, embed_file=pretrain_embd_path)
    ner_model = Model(len(char2id), model_path, embd, batch_size=batch_s, seq_len=seq_len, emb_size=emb_size)
    return ner_model
ner_model=load_model()

if __name__ == '__main__':
    test_file = sys.argv[1]
    ner_model=ner_model
    st=time.time()
    l=0
    for i in range(1):
        for line in open(test_file).readlines()[:100]:
            l+=len(line)
            print json.dumps(ner_model.get_ner(line), ensure_ascii=False)
    et=time.time()
    print l/(et-st)

        
