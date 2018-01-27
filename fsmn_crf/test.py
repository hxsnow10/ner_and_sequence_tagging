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

class Model(object):
    def __init__(self, vocab_size, path_model, embedding, batch_size=1, seq_len=200):
        self.num_steps = seq_len
        self.batch_size = batch_size
        # self.embedding = embedding
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            # self.ner_model = NerModel(vocab_size, embedding, False, batch_size=self.batch_size)
            self.ner_model = NerModelDev(vocab_size, embedding, False, batch_size=self.batch_size)
            # self.ner_model = NerModelTrainEmbd(vocab_size, False, batch_size=self.batch_size)
        config = tf.ConfigProto(device_count={'GPU': 0})
        self.sess = tf.Session(config=config)
        saver = tf.train.Saver()
        print "restore tensorflow model ..."
        saver.restore(self.sess, path_model)
        # char2id, self.id2char, label2id, self.id2label = helper.buildMap(None)
        char2id, self.id2char, label2id, self.id2label = helper.buildMap(None, savemap_file=savemap_file_path,
                                                                         use_pretrain_embd=use_pretrain_embd)
        self.letter2id = helper.buildCharMap()
        self.label2id=label2id
    #@profile
    def predict(self, X, X_words=None):
        viterbi_sequences = []
        # logits, sequence_actual_length, transition_params = self.ner_model.sess.run(
        #     [self.ner_model.logits, self.ner_model.length, self.ner_model.transition_params],
        #     feed_dict={self.ner_model.input_data:X})
        # X_char = helper.wordBatch2charBatch(X, self.id2char, self.letter2id)
        logits, sequence_actual_length, transition_params = self.sess.run(
            [self.ner_model.logits, self.ner_model.length, self.ner_model.transition_params],
            feed_dict={self.ner_model.input_data: X})
        for logit, seq_len in zip(logits, sequence_actual_length):
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


if __name__ == '__main__':
    model_path, test_file = sys.argv[1:3]
    batch_s = 5
    seq_len = 200
    # savemap_file_path = "savedMap_fr_pg.pkl"
    savemap_file_path = "map.pkl"
    pretrain_embd_path = "xhw_char_128.pkl"
    # w2v_path = "ar_tweet_w2v_sg.txt.gz"
    use_pretrain_embd = True
    char2id, id2char, label2id, id2label = helper.buildMap(None, savemap_file=savemap_file_path,
                                                           use_pretrain_embd=use_pretrain_embd)
    # char2id, id2char, label2id, id2label = helper.buildMap(None)
    vocab, embd = loadWord2vec(w2v_file=None, embed_file=pretrain_embd_path)
    ner_model = Model(len(char2id), model_path, embd, batch_size=batch_s, seq_len=seq_len)
    # ner_model = Model(len(char2id), model_path, embedding=None, batch_size=batch_size, seq_len=seq_len)
    # X_train, y_train, X_val, y_val = helper.getTrain(None, test_file, seq_max_len=seq_len,
    #                                                  is_shuffle=False, char2id=char2id, label2id=label2id)
    y_true = []
    from data_utils import NerDataset
    test_data=NerDataset(test_file, char2id, label2id, batch_s, seq_len)
    ner_model.evaluate(test_data)
