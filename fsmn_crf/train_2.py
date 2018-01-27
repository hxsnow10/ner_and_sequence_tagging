#coding:utf-8
import tensorflow as tf
import helper
import math
from ner import NerModel, NerModelDev
# from ner_train_embd import NerModelTrainEmbd
import numpy as np
from os import makedirs
import os
import sys
from load_embedding import loadWord2vec
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s %(levelname)s] - %(message)s')
from data_utils import NerDataset
"""
usage:
    python train_2.py train_file_path save_model_path
"""

class Train(object):
    def __init__(self, ner_model, data_path, batch_size=1, num_epochs=100, log_dir='./log'):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.log_dir=log_dir
        # self.char2id, self.id2char, self.label2id, self.id2label = helper.buildMap(data_path)
        '''
        self.char2id, self.id2char, self.label2id, self.id2label = helper.buildMap(data_path,
                                                                            savemap_file=savemap_file_path,
                                                                            use_pretrain_embd=use_pretrain_embd,
                                                                            w2v_file=w2v_path, embed_file=pretrain_embd_path)
        '''
        self.ner_model = ner_model
        self.train_max_patience = 20  # 20次epoch中cross loss都没有减少就停止训练
        self.letter2id = helper.buildCharMap()

    def train(self, sess, save_file, train_data, dev_data, X_train_words=None, X_val_words=None):
        current_patience = 0
        min_cross_loss = 1000
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(self.log_dir, sess.graph)
        trainloss = tf.summary.scalar("training_loss", self.ner_model.loss)
        global_step=0
        for epoch in range(self.num_epochs): 
            # print "current epoch: %d" % (epoch)
            cost_loss = 0.
            #X_w = map(X_train_words.__getitem__, indexs)
            # for iteration in range(self.num_iterations):
            for iteration,(X_train_batch,y_train_batch) in enumerate(train_data):
                print "this iteration is %d"%(iteration)
                global_step+=1
                _, train_loss, loss_train, length =\
                    sess.run([
                        self.ner_model.optimizer,
                        trainloss,
                        self.ner_model.loss,
                        # self.ner_model.max_scores,
                        # self.ner_model.max_scores_pre,
                        self.ner_model.length,
                    ],
                    feed_dict={
                        self.ner_model.input_data:X_train_batch,
                        self.ner_model.targets:y_train_batch,
                        self.ner_model.epoch:epoch,
                        # self.ner_model.input_char:X_char #char cnn
                        # self.ner_model.embedding_placeholder:embedding
                        # self.ner_model.targets_transition:transition_batch
                    })
                # print "the loss : %f"%loss_train#, X_train_batch
                cost_loss += loss_train
                writer.add_summary(train_loss, global_step)
                # if iteration % 100 == 0:
                #     presicion_loc, recall_loc, f_loc, presicion_org, recall_org, f_org, presicion_per, recall_per, f_per = \
                #     self.validationBatch(sess, X_val, y_val)
                #     print "iteration: %5d, valid , valid precision: LOC %.5f, ORG %.5f, PER %.5f, valid recall: LOC %.5f, ORG %.5f, PER %.5f, valid f1: LOC %.5f, ORG %.5f, PER %.5f" %\
                #     (iteration, presicion_loc, presicion_org, presicion_per, recall_loc, recall_org, recall_per, f_loc, f_org, f_per)
                #
                #     if f_loc + f_org + f_per >= max_f:
                #         max_f = f_loc + f_org + f_per
                #
                #         saver = tf.train.Saver()
                #         save_path = saver.save(sess, save_file)
                #         print "saved the best model with f1: %.5f" % (max_f / 3.0)
                #
                #     self.last_f = f_loc + f_org + f_per
            cost_loss /= float(iteration)
            # cross_loss = self.evaluateLoss(sess, X_val, y_val)
            cross_loss = self.evaluateLoss(sess, dev_data)
            logging.info('epoch:{0}, train loss:{1}, cross loss:{2}'.format(epoch, cost_loss, cross_loss))
            if cross_loss < min_cross_loss:
                min_cross_loss = cross_loss
                current_patience = 0
                # save model
                saver.save(sess, save_file)
                logging.info('model has saved to %s!' % save_file)
            else:
                current_patience += 1
                logging.info('no improvement, current patience: %d / %d' %
                      (current_patience, self.train_max_patience))
                if self.train_max_patience and current_patience >= self.train_max_patience:
                    logging.info('\nfinished training! (early stopping, max patience: %d)'
                          % self.train_max_patience)
                    return


    def evaluateLoss(self,sess, dev_data):
        cross_loss = 0.
        for i,(x_v,y_v) in enumerate(dev_data):
            # x_v, y_v = helper.nextBatch(X_val, y_val, start_index=i*self.batch_size, batch_size=self.batch_size)
            loss = sess.run(self.ner_model.loss, feed_dict={
                self.ner_model.input_data: x_v,
                self.ner_model.targets: y_v})
            cross_loss += loss
        return cross_loss / float(i)


    #Validate the data
    def validationBatch(self, sess, X, y_true):
        results = []
        length = int(math.ceil(1.0 * len(X) / self.batch_size))
        for i in range(length):
            predict_data = helper.nextBatch_for_predict(X, start_index= i * self.batch_size, batch_size=self.batch_size)
            # length, max_scores, max_scores_pre = sess.run([self.ner_model.length, self.ner_model.max_scores, self.ner_model.max_scores_pre], feed_dict={self.ner_model.input_data:predict_data})
            # predicts = self.viterbi(max_scores, max_scores_pre, length, self.batch_size)
            # results.extend(predicts)
            logits, sequence_actual_length, transition_params = sess.run(
                [self.ner_model.logits, self.ner_model.length, self.ner_model.transition_params], feed_dict={self.ner_model.input_data:predict_data})
            for logit, seq_len in zip(logits, sequence_actual_length):
                logit_actual = logit[:seq_len]
                viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                    logit_actual, transition_params)
                # viterbi_sequences.append(viterbi_sequence)
                results.append(viterbi_sequence)

        presicion_loc, recall_loc, f_loc, presicion_org, recall_org, f_org, presicion_per, recall_per, f_per = self.evaluate(X, y_true, results)
        print "presicion_loc:%f, recall_loc:%f, f_loc:%f, presicion_org:%f, recall_org:%f, f_org:%f, presicion_per:%f, recall_per:%f, f_per:%f"%\
        (presicion_loc, recall_loc, f_loc, presicion_org, recall_org, f_org, presicion_per, recall_per, f_per)
        return presicion_loc, recall_loc, f_loc, presicion_org, recall_org, f_org, presicion_per, recall_per, f_per

    def viterbi(self, max_scores, max_scores_pre, length, predict_size=128):
        best_paths = []
        for m in range(predict_size):
            path = []
            last_max_node = np.argmax(max_scores[m][length[m]])
            # last_max_node = 0
            for t in range(1, length[m] + 1)[::-1]:
                last_max_node = max_scores_pre[m][t][last_max_node]
                path.append(last_max_node)
            path = path[::-1]
            best_paths.append(path)
        return best_paths

    # Evaluation the data
    def evaluate(self, X, y_true, y_pred): 
        presicion_loc = presicion_org = presicion_per = -1 
        recall_loc = recall_org = recall_per = -1 
        f_loc = f_org = f_per = -1 
        all_pre_loc = all_pre_org = all_pre_per = 0 
        all_true_loc = all_true_org = all_true_per = 0 
        true_loc = true_org = true_per = 0 
        for i in range(len(X)): 
            LOC_true = set() 
            LOC_pred = set() 
            ORG_true = set() 
            ORG_pred = set() 
            PER_true = set() 
            PER_pred = set() 
            for j in range(len(y_pred[i])): 
                if self.id2label[y_true[i][j]] == "LOC": 
                    LOC_true.add(j) 
                if self.id2label[y_pred[i][j]] == "LOC": 
                    LOC_pred.add(j) 
         
            for j in range(len(y_pred[i])): 
                if self.id2label[y_true[i][j]] == "ORG": 
                    ORG_true.add(j) 
                if self.id2label[y_pred[i][j]] == "ORG": 
                    ORG_pred.add(j) 
 
            for j in range(len(y_pred[i])): 
                if self.id2label[y_true[i][j]] == "PER": 
                    PER_true.add(j) 
                if self.id2label[y_pred[i][j]] == "PER": 
                    PER_pred.add(j) 
 
            #the recall of LOC 
            true_loc += len(LOC_true & LOC_pred) 
            all_pre_loc += len(LOC_pred) 
            all_true_loc += len(LOC_true) 
 
            true_org += len(ORG_true & ORG_pred) 
            all_true_org += len(ORG_true) 
            all_pre_org += len(ORG_pred) 
             
            true_per += len(PER_true & PER_pred) 
            all_pre_per += len(PER_pred) 
            all_true_per += len(PER_true) 
        if all_pre_loc != 0 and all_true_loc != 0: 
            presicion_loc = true_loc * 1.0 / all_pre_loc 
            recall_loc = true_loc * 1.0 / all_true_loc 
            if presicion_loc + recall_loc != 0: 
                f_loc = 2 * presicion_loc * recall_loc / (presicion_loc + recall_loc) 
 
        if all_pre_org != 0 and all_true_org != 0: 
            presicion_org = true_org * 1.0 / all_pre_org 
            recall_org = true_org * 1.0 / all_true_org 
            if presicion_org + recall_org != 0: 
                f_org = 2 * presicion_org * recall_org / (presicion_org + recall_org) 
 
        if all_pre_per != 0 and all_true_per != 0: 
            presicion_per = true_per * 1.0 / all_pre_per 
            recall_per = true_per * 1.0 / all_true_per 
            if presicion_per + recall_per != 0: 
                f_per = 2 * presicion_per * recall_per / (presicion_per + recall_per) 
 
        return presicion_loc, recall_loc, f_loc, presicion_org, recall_org, f_org, presicion_per, recall_per, f_per          

if __name__ == "__main__":
    train_file, dev_file, save_path, log_dir = sys.argv[1:5]
    print "train data path:", train_file
    print "save mode path:", save_path
    # savemap_file_path = "savedMap_fr_pg.pkl"
    if not os.path.exists(save_path):
        makedirs(save_path)
    save_path=os.path.join(save_path,"model")
    savemap_file_path = "map.pkl"
    emb_size=300
    pretrain_embd_path = "xhw_char_{}.pkl".format(emb_size)
    w2v_path = "xhw_char_{}.vec".format(emb_size)#"ar_tweet_w2v_sg.txt.gz"
    use_pretrain_embd = True
    char2id, id2char, label2id, id2label = helper.buildMap(train_file,
                                                           savemap_file=savemap_file_path,
                                                           use_pretrain_embd=use_pretrain_embd,
                                                           w2v_file=w2v_path, embed_file=pretrain_embd_path)
    # char2id, id2char, label2id, id2label = helper.buildMap(train_file)
    vocab, embedding = loadWord2vec(w2v_file=w2v_path, embed_file=pretrain_embd_path)
    seq_len = 200
    batch_s = 128#64
    logging.info(label2id)
    logging.info('len(char2id): %d' % len(char2id))
    # sys.exit(0)
    # config = tf.ConfigProto(device_count = {'GPU': 0})
    train_data=NerDataset(train_file, char2id, label2id, batch_s, seq_len)
    dev_data=NerDataset(dev_file, char2id, label2id, batch_s, seq_len)
    for k,x in enumerate(dev_data):
        for s in x:
            print s.shape
        if k>=3:
            break
    config = tf.ConfigProto(device_count={'GPU': 0})
    config=None
    with tf.Graph().as_default(), tf.Session(config=config) as session:
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            # ner_model = NerModel(len(char2id), embedding, True, batch_size=batch_s)
            ner_model = NerModelDev(len(char2id), embedding, True, batch_size=batch_s, seq_len=seq_len, emb_size=emb_size)
            # ner_model = NerModelTrainEmbd(len(char2id), True, batch_size=batch_s)
        tf.global_variables_initializer().run()
        train = Train(ner_model, train_file,batch_size=batch_s, num_epochs=seq_len, log_dir=log_dir)#batch_
        # X_train, y_train, X_val, y_val, X_train_words, X_val_words = helper.getTrain(None, train_file, seq_max_len=seq_len,
        #                                                          char2id=char2id, label2id=label2id, get_words=False)
        # train.train(session, save_path, X_train, y_train, X_val, y_val)
        train.train(session, save_path, train_data, dev_data)
