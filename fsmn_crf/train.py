import tensorflow as tf
import helper
import math
from ner import NerModel
import numpy as np
import sys
class Train(object):
    def __init__(self, ner_model, data_path, batch_size, num_iterations, num_epochs):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_iterations = num_iterations
        self.char2id, self.id2char, self.label2id, self.id2label = helper.buildMap(data_path)
        #self.ner_model = NerModel(len(self.char2id), True)         
        self.ner_model = ner_model       

    def train(self, sess, save_file, X_train, y_train, X_val, y_val):
        count = 0
        max_f = -1
        writer = tf.summary.FileWriter('./graphs_ar', sess.graph)
        trainloss = tf.summary.scalar("training_loss", self.ner_model.loss)
        for epoch in range(self.num_epochs): 
            print "current epoch: %d" % (epoch)
            for iteration in range(self.num_iterations):
                print "this iteration is %d"%(iteration)
                X_train_batch, y_train_batch = helper.nextRandomBatch(X_train, y_train, batch_size=self.batch_size)
                transition_batch = helper.getTransition(y_train_batch)
                _, train_loss, loss_train, max_scores, max_scores_pre, length =\
                    sess.run([
                        self.ner_model.optimizer,
                        trainloss,
                        self.ner_model.loss,
                        self.ner_model.max_scores,
                        self.ner_model.max_scores_pre,
                        self.ner_model.length,
                    ],
                    feed_dict={
                        self.ner_model.input_data:X_train_batch,
                        self.ner_model.targets:y_train_batch,
                        self.ner_model.targets_transition:transition_batch
                    })
                print "the loss : %f"%loss_train#, X_train_batch
                writer.add_summary(train_loss, epoch*self.num_iterations+iteration)
                if iteration % 100 == 0:
                  presicion_loc, recall_loc, f_loc, presicion_org, recall_org, f_org, presicion_per, recall_per, f_per = \
                  self.validationBatch(sess, X_val, y_val)
                  print "iteration: %5d, valid , valid precision: LOC %.5f, ORG %.5f, PER %.5f, valid recall: LOC %.5f, ORG %.5f, PER %.5f, valid f1: LOC %.5f, ORG %.5f, PER %.5f" %\
                  (iteration, presicion_loc, presicion_org, presicion_per, recall_loc, recall_org, recall_per, f_loc, f_org, f_per)

                  if f_loc + f_org + f_per >= max_f:
                    max_f = f_loc + f_org + f_per

                    saver = tf.train.Saver()
                    save_path = saver.save(sess, save_file)
                    print "saved the best model with f1: %.5f" % (max_f / 3.0)

                  self.last_f = f_loc + f_org + f_per

    #Validate the data
    def validationBatch(self, sess, X, y_true):
        results = []
        length = int(math.ceil(1.0 * len(X) / self.batch_size))
        for i in range(length):
            predict_data = helper.nextBatch_for_predict(X, start_index= i * self.batch_size, batch_size=self.batch_size)
            length, max_scores, max_scores_pre = sess.run([self.ner_model.length, self.ner_model.max_scores, self.ner_model.max_scores_pre], feed_dict={self.ner_model.input_data:predict_data})
            predicts = self.viterbi(max_scores, max_scores_pre, length, self.batch_size)
            results.extend(predicts)
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
    train_file, save_path = sys.argv[1:3]
    char2id, id2char, label2id, id2label = helper.buildMap(train_file)
    # print char2id
    # sys.exit(0)
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            ner_model = NerModel(len(char2id), True)         
        tf.global_variables_initializer().run()
        train = Train(ner_model, train_file, 1, 500, 100)
        X_train, y_train, X_val, y_val = helper.getTrain(train_file, train_file)
        train.train(session, save_path, X_train, y_train, X_val, y_val)
