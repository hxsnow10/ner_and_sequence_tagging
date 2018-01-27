import math
import helper
import numpy as np
import tensorflow as tf
import random, sys
from tensorflow.contrib.rnn import static_rnn

class BILSTM_CRF(object):
    
    def __init__(self, num_chars, num_classes, num_steps=200, num_epochs=100, embedding_matrix=None, is_training=True, is_crf=True, weight=False):
        # Parameter
        self.max_f1 = 0
        self.last_f = 1
        self.learning_rate = 0.002
        self.dropout_rate = 0.0
        self.batch_size = 1
        self.num_layers = 1   
        self.emb_dim = 100
        self.hidden_dim = 100
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.num_chars = num_chars
        self.num_classes = num_classes
        
        # placeholder of x, y and weight
        self.inputs = tf.placeholder(tf.int32, [None, self.num_steps])
        self.targets = tf.placeholder(tf.int32, [None, self.num_steps])
        self.targets_weight = tf.placeholder(tf.float32, [None, self.num_steps])
        self.targets_transition = tf.placeholder(tf.int32, [None])
        
        # char embedding
        if embedding_matrix != None:
            self.embedding = tf.Variable(embedding_matrix, trainable=False, name="emb", dtype=tf.float32)
        else:
            self.embedding = tf.get_variable("emb", [self.num_chars, self.emb_dim])
        self.inputs_emb = tf.nn.embedding_lookup(self.embedding, self.inputs)
        self.inputs_emb = tf.transpose(self.inputs_emb, [1, 0, 2])
        self.inputs_emb = tf.reshape(self.inputs_emb, [-1, self.emb_dim])
#        self.inputs_emb = tf.split(0, self.num_steps, self.inputs_emb)
        self.inputs_emb = tf.split(self.inputs_emb, self.num_steps, 0)

        # lstm cell
        lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim)
        lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim)

        # dropout
        if is_training:
            lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(lstm_cell_fw, output_keep_prob=(1 - self.dropout_rate))
            lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(lstm_cell_bw, output_keep_prob=(1 - self.dropout_rate))

        lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell_fw] * self.num_layers)
        lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell_bw] * self.num_layers)

        # get the length of each sample
        self.length = tf.reduce_sum(tf.sign(self.inputs), reduction_indices=1)
        self.length = tf.cast(self.length, tf.int32)  
        
        # forward and backward
        self.outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
            lstm_cell_fw, 
            lstm_cell_bw,
            self.inputs_emb, 
            dtype=tf.float32,
            sequence_length=self.length
        )
        
        # softmax
        self.outputs = tf.reshape(tf.concat(self.outputs, 1), [-1, self.hidden_dim * 2])
        self.softmax_w = tf.get_variable("softmax_w", [self.hidden_dim * 2, self.num_classes])
        self.softmax_b = tf.get_variable("softmax_b", [self.num_classes])
        self.logits = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b

        #two crf for different data
        if not is_crf:
            pass
        else:
            self.loss_source, self.max_scores_source, self.max_scores_pre_source = self.crf(self.logits, "source")
            self.loss_target, self.max_scores_target, self.max_scores_pre_target = self.crf(self.logits, "target")

        # summary
        self.train_summary_source = tf.contrib.deprecated.scalar_summary("loss", self.loss_source)
        self.train_summary_target = tf.contrib.deprecated.scalar_summary("loss", self.loss_target)
        self.val_summary_source = tf.contrib.deprecated.scalar_summary("loss", self.loss_source)        
        self.val_summary_target = tf.contrib.deprecated.scalar_summary("loss", self.loss_target)        
        
        self.optimizer_source = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_source) 
        self.optimizer_target = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_target) 

    def crf(self, logits, flag):
        tags_scores = tf.reshape(logits, [self.batch_size, self.num_steps, self.num_classes])
        transitions = tf.get_variable("transitions" + flag, [self.num_classes + 1, self.num_classes + 1])
            
        dummy_val = -1000
        class_pad = tf.Variable(dummy_val * np.ones((self.batch_size, self.num_steps, 1)), dtype=tf.float32)
        observations = tf.concat([tags_scores, class_pad], 2)

        begin_vec = tf.Variable(np.array([[dummy_val] * self.num_classes + [0] for _ in range(self.batch_size)]), trainable=False, dtype=tf.float32)
        end_vec = tf.Variable(np.array([[0] + [dummy_val] * self.num_classes for _ in range(self.batch_size)]), trainable=False, dtype=tf.float32) 
        begin_vec = tf.reshape(begin_vec, [self.batch_size, 1, self.num_classes + 1])
        end_vec = tf.reshape(end_vec, [self.batch_size, 1, self.num_classes + 1])

        observations = tf.concat([begin_vec, observations, end_vec], 1)

        mask = tf.cast(tf.reshape(tf.sign(self.targets),[self.batch_size * self.num_steps]), tf.float32)
            
        # point score
        point_score = tf.gather(tf.reshape(tags_scores, [-1]), tf.range(0, self.batch_size * self.num_steps) * self.num_classes + tf.reshape(self.targets,[self.batch_size * self.num_steps]))
        point_score *= mask
            
        # transition score
        trans_score = tf.gather(tf.reshape(transitions, [-1]), self.targets_transition)
           
        # real score
        target_path_score = tf.reduce_sum(point_score) + tf.reduce_sum(trans_score)
            
        # all path score
        total_path_score, max_scores, max_scores_pre  = self.forward(observations, transitions, self.length)
            
        # loss
        loss = - (target_path_score - total_path_score)
        
        return loss, max_scores, max_scores_pre

    def logsumexp(self, x, axis=None):
        x_max = tf.reduce_max(x, reduction_indices=axis, keep_dims=True)
        x_max_ = tf.reduce_max(x, reduction_indices=axis)
        return x_max_ + tf.log(tf.reduce_sum(tf.exp(x - x_max), reduction_indices=axis))

    def forward(self, observations, transitions, length, is_viterbi=True, return_best_seq=True):
        length = tf.reshape(length, [self.batch_size])
        transitions = tf.reshape(tf.concat([transitions] * self.batch_size, 0), [self.batch_size, 6, 6])
        observations = tf.reshape(observations, [self.batch_size, self.num_steps + 2, 6, 1])
        observations = tf.transpose(observations, [1, 0, 2, 3])
        previous = observations[0, :, :, :]
        max_scores = []
        max_scores_pre = []
        alphas = [previous]
        for t in range(1, self.num_steps + 2):
            previous = tf.reshape(previous, [self.batch_size, 6, 1])
            current = tf.reshape(observations[t, :, :, :], [self.batch_size, 1, 6])
            alpha_t = previous + current + transitions
            if is_viterbi:
                max_scores.append(tf.reduce_max(alpha_t, reduction_indices=1))
                max_scores_pre.append(tf.argmax(alpha_t, dimension=1))
            alpha_t = tf.reshape(self.logsumexp(alpha_t, axis=1), [self.batch_size, 6, 1])
            alphas.append(alpha_t)
            previous = alpha_t           
            
        alphas = tf.reshape(tf.concat(alphas, 0), [self.num_steps + 2, self.batch_size, 6, 1])
        alphas = tf.transpose(alphas, [1, 0, 2, 3])
        alphas = tf.reshape(alphas, [self.batch_size * (self.num_steps + 2), 6, 1])

        last_alphas = tf.gather(alphas, tf.range(0, self.batch_size) * (self.num_steps + 2) + length)
        last_alphas = tf.reshape(last_alphas, [self.batch_size, 6, 1])

        max_scores = tf.reshape(tf.concat(max_scores, 0), (self.num_steps + 1, self.batch_size, 6))
        max_scores_pre = tf.reshape(tf.concat(max_scores_pre, 0), (self.num_steps + 1, self.batch_size, 6))
        max_scores = tf.transpose(max_scores, [1, 0, 2])
        max_scores_pre = tf.transpose(max_scores_pre, [1, 0, 2])

        return tf.reduce_sum(self.logsumexp(last_alphas, axis=1)), max_scores, max_scores_pre        

    def train(self, sess, save_file, X_train_source, y_train_source, X_val_source, y_val_source,
            X_train_target, y_train_target, X_val_target, y_val_target, X_test_source, y_test_source):
        char2id, id2char = helper.loadMap("char2id")
        label2id, id2label = helper.loadMap("label2id")
        
        merged = tf.contrib.deprecated.merge_all_summaries()
        summary_writer_train_source = tf.summary.FileWriter('loss_log/train_loss_source', sess.graph)
        summary_writer_train_target = tf.summary.FileWriter('loss_log/train_loss_target', sess.graph)
        summary_writer_val_source = tf.summary.FileWriter('loss_log/val_loss_source', sess.graph)
        summary_writer_val_target = tf.summary.FileWriter('loss_log/val_loss_target', sess.graph)

        len_source = len(X_train_source)
        len_target = len(X_train_target)
        source_probability = len_source * 1.0 / (len_source + len_target)
        count_source = 0
        count_target = 0
        for epoch in range(self.num_epochs):
            print "current epoch: %d" % (epoch)
            if random.random() <= source_probability:
                sh_index = np.arange(len_source)
                np.random.shuffle(sh_index)
                X_train = X_train_source[sh_index]
                y_train = y_train_source[sh_index]
                self.train_an_iteration(sess, save_file, X_train, y_train, X_val_source, y_val_source, char2id, id2char, label2id, id2label, summary_writer_train_source, summary_writer_train_target, summary_writer_val_source, summary_writer_val_target, flag="source", is_summary=(count_source%10==0), is_validation=(count_source%10==0), X_test_source=X_test_source, y_test_source=y_test_source) 
                count_source += 1
            else :
                sh_index = np.arange(len_target)
                np.random.shuffle(sh_index)
                X_train = X_train_target[sh_index]
                y_train = y_train_target[sh_index]
                self.train_an_iteration(sess, save_file, X_train, y_train, X_val_target, y_val_target, char2id, id2char, label2id, id2label, summary_writer_train_source, summary_writer_train_target, summary_writer_val_source, summary_writer_val_target, flag="target", is_summary=(count_target%10==0), is_validation=(count_target%10==0), X_test_source=X_test_source, y_test_source=y_test_source)

    def train_an_iteration(self, sess, save_file, X_train, y_train, X_val, y_val, char2id, id2char, label2id, id2label, summary_writer_train_source, summary_writer_train_target, summary_writer_val_source, summary_writer_val_target, flag="source", is_summary=False, is_validation=False, X_test_source=None, y_test_source=None):
        saver = tf.train.Saver()
        
        num_iterations = int(math.ceil(1.0 * len(X_train) / self.batch_size))
        
        cnt = 0
        for iteration in range(num_iterations):
            # train, the flag indicate the data source
            X_train_batch, y_train_batch = helper.nextBatch(X_train, y_train, start_index=iteration * self.batch_size, batch_size=self.batch_size)
#            y_train_weight_batch = 1 + np.array((y_train_batch == label2id['B']) | (y_train_batch == label2id['E']), float)
            transition_batch = helper.getTransition(y_train_batch)
                
            _, loss_train, max_scores, max_scores_pre, length, train_summary =\
                sess.run([
                    self.optimizer_source if flag == "source" else self.optimizer_target,
                    self.loss_source if flag == "source" else self.loss_target,
                    self.max_scores_source if flag == "source" else self.max_scores_target,
                    self.max_scores_pre_source if flag == "source" else self.max_scores_pre_target,
                    self.length,
                    self.train_summary_source if flag == "source" else self.train_summary_target
                ], 
                feed_dict={
                    self.targets_transition:transition_batch, 
                    self.inputs:X_train_batch, 
                    self.targets:y_train_batch, 
 #                   self.targets_weight:y_train_weight_batch
                })

##            predicts_train = self.viterbi(max_scores, max_scores_pre, length, predict_size=self.batch_size)
##            if is_summary:
##                cnt += 1
##                presicion_loc, recall_loc, f_loc, presicion_org, recall_org, f_org, presicion_per, recall_per, f_per  = self.evaluate(X_train_batch, y_train_batch, predicts_train, id2char, id2label)
#                if flag == "source":
#                    summary_writer_train_source.add_summary(train_summary, cnt)
#                else:
#                    summary_writer_train_target.add_summary(train_summary, cnt)
                 
##                print "iteration: %5d, %s train loss: %5d, train precision: LOC %.5f, ORG %.5f, PER %.5f, train recall: LOC %.5f, ORG %.5f, PER %.5f, train f1: LOC %.5f, ORG %.5f, PER %.5f" % (iteration, flag, loss_train, presicion_loc, presicion_org, presicion_per, recall_loc, recall_org, recall_per, f_loc, f_org, f_per)
            # validation
    ##        if is_validation:
    ##            X_val_batch, y_val_batch = helper.nextRandomBatch(X_test_source, y_test_source, batch_size=self.batch_size)
#   ##             y_val_weight_batch = 1 + np.array((y_val_batch == label2id['B']) | (y_val_batch == label2id['E']), float)
    ##            transition_batch = helper.getTransition(y_val_batch)
    ##                
    ##            loss_val, max_scores, max_scores_pre, length, val_summary =\
    ##                sess.run([
    ##                    self.loss_source if flag == "source" else self.loss_target,
    ##                    self.max_scores_source if flag == "source" else self.max_scores_target,
    ##                   self.max_scores_pre_source if flag == "source" else self.max_scores_pre_target,
    ##                  self.length,
    ##                 self.train_summary_source if flag == "source" else self.train_summary_target
    ##                ],
    ##                feed_dict={
    ##                    self.targets_transition:transition_batch, 
    ##                    self.inputs:X_val_batch, 
    ##                    self.targets:y_val_batch, 
 #  ##                     self.targets_weight:y_val_weight_batch
    ##                })
     ##           
     ##           predicts_val = self.viterbi(max_scores, max_scores_pre, length, predict_size=self.batch_size)
     ##           presicion_loc, recall_loc, f_loc, presicion_org, recall_org, f_org, presicion_per, recall_per, f_per  = self.evaluate(X_val_batch, y_val_batch, predicts_val, id2char, id2label)
#               if flag == "source":
#                    summary_writer_val_source.add_summary(val_summary, cnt)
#                else:
#                   summary_writer_val_target.add_summary(val_summary, cnt)
            
            presicion_loc, recall_loc, f_loc, presicion_org, recall_org, f_org, presicion_per, recall_per, f_per = self.predictBatch(sess, X_test_source, y_test_source, id2label, id2char, label2id, self.batch_size, "source")
            print "iteration: %5d, %s valid , valid precision: LOC %.5f, ORG %.5f, PER %.5f, valid recall: LOC %.5f, ORG %.5f, PER %.5f, valid f1: LOC %.5f, ORG %.5f, PER %.5f" % (iteration, flag, presicion_loc, presicion_org, presicion_per, recall_loc, recall_org, recall_per, f_loc, f_org, f_per)

            if (f_loc + f_org + f_per >= self.max_f1 and self.last_f >= self.max_f1) or (iteration == 10000):
                self.max_f1 = f_loc + f_org + f_per
                save_path = saver.save(sess, save_file)
                print "saved the best model with f1: %.5f" % (self.max_f1 / 3.0)
            self.last_f = f_loc + f_org + f_per

    def test(self, sess, X_test, X_test_str, output_path):
        char2id, id2char = helper.loadMap("char2id")
        label2id, id2label = helper.loadMap("label2id")
        num_iterations = int(math.ceil(1.0 * len(X_test) / self.batch_size))
        print "number of iteration: " + str(num_iterations)
        with open(output_path, "wb") as outfile:
            for i in range(num_iterations):
                print "iteration: " + str(i + 1)
                results = []
                X_test_batch = X_test[i * self.batch_size : (i + 1) * self.batch_size]
                X_test_str_batch = X_test_str[i * self.batch_size : (i + 1) * self.batch_size]
                if i == num_iterations - 1 and len(X_test_batch) < self.batch_size:
                    X_test_batch = list(X_test_batch)
                    X_test_str_batch = list(X_test_str_batch)
                    last_size = len(X_test_batch)
                    X_test_batch += [[0 for j in range(self.num_steps)] for i in range(self.batch_size - last_size)]
                    X_test_str_batch += [['x' for j in range(self.num_steps)] for i in range(self.batch_size - last_size)]
                    X_test_batch = np.array(X_test_batch)
                    X_test_str_batch = np.array(X_test_str_batch)
                    results = self.predictBatch(sess, X_test_batch, X_test_str_batch, id2label)
                    results = results[:last_size]
                else:
                    X_test_batch = np.array(X_test_batch)
                    results = self.predictBatch(sess, X_test_batch, X_test_str_batch, id2label)
                
                for i in range(len(results)):
                    doc = ''.join(X_test_str_batch[i])
                    outfile.write(doc + "<@>" +" ".join(results[i]).encode("utf-8") + "\n")

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

    def predictBatch(self, sess, X, y_true, id2label, id2char, label2id, batch_size, flag):
        results = []
        print "***********************&"
        for line in X:
            for item in line:
                print item
            print ""
        print "***********************&&"
        for item in y_true:
            print item
        print "***********************&&&"
        length = int(math.ceil(1.0 * len(X) / self.batch_size)) 
        for i in range(length):
##            predict_data = helper.nextBatch_predict([X[i]], i)
             
##            if flag == "source":
##                length, max_scores, max_scores_pre = sess.run([self.length, self.max_scores_source, self.max_scores_pre_source], feed_dict={self.inputs:predict_data})
##            else :
##                length, max_scores, max_scores_pre = sess.run([self.length, self.max_scores_target, self.max_scores_pre_target], feed_dict={self.inputs:predict_data})
##            predicts = self.viterbi(max_scores, max_scores_pre, length, batch_size)
##            results.append(predicts[0])
##      return results
            
            predict_data = helper.nextBatch_for_predict(X, start_index= i * self.batch_size, batch_size=self.batch_size)
            length, max_scores, max_scores_pre = sess.run([self.length, self.max_scores_source, self.max_scores_pre_source], feed_dict={self.inputs:predict_data})
            predicts = self.viterbi(max_scores, max_scores_pre, length, batch_size)
            results.extend(predicts)
#        y_true = []
#        with open("./a", 'r') as file_in:
#            tmp  = []
#            for line in file_in:
#                if line == "\n":
#                    y_true.append(tmp)
#                    tmp = []
#                else :
#                    tmp.append(label2id[line.strip().split()[1]])
        presicion_loc, recall_loc, f_loc, presicion_org, recall_org, f_org, presicion_per, recall_per, f_per = self.evaluate(X, y_true, results, id2char, id2label)
        print presicion_loc, recall_loc, f_loc, presicion_org, recall_org, f_org, presicion_per, recall_per, f_per
        return presicion_loc, recall_loc, f_loc, presicion_org, recall_org, f_org, presicion_per, recall_per, f_per

#            tt = []
#            for j in X[i]:
#                if j != 0:
#                    tt.append(1)
#            for item in predict_data[0]:
#                sys.stdout.write(str(item) + " ")
#            sys.stdout.write("\n")
#            print str(len(predict_data))
#            print str(len(predict_data[0]))
#            print "train length" + " " + str(len(tt))
#            for item in predicts[0]:
#                sys.stdout.write(str(item) + " ")
#            sys.stdout.write("\n")
#            print "result length" + " " + str(len(predicts))
#            print "i" + " " + str(i)
#            return predicts[0]
    
    def getSaver(self, sess):
        saver = tf.train.Saver()
        saver.restore(sess, "./model/model")

    def evaluate(self, X, y_true, y_pred, id2char, id2label):
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
              #  print id2char[y_true[i][j]]
              #  print id2char[y_pred[i][j]]
                if id2label[y_true[i][j]] == "LOC":
                    LOC_true.add(j)
                if id2label[y_pred[i][j]] == "LOC":
                    LOC_pred.add(j)
        
            for j in range(len(y_pred[i])):
                if id2label[y_true[i][j]] == "ORG":
                    ORG_true.add(j)
                if id2label[y_pred[i][j]] == "ORG":
                    ORG_pred.add(j)

            for j in range(len(y_pred[i])):
                if id2label[y_true[i][j]] == "PER":
                    PER_true.add(j)
                if id2label[y_pred[i][j]] == "PER":
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
#        print "true_loc: %.5f, true_org: %.5f, true_per: %.5f, all_pre_loc: %.5f, all_true_loc:%.5f, all_pre_org:%.5f, all_true_org:%.5f, all_pre_per:%.5f, all_true_per:%.5f"%(true_loc, true_org, true_per, all_pre_loc, all_true_loc, all_pre_org, all_true_org, all_pre_per, all_true_per)
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

