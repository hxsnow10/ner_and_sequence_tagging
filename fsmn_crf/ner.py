#coding:utf-8
import tensorflow as tf
from fsmn import FSMN, BiFSMN, AttbBiFSMN
# from crf import CRF
import helper
class NerModel(object):
    '''
    this is the ner model using the crf and fsmn,. the model has one hidden layer.
    '''
    def __init__(self, vocab_size, embedding, is_training, batch_size=None):
        self.vocab_size = vocab_size
        self.hidden_size = 300
        self.memory_size = 5
        self.batch_size = batch_size#1
        self.num_steps = 200#50
        self.num_classes = 6#5
        self.learning_rate = 0.001
        self.keep_prob = 0.5
        self.trainable = True
        self.input_data = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
        self.targets = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
        self.embedding_placeholder = tf.constant(embedding, dtype=tf.float32)
        
        self.fsmn = FSMN(self.memory_size, self.hidden_size, self.hidden_size)
        self.build_graph(is_training) 

    def build_graph(self, is_training):
        # embedding = tf.get_variable("embedding", [self.vocab_size, self.hidden_size], initializer=tf.random_uniform_initializer(-0.1, 0.1))

        embedding = tf.Variable(self.embedding_placeholder, trainable=self.trainable, name="embedding")
        # embedding = tf.Variable(self.embedding_placeholder, trainable=True, name="embedding")
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)


        # Add dropout after embedding layer
        if is_training:
            inputs = tf.nn.dropout(inputs, self.keep_prob)

        # Claculate FSMN Layer
        outputs0 = self.fsmn(inputs)
        # Relu
        outputs1 = tf.nn.relu(outputs0)

        outputs = tf.reshape(outputs1, [-1, self.hidden_size])
        softmax_w = tf.get_variable("softmax_w", [self.hidden_size, self.num_classes])
        softmax_b = tf.get_variable("softmax_b", [self.num_classes])
        # self.logits = tf.matmul(outputs, softmax_w) + softmax_b
        self.logits = tf.reshape(
            tf.matmul(outputs, softmax_w) + softmax_b,
            shape=[-1, self.num_steps, self.num_classes], name='logits')

        self.length = tf.reduce_sum(tf.sign(self.input_data), axis=1)
        self.length = tf.cast(self.length, tf.int32)

        # #bilstm
        # fw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
        # bw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
        # # inputs_seq = tf.unstack(inputs, self.num_steps, 1)
        # inputs_seq = tf.transpose(inputs, [1, 0, 2])
        # inputs_seq = tf.reshape(inputs_seq, [-1, self.hidden_size])
        # inputs_seq = tf.split(inputs_seq, self.num_steps)
        # outputs0, _, _ = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell, inputs_seq,
        #                                                          dtype=tf.float32, sequence_length=self.length)
        # # rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs, dtype=tf.float32)
        #
        # # Relu
        # outputs1 = tf.nn.relu(outputs0)
        # self.outputs = tf.reshape(tf.concat(outputs1, 1), [-1, self.hidden_size*2])
        # self.softmax_w = tf.get_variable("softmax_w", [self.hidden_size*2, self.num_classes])
        # self.softmax_b = tf.get_variable("softmax_b", [self.num_classes])
        # self.logits = tf.reshape(
        #     tf.matmul(self.outputs, self.softmax_w) + self.softmax_b,
        #     shape=[self.batch_size, self.num_steps, self.num_classes])
        # # self.logits = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b

        
        #Add the crf layer
        # crf = CRF(self.batch_size, self.num_classes, self.num_steps, self.targets, self.length, self.targets_transition)
        # self.loss, self.max_scores, self.max_scores_pre = crf(self.logits)
        # self.training_loss = tf.summary.scalar("training_loss", self.loss)

        #NEW CRF
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self.logits, self.targets, self.length)
        self.loss = tf.reduce_mean(-log_likelihood)

        #Optimizer the loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # self.sess = tf.Session()
        # config = tf.ConfigProto(device_count={'GPU': 0})
        # self.sess = tf.Session(config=config)
        #
        # # init all variable
        # init = tf.global_variables_initializer()
        # self.sess.run(init)
       

class NerModelDev(object):
    def __init__(self,vocab_size, embedding, is_training, batch_size=50, seq_len=200, emb_size=128):
        # NerModel.__init__(self, vocab_size, embedding, is_training, batch_size=None)
        self.mem_size_a = 5
        self.mem_size_c = 5
        
        self.use_char=False
        #character-level cnn param
        self.letter2id = helper.buildCharMap()
        self.char_vocab_size = len(self.letter2id)
        self.char_embed_size = 25
        self.word_length = 20
        self.filter_sizes = [2, 3]#[2, 3, 4]#concat[2,3,4]*20, attention[2,3]*64
        self.num_filter = len(self.filter_sizes)
        self.num_per_filter = 150#20
        self.input_char = tf.placeholder(tf.int32, [None, self.word_length])
        # self.input_char = tf.placeholder(tf.int32, [self.batch_size * self.num_steps, self.word_length])

        self.vocab_size = vocab_size
        self.word_embed_size = emb_size
        self.hidden_size = self.word_embed_size #+ self.num_filter * self.num_per_filter
        self.memory_size = 5
        self.batch_size = batch_size  # 1
        self.num_steps = seq_len  # 50
        self.num_classes = 18  # 5
        self.learning_rate = 0.001
        self.keep_prob = 0.5
        self.trainable = True
        self.input_data = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
        self.inputs=self.input_data
        self.targets = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
        self.embedding_placeholder = tf.constant(embedding, dtype=tf.float32)

        self.fsmn = BiFSMN(self.mem_size_a, self.mem_size_c, self.hidden_size, self.hidden_size)#v4
        # self.fsmn = AttbBiFSMN(self.mem_size_a, self.mem_size_c,self.word_embed_size, self.word_embed_size, self.num_steps)#v9#v5#v1#
        # self.fsmn2 = AttbBiFSMN(self.mem_size_a, self.mem_size_c, self.hidden_size, self.hidden_size)  # v7
        # self.fsmn2 = BiFSMN(self.mem_size_a, self.mem_size_c, self.hidden_size, self.hidden_size)  # v8 -> v9
        self.fsmn2 = AttbBiFSMN(self.mem_size_a, self.mem_size_c, self.word_embed_size, self.word_embed_size, self.num_steps)#v10

        self.build_graph(is_training)
        print self.inputs
        print self.outputs

    def build_graph(self, is_training):
        with tf.name_scope("word-embedding"):
            embedding = tf.Variable(self.embedding_placeholder, trainable=self.trainable, name="embedding")
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # #v1
        # ########char-embedding // word-embedding -> attention-based bifsmn
        # # empoly cnn extract character-level features
        # with tf.name_scope("char-cnn"):
        #     char_feat = self.cnn()
        #     inputs_tmp = tf.reshape(inputs, [-1, self.word_embed_size])
        #     inputs_tmp = tf.concat([inputs_tmp, char_feat], 1)
        #     inputs = tf.reshape(inputs_tmp, [self.batch_size, self.num_steps, self.hidden_size])
        # if is_training:
        #     inputs = tf.nn.dropout(inputs, self.keep_prob)
        #
        # # Claculate FSMN Layer
        # with tf.name_scope("fsmn"):
        #     outputs_0 = self.fsmn(inputs)#attention-based bifsmn
        #
        # #v7 based on v1
        # if is_training:
        #     outputs_0 = tf.nn.dropout(outputs_0, self.keep_prob)
        # with tf.name_scope("fsmn2"):
        #     outputs0 = self.fsmn2(outputs_0)#attention-based bifsmn


        #V3
        # ######cnn -> char_vec concat with word_vec -> drop -> bifsmn
        # #empoly cnn extract character-level features
        # with tf.name_scope("char-cnn"):
        #     char_feat = self.cnn()
        #     inputs_tmp = tf.reshape(inputs, [-1, self.word_embed_size])
        #
        # with tf.name_scope('Attention-mechanism'):
        #     # inputs_tmp = tf.concat([inputs_tmp, char_feat], 1)
        #     W_word = tf.Variable(tf.truncated_normal([1, self.word_embed_size], stddev=0.1))#v4#tf.Variable(tf.constant(0.1))#v3
        #     W_char = tf.Variable(tf.truncated_normal([1, self.word_embed_size], stddev=0.1))#v4#tf.Variable(tf.constant(0.1))#v3
        #     W_0 = tf.Variable(tf.truncated_normal([1, self.word_embed_size], stddev=0.1))#v4#tf.Variable(tf.constant(0.1))#v3
        #     z = tf.sigmoid(tf.multiply(W_0, tf.tanh(tf.add(tf.multiply(W_word,inputs_tmp), tf.multiply(W_char,char_feat)))))
        #     inputs_tmp = tf.add(tf.multiply(z, inputs_tmp), tf.multiply(1-z, char_feat))
        #     inputs = tf.reshape(inputs_tmp, [self.batch_size, self.num_steps, self.hidden_size])
        #
        # if is_training:
        #     inputs = tf.nn.dropout(inputs, self.keep_prob)
        #
        # # Claculate FSMN Layer
        # with tf.name_scope("fsmn"):
        #     outputs0 = self.fsmn(inputs)#attention-based bifsmn V5 #bifsmnV4

        #V2 -> v6 -> v9
        ######## drop -> attention-based bifsmn -> cnn -> concat char_vec and word_vec
        #Add dropout after embedding layer
        if is_training:
            inputs = tf.nn.dropout(inputs, self.keep_prob)

        # Claculate FSMN Layer
        with tf.name_scope("fsmn"):
            inputs0 = self.fsmn(inputs)#attention-based bifsmn
        if self.use_char:
            #empoly cnn extract character-level features
            with tf.name_scope("char-cnn"):
                char_feat = self.cnn()
                inputs_tmp = tf.reshape(inputs0, [-1, self.word_embed_size])
                # inputs_tmp = tf.concat([inputs_tmp, char_feat], 1)#v9
                # outputs_0 = tf.reshape(inputs_tmp, [self.batch_size, self.num_steps, self.hidden_size])#v9

            # #V6 based on v2 ->v10
            with tf.name_scope('Attention-mechanism'):
                # inputs_tmp = tf.concat([inputs_tmp, char_feat], 1)
                W_word = tf.Variable(tf.truncated_normal([1, self.word_embed_size], stddev=0.1))#v4#tf.Variable(tf.constant(0.1))#v3
                W_char = tf.Variable(tf.truncated_normal([1, self.word_embed_size], stddev=0.1))#v4#tf.Variable(tf.constant(0.1))#v3
                W_0 = tf.Variable(tf.truncated_normal([1, self.word_embed_size], stddev=0.1))#v4#tf.Variable(tf.constant(0.1))#v3
                z = tf.sigmoid(tf.multiply(W_0, tf.tanh(tf.add(tf.multiply(W_word,inputs_tmp), tf.multiply(W_char,char_feat)))))
                inputs_atb = tf.add(tf.multiply(z, inputs_tmp), tf.multiply(1-z, char_feat))
                outputs_0 = tf.reshape(inputs_atb, [self.batch_size, self.num_steps, self.hidden_size])
        else:
            outputs_0 = tf.reshape(inputs0, [self.batch_size, self.num_steps, self.hidden_size])

        #v9 based on v6 -> v10
        if is_training:
            outputs_0 = tf.nn.dropout(outputs_0, self.keep_prob)
        with tf.name_scope("fsmn2"):
            outputs0 = outputs_0
            #outputs0 = self.fsmn2(outputs_0)#bifsmn

        # Relu
        outputs1 = tf.nn.relu(outputs0)
        with tf.name_scope("softmax"):
            outputs = tf.reshape(outputs1, [-1, self.hidden_size])
            softmax_w = tf.get_variable("softmax_w", [self.hidden_size, self.num_classes])
            softmax_b = tf.get_variable("softmax_b", [self.num_classes])
            # self.logits = tf.matmul(outputs, softmax_w) + softmax_b
            self.logits = tf.reshape(
                tf.matmul(outputs, softmax_w) + softmax_b,
                shape=[-1, self.num_steps, self.num_classes], name='logits')

            self.length = tf.reduce_sum(tf.sign(self.input_data), axis=1)
            self.length = tf.cast(self.length, tf.int32)

        # NEW CRF
        with tf.name_scope('crf'):
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.targets, self.length)
            self.loss = tf.reduce_mean(-log_likelihood)
        self.outputs=[self.logits, self.length, self.transition_params]
        
        # Optimizer the loss
        if is_training:
            self.epoch=tf.placeholder(tf.float32,None)
            self.learning_rate=self.learning_rate/tf.pow(2.0,self.epoch)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def cnn(self):
        with tf.name_scope("char-cnn-embedding"):
            self.char_cnn_W = tf.Variable(
                tf.random_uniform([self.char_vocab_size, self.char_embed_size], -1.0, 1.0),
                name="cnn-W")
            self.embedded_chars = tf.nn.embedding_lookup(self.char_cnn_W, self.input_char)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.char_embed_size, 1, self.num_per_filter]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_per_filter]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.word_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_per_filter * len(self.filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        return h_pool_flat


