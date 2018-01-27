import tensorflow as tf
from fsmn import FSMN
from crf import CRF 
class NerModelTrainEmbd(object):
    '''
    this is the ner model using the crf and fsmn,. the model has one hidden layer.
    '''
    def __init__(self, vocab_size, is_training, batch_size=None, label_size=4):
        self.vocab_size = vocab_size
        self.hidden_size = 128 
        self.memory_size = 5
        self.batch_size = batch_size#1
        self.num_steps = 200#50
        self.num_classes = label_size+1#5
        self.learning_rate = 0.001
        self.keep_prob = 0.5
        self.trainable = True
        self.input_data = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
        self.targets = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
        # self.embedding_placeholder = tf.constant(embeding, dtype=tf.float32)
        
        self.fsmn = FSMN(self.memory_size, self.hidden_size, self.hidden_size)
        self.build_graph(is_training) 

    def build_graph(self, is_training):
        embedding = tf.get_variable("embedding", [self.vocab_size, self.hidden_size], initializer=tf.random_uniform_initializer(-0.1, 0.1))

        # embedding = tf.Variable(self.embedding_placeholder, trainable=self.trainable, name="embedding")
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
       
    






