import time
import helper
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from BILSTM_CRF import BILSTM_CRF

# python train.py train.in model -v validation.in -c char_emb -e 10 -g 2

parser = argparse.ArgumentParser()
parser.add_argument("train_source_path", help="the path of the source train file")
parser.add_argument("train_target_path", help="the path of the target train file")
parser.add_argument("dict_path", help="the path of dict file path")
parser.add_argument("save_path", help="the path of the saved model")
parser.add_argument("test_source_path", help="the path of the test source path")
parser.add_argument("-v","--val_path", help="the path of the validation file", default=None)
parser.add_argument("-e","--epoch", help="the number of epoch", default=1000000000, type=int)
parser.add_argument("-c","--char_emb", help="the char embedding file", default=None)
parser.add_argument("-g","--gpu", help="the id of gpu, the default is 0", default=0, type=int)

args = parser.parse_args()

train_source_path = args.train_source_path
train_target_path = args.train_target_path
dict_path = args.dict_path
save_path = args.save_path
test_source_path = args.test_source_path
val_path = args.val_path
num_epochs = args.epoch
emb_path = args.char_emb
gpu_config = "/gpu:"+str(args.gpu)
num_steps = 200 # it must consist with the test

start_time = time.time()
print "preparing train and validation data"
helper.get_dict(dict_path)
X_train_source, y_train_source, X_val_source, y_val_source = helper.getTrain(dict_path=dict_path, train_path=train_source_path, val_path=test_source_path, seq_max_len=num_steps)
X_train_target, y_train_target, X_val_target, y_val_target = helper.getTrain(dict_path=dict_path, train_path=train_target_path, val_path=test_source_path, seq_max_len=num_steps)
X_test_source, y_test_source, X_test_source_val, y_test_source_val = helper.getTrain(dict_path=dict_path, train_path=test_source_path, val_path=test_source_path, seq_max_len=num_steps, is_shuffle=False)
char2id, id2char = helper.loadMap("char2id")
label2id, id2label = helper.loadMap("label2id")
num_chars = len(id2char.keys())
num_classes = len(id2label.keys())
print num_chars
print num_classes
if emb_path != None:
	embedding_matrix = helper.getEmbedding(emb_path)
else:
	embedding_matrix = None
print "building model"
config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    with tf.device(gpu_config):
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = BILSTM_CRF(num_chars=num_chars, num_classes=num_classes, num_steps=num_steps, num_epochs=num_epochs, embedding_matrix=embedding_matrix, is_training=True)
        
        print "training model"
        tf.initialize_all_variables().run()
#        saver = tf.train.Saver()
#        saver.restore(sess, "model7_beifen3/model7_v4")

        model.train(sess, save_path, X_train_source, y_train_source, X_val_source, y_val_source, X_train_target, y_train_target, X_val_target, y_val_target, X_test_source, y_test_source)

        print "final best f1 is: %f" % (model.max_f1)
 
        end_time = time.time()
        print "time used %f(hour)" % ((end_time - start_time) / 3600)
