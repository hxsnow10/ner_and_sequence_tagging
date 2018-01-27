#encoding:utf-8
import time
import helper
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from BILSTM_CRF import BILSTM_CRF
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
# python test.py model test.in test.out -c char_emb -g 2

parser = argparse.ArgumentParser()
parser.add_argument("-model_path", help="the path of model file")
parser.add_argument("-test_path", help="the path of test file")
parser.add_argument("-output_path", help="the path of output file")
parser.add_argument("-the_flag", help="the flag of the source")
parser.add_argument("-c","--char_emb", help="the char embedding file", default=None)
parser.add_argument("-g","--gpu", help="the id of gpu, the default is 0", default=0, type=int)
args = parser.parse_args()

dict_path = "data/fr_wiki_polyglot_prepared_all.txt"
model_path = args.model_path
test_path = args.test_path
output_path = args.output_path
the_flag = args.the_flag
gpu_config = "/gpu:"+str(args.gpu)
emb_path = args.char_emb
num_steps = 200 # it must consist with the train

start_time = time.time()

copy = []
with open("./twitter_labeled_test", 'r') as file_in:
    tmp = []
    for line in file_in:
        if line == "\n":
            copy.append(tmp)
            tmp = []
        else :
            line_split = line.strip().split()
            tmp.append(line_split[0])
print "preparing test data"
#X_test, X_test_str = helper.getTest(test_path=test_path, seq_max_len=num_steps)
char2id, id2char = helper.loadMap("char2id")
label2id, id2label = helper.loadMap("label2id")

X = helper.getPredict(test_path, char2id)
num_chars = len(id2char.keys())
num_classes = len(id2label.keys())
if emb_path != None:
	embedding_matrix = helper.getEmbedding(emb_path)
else:
	embedding_matrix = None

y_true = []
with open(test_path, 'r') as file_in:
    tmp  = []
    for line in file_in:
        if line == "\n":
#            if len(tmp) <= 200:
            y_true.append(tmp)
            tmp = []
        else :
            split_tmp = line.strip().split()
            tmp.append(label2id[split_tmp[1]])
y_true = helper.padding(y_true, 200)

X, y_true, X_test_source_val, y_test_source_val = helper.getTrain(dict_path=dict_path, train_path=test_path, val_path="./test.txt", seq_max_len=num_steps, is_shuffle=False)

file_out = open(output_path, 'w')
print "building model"
config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    with tf.device(gpu_config):
#		initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope("model"):
            model = BILSTM_CRF(num_chars=num_chars, num_classes=num_classes, num_steps=num_steps, embedding_matrix=embedding_matrix, is_training=False)
            
        print "loading model parameter"
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        
        print "finish!"
        results = model.predictBatch(sess, X, y_true, id2label, id2char, label2id, 1, the_flag)
        length = len(X)
        for i in range(length):
            rows = len(results[i])
            for j in range(rows):
                file_out.write(copy[i][j] + " " + id2label[results[i][j]] + "\n")
#                file_out.write(id2label[results[i][j]] + "\n")
            file_out.write("\n")
            
#		print "testing"
#		model.test(sess, X_test, X_test_str, output_path)

#		end_time = time.time()
#		print "time used %f(hour)" % ((end_time - start_time) / 3600)
