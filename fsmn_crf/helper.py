import numpy as np
import csv
import pandas as pd
import os
from load_embedding import loadWord2vec
import pickle
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
#Get the id of the word and the label
def buildMap(train_path="train.in",savemap_file="savedMap.pkl", use_pretrain_embd=True, **kwargs):
    if os.path.isfile(savemap_file):
        print "use exist buildMap:", savemap_file
        with open(savemap_file, 'r') as fr:
            char2id, id2char, label2id, id2label = pickle.load(fr)

    else:
        print "no savedmap file exist, build it ...."
        df_train = pd.read_csv(train_path, delimiter=' ', quoting=csv.QUOTE_NONE, skip_blank_lines=False, encoding='utf-8', header=None, names=["char", "label"])

        chars = sorted(list(set(df_train["char"][df_train["char"].notnull()])))
        labels = sorted(list(set(df_train["label"][df_train["label"].notnull()])))
        char2id = dict(zip(chars, range(1, len(chars) + 1)))
        label2id = dict(zip(labels, range(1, len(labels) + 1)))
        id2char = dict(zip(range(1, len(chars) + 1), chars))
        id2label =  dict(zip(range(1, len(labels) + 1), labels))
        id2char[0] = "<PAD>"
        id2label[0] = "<PAD>"
        char2id["<PAD>"] = 0
        label2id["<PAD>"] = 0
        id2char[len(chars) + 1] = "<NEW>"
        char2id["<NEW>"] = len(chars) + 1

        if use_pretrain_embd:
            print "building map use pretrained embedding."
            vocab, embd = loadWord2vec(**kwargs)
            char2id = dict(zip(vocab, range(len(vocab))))
            id2char = dict(zip(range(len(vocab)), vocab))
            # char2id["<PAD>"] = len(vocab) + 1
            # id2char[len(vocab) + 1] = "<PAD>"
            # id2char[len(vocab) + 2] = "<NEW>"
            # char2id["<NEW>"] = len(vocab) + 2
        with open(savemap_file, 'w') as fw:
            pickle.dump([char2id, id2char, label2id, id2label], fw)
        print "save builded map to:", savemap_file
    #saveMap(id2char, id2label)
    # pretrain_file = 'pretrain_embd.pkl'


    return char2id, id2char, label2id, id2label

#Get next random Batch
def nextRandomBatch(X, y, batch_size=1):
    X_batch = []
    y_batch = []
    for i in range(batch_size):
        index = np.random.randint(len(X))
        X_batch.append(X[index])
        y_batch.append(y[index])
    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)
    return X_batch, y_batch

#Get next batch one by one
def nextBatch(X, y, start_index, batch_size=1):
    last_index = start_index + batch_size
    X_batch = list(X[start_index:min(last_index, len(X))])
    y_batch = list(y[start_index:min(last_index, len(X))])
    if last_index > len(X):
        left_size = last_index - (len(X))
        for i in range(left_size):
            index = np.random.randint(len(X))
            X_batch.append(X[index])
            y_batch.append(y[index])
    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)
    return X_batch, y_batch

def nextBatchWord(X,y,X_words, start_index, batch_size=1):
    last_index = start_index + batch_size
    X_batch = list(X[start_index:min(last_index, len(X))])
    X_words_batch = X_words[start_index:min(last_index, len(X))]
    y_batch = list(y[start_index:min(last_index, len(X))])
    if last_index > len(X):
        left_size = last_index - (len(X))
        for i in range(left_size):
            index = np.random.randint(len(X))
            X_batch.append(X[index])
            y_batch.append(y[index])
            X_words_batch.append(X_words[index])
    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)
    return X_batch, y_batch, X_words_batch

#Get next batch for predict
def nextBatch_for_predict(X, start_index, batch_size=1):
    last_index = start_index + batch_size
    X_batch = list(X[start_index:min(last_index, len(X))])
    if last_index > len(X):
        left_size = last_index - (len(X))
        for i in range(left_size):
            index = np.random.randint(len(X))
            X_batch.append(X[index])
    X_batch = np.array(X_batch)
    return X_batch

# use "0" to padding the sentence
def padding(sample, seq_max_len):
    for i in range(len(sample)):
        if len(sample[i]) < seq_max_len:
            sample[i] += [0 for _ in range(seq_max_len - len(sample[i]))]
    return sample

#Get the train data
def getTrain(dict_path, train_path, train_val_ratio=0.9, use_custom_val=False, seq_max_len=50,
             is_shuffle=False, char2id=None,  label2id=None, get_words=False):
    # char2id, id2char, label2id, id2label = buildMap(dict_path)
    df_train = pd.read_csv(train_path, delimiter=' ', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None,
                           encoding='utf-8', names=["char", "label"])

    # map the char and label into id
    words = []
    words_per_sent = []
    tmp_char_id = []
    for item in df_train.char:
        if unicode(item) == unicode(np.nan):
            tmp_char_id.append(-1)
            if len(words_per_sent) < seq_max_len:
                words_per_sent += ['<PAD>'] * (seq_max_len - len(words_per_sent))
            words.append(words_per_sent)
            words_per_sent = []
        else:
            try:
                words_per_sent.append(item)
                tmp_char_id.append(char2id[item])
            except Exception,e:
                print 'FUCK', e
                tmp_char_id.append(char2id["UNT"])
                # print "unknown char",item, "!"*30
                # tmp_char_id.append(char2id["<PAD>"])

    df_train["char_id"] = tmp_char_id
    # df_train["char_id"] = df_train.char.map(lambda x : -1 if str(x) == str(np.nan) else char2id[x])
    df_train["label_id"] = df_train.label.map(lambda x: 0 if str(x) == str(np.nan) else label2id[x])
    print df_train["label_id"]

    # convert the data in maxtrix
    X, y = prepare(df_train["char_id"], df_train["label_id"], seq_max_len)

    if is_shuffle:
        # shuffle the samples
        num_samples = len(X)
        indexs = np.arange(num_samples)
        np.random.shuffle(indexs)
        X = X[indexs]
        y = y[indexs]
        words = map(words.__getitem__, indexs)

        # split the data into train and validation set
        X_train = X[:int(num_samples * train_val_ratio)]
        y_train = y[:int(num_samples * train_val_ratio)]
        X_val = X[int(num_samples * train_val_ratio):]
        y_val = y[int(num_samples * train_val_ratio):]

        X_train_words = words[:int(num_samples * train_val_ratio)]
        X_val_words = words[int(num_samples * train_val_ratio):]

    else:
        X_train = X
        y_train = y
        X_val = None
        y_val = None
        X_train_words = words
        X_val_words = None
    # print "train size: %d, validation size: %d" %(len(X_train), len(y_val))
    # assert len(words) == len(X)
    if get_words == True:
        return X_train, y_train, X_val, y_val, X_train_words, X_val_words
    else:
        return X_train, y_train, X_val, y_val

# pad or truncate the training data
def prepare(chars, labels, seq_max_len, is_padding=True):
    X = []
    y = []
    tmp_x = []
    tmp_y = []
    for record in zip(chars, labels):
        c = record[0]
        l = record[1]
        # empty line
        if c == -1:
            if len(tmp_x) <= seq_max_len:
                X.append(tmp_x)
                y.append(tmp_y)
            tmp_x = []
            tmp_y = []
        else:
            tmp_x.append(c)
            tmp_y.append(l)
    if is_padding:
        X = np.array(padding(X, seq_max_len))
    else:
        X = np.array(X)
    y = np.array(padding(y, seq_max_len))

    return X, y

def getTransition(y_train_batch):
        transition_batch = []
        for m in range(len(y_train_batch)):
                y = [5] + list(y_train_batch[m]) + [0]
                for t in range(len(y)):
                        if t + 1 == len(y):
                                continue
                        i = y[t]
                        j = y[t + 1]
                        if i == 0:
                                break
                        transition_batch.append(i * 6 + j)
        transition_batch = np.array(transition_batch)
        return transition_batch


def getPredict(predict_words, char2id, seq_max_len=200):
    tmp_char_id = []
    for item in predict_words:
        if unicode(item) == unicode(np.nan):
            tmp_char_id.append(-1)
        else:
            try:
                tmp_char_id.append(char2id[item])
            except:
                tmp_char_id.append(char2id["UNT"])
    # print "helper:tmp_char_id", tmp_char_id
    X = prepare_for_predict(tmp_char_id, seq_max_len)
    # print "helper:X", X

    return X


def prepare_for_predict(chars, seq_max_len, is_padding=True):
    X = []
    tmp_x = []

    for c in chars:
        if c == -1:
            if len(tmp_x) <= seq_max_len:
                X.append(tmp_x)
            tmp_x = []
        else:
            tmp_x.append(c)
    X.append(tmp_x)
    if is_padding:
        X = np.array(padding(X, seq_max_len))
    else:
        X = np.array(X)

    return X

def buildCharMap(lang='ar'):
    chars = []
    if lang == 'ar':
        chars.append(u'<PAD>')  # padding
        start_char = int('0600', 16)
        end_char = int('06ff', 16)
        for i in range(start_char, end_char+1):
            chars.append(unichr(i))
        chars.append(u'UNT')#unknown char
    letter2id = dict(zip(chars, range(len(chars))))
    return letter2id

def getCharId(word, letter2id, word_len=20):
    word = word.decode('utf-8')
    chars = list(word)
    if len(word) > 20:
        # print word
        chars = chars[:word_len]
    lenword = len(chars)
    left_pad = (word_len-lenword)/2
    right_pad = word_len - left_pad - lenword
    chars_new = ['<PAD>']*left_pad
    chars_new += chars
    chars_new += ['<PAD>']*right_pad
    charid = []
    for it in chars_new:
        try:
            charid.append(letter2id[it])
        except:
            charid.append(letter2id['UNT'])
            # print "unknown char found!"
    assert len(charid) == word_len
    return charid

def wordidBatch2charBatch(wordbatch, id2word, letter2id):
    batch_size, seq_len = np.shape(wordbatch)
    word_id_flat = wordbatch.reshape([-1])
    words = map(lambda x:id2word[x], word_id_flat)
    char_id_batch = []
    for word in words:
        if word != '<PAD>':
            word_char_id = getCharId(word, letter2id)
        else:
            word_char_id = [letter2id['<PAD>']]*20
        char_id_batch.append(word_char_id)
    assert len(char_id_batch) == batch_size * seq_len
    return char_id_batch

def wordBatch2charBatch(wordbatch, id2word, letter2id):
    batch_size = len(wordbatch)
    words_flat =[i for it in wordbatch for i in it]
    words_count = len(words_flat)
    char_id_batch = []
    for word in words_flat:
        if word != '<PAD>':
            word_char_id = getCharId(word, letter2id)
        else:
            word_char_id = [letter2id['<PAD>']]*20
        char_id_batch.append(word_char_id)
    assert len(char_id_batch) == words_count
    return char_id_batch
