from data_utils import get_trimmed_glove_vectors, load_vocab, \
    get_processing_word, CoNLLDataset
from model import NERModel
from config import Config

# create instance of config
config = Config()

# load vocabs
vocab_words = load_vocab(config.words_filename)
vocab_tags  = load_vocab(config.tags_filename)
vocab_chars = load_vocab(config.chars_filename)
config.nwords=len(vocab_words)
# get processing functions
processing_word = get_processing_word(vocab_words, vocab_chars,
                lowercase=True, chars=config.chars)
processing_tag  = get_processing_word(vocab_tags, 
                lowercase=False)

# get pre trained embeddings
# embeddings = get_trimmed_glove_vectors(config.trimmed_filename)
embeddings=None

# create dataset
dev   = CoNLLDataset(config.dev_filename, processing_word,
                    processing_tag, config.max_iter)
test  = CoNLLDataset(config.test_filename, processing_word,
                    processing_tag, config.max_iter)
train = CoNLLDataset(config.train_filename, processing_word,
                    processing_tag, config.max_iter)

# build model
model = NERModel(config, embeddings, ntags=len(vocab_tags),
                                     nchars=len(vocab_chars))
model.build()
#x=raw_input('xxxxxxx')
# train, evaluate and interact
# model.train(train, dev, vocab_tags)
import time
start=time.time()
model.evaluate(dev, vocab_tags)
print time.time()-start
#model.interactive_shell(vocab_tags, processing_word, test)

