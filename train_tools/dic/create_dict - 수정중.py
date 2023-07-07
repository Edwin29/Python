from utils.Preprocess import Preprocess
from tensorflow.keras import preprocessing
import pickle

# import corpus
def read_corpus_data(filename):
    with open(filename, 'r', encoding='UTF8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:] # remove Header
    return data

corpus_data1 = read_corpus_data('./corpus.txt')

# extract keyword in corpus and create a dictionary list
p = Preprocess()
dict = []
for c in corpus_data1:
    pos = p.pos(c[1])
    for k in pos:
        dict.append(k[0])

# generate 'word2index' for dictionary
# at first, use OOV (not 0)
tokenizer = preprocessing.text.Tokenizer(oov_token='OOV')
tokenizer.fit_on_texts(dict)
word_index = tokenizer.word_index
print(len(word_index))

# 사전 파일 생성
f = open("chatbot_dict.bin", "wb")
try:
    pickle.dump(word_index, f)
except Exception as e:
    print(e)
finally:
    f.close()
