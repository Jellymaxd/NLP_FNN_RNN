import os
from io import open

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self,  path):
        self.dictionary = Dictionary()
        self.train_x, self.train_y = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid_x, self.valid_y = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test_x, self.test_y = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                # skip sentence with less than 8 words
                if len(words)<8:
                    continue
                else:
                    for i, word in enumerate(words):
                        self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            # target ids and context ids
            cidss = []
            tidss=[]
            for line in f:
                words = line.split() + ['<eos>']

                #skip sentence with less than 8 words or reach sentence length limit
                for i,word in enumerate(words):
                    if i+7>=len(words):
                        break
                    cids=[self.dictionary.word2idx[words[i]],self.dictionary.word2idx[words[i+1]]
                    ,self.dictionary.word2idx[words[i+2]],self.dictionary.word2idx[words[i+3]],self.dictionary.word2idx[words[i+4]]
                    ,self.dictionary.word2idx[words[i+5]],self.dictionary.word2idx[words[i+6]]]
                    tids=[self.dictionary.word2idx[words[i+7]]]
                    cidss.append(cids)
                    tidss.append(tids)
        return cidss, tidss
