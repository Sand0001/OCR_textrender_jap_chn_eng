from textrenderer.corpus.corpus import Corpus
import numpy as np
import random


class EngCorpus(Corpus):
    """
    Load English corpus by words, and get random {self.length} words as result
    """
    def prob(self, probability):
        r = random.randint(0, 100)
        #print ("Prob : ", r)
        if r <= probability * 100:
            return True
        else:
            return False
    def load_subscript(self):
        self.subscript_list = []
        for line in open('./data/corpus/suscripts.dat'):
            parts = line.strip('\r\n ').split(' ')
            if parts[0] not in self.charsets:
                continue
            self.subscript_list.append(parts[0])
        print("Load subscripts List : ", len(self.subscript_list))

    def load(self):
        self.load_corpus_path()
        self.load_subscript()

        for i, p in enumerate(self.corpus_path):
            print("Load {} th eng corpus".format(i))
            with open(p, encoding='utf-8') as f:
                data = f.read()

            lines = data.split('\n')
            for line in lines:
                for word in line.split(' '):
                    word = word.strip()
                    word = ''.join(filter(lambda x: x in self.charsets, word))

                    if word != u'' and len(word) > 2:
                        self.corpus.append(word)
            print("Word count {}".format(len(self.corpus)))

    def get_sample(self, img_index):
        start = np.random.randint(0, len(self.corpus) - self.length)
        words = self.corpus[start:start + self.length]
        word = ' '.join(words)

        return word
    def get_sample_add_script(self,img_index):
        start = np.random.randint(0, len(self.corpus) - self.length)
        words = self.corpus[start:start + self.length]
        word = ' '.join(words)
        if self.prob(1):
            #有一定的几率全大写
            word = word.upper()
        if self.prob(0.03):           #  有一定的几率将word中的字母随机替换成角标
            subscript_index_list = np.random.randint(0,len(word),(np.random.randint(len(word)//2)))
            word = list(word)
            for subscript_index in subscript_index_list:

                word[subscript_index] = np.random.choice(self.subscript_list)
            word = ''.join(word)
        return word,'eng'
