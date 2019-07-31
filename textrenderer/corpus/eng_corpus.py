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
    def load_eng_chars(self):
        self.chars_list = []
        charsTxt_lines = open('chars/eng.txt','r').readlines()
        for line in charsTxt_lines:
            line = line.strip('\n')
            self.chars_list.append(line)
    def load(self):
        self.load_corpus_path()
        self.load_subscript()
        self.load_eng_chars()

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
    def choose_line(self, corpus):
        line = corpus.content
        language = corpus.language
        eng_whitespace_pos_list = corpus.eng_whitespace_pos_list
        length = self.length
        #if self.iseng(line):
        #汉字算长度2，英文算1
        length = 2 * self.length
        ##尝试找到一个完整单词的界限，尽量不要截断单词，最多尝试6步
        max_step = 5

        if language == 'eng':
            pos = np.random.randint(0, len(eng_whitespace_pos_list) - 1)
            start = eng_whitespace_pos_list[pos]
            start += 1
        else:
            start = np.random.randint(0, len(line) - length - max_step)
            length = length  +  (random.randint(0, 8) - 4)

        return self.get_content_of_len_from_pos(line, length, start, max_step)
    def get_content_of_len_from_pos(self, content, length, pos, max_step = 6):
        word = ''
        cur_len = 0
        start = pos
        #rand_len = length  +  (random.randint(0, 8) - 4)
        #length = rand_len
        while cur_len < length and start < len(content):
            c = content[start]
            if self.ischinese(c):
                cur_len += 2
            else:
                cur_len += 1
            word += content[start]
            start += 1
        isalpha = lambda  x: x>= 'a' and x<='z' or x >= 'A' and x <= 'Z'
        #如果结尾是个单词，那么往后继续查， 直到找到空格，尽量保证单词的完整性
        if isalpha(word[len(word) - 1]):
            while cur_len < length + max_step and start < len(content):
                c = content[start]
                if c == ' ':
                    break
                if self.ischinese(c):
                    cur_len += 2
                else:
                    cur_len += 1
                word += content[start]
                start += 1
        word = word.strip(' ')
        return word
    def ischinese(self, word):
        for ch in word:
            if '\u4e00' <= ch <= '\u9fff':
                return True
        return False

    def get_sample_add_script(self,img_index):
        retry_num = 10
        ok = False
        for i in range(retry_num):
            if ok == False:
                ok = True
                start = np.random.randint(0, len(self.corpus) - self.length)
                words = self.corpus[start:start + self.length]
                word = ' '.join(words)
                for w in word:
                    if w not in self.chars_list:
                        #print(word)
                        ok = False
                        break
                if ok == True:
                    break
                else:
                    continue


        #language = self.corpus.language
        # retry_num = 10
        # OK = False
        # for i in range(0, retry_num):
        #     word = self.choose_line(self.corpus)
        #     #print ("try : ", word)
        #     # if word in self.has_been_created_text:
        #     #     #print ("choose already exists : ", word)
        #     #     continue
        #     OK = True
        #     break
        # if False == OK:
        #     #print  ("failed to find sample after tried : ", retry_num)
        #     #raise Exception("Failed to found sample")
        #     return None
        #self.has_been_created_text[word] = 1
        if self.prob(0.03):
            #有一定的几率全大写
            word = word.upper()
        if self.prob(0.03):           #  有一定的几率将word中的字母随机替换成角标
            subscript_index_list = np.random.randint(0,len(word),(np.random.randint(len(word)//2)))
            word = list(word)
            for subscript_index in subscript_index_list:

                word[subscript_index] = np.random.choice(self.subscript_list)
            word = ''.join(word)
        return word,'eng'
