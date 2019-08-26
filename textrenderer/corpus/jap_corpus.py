import random
import numpy as np

from textrenderer.corpus.corpus import Corpus

class DIPCorpus:
    def __init__(self):
        self.content = ''
        self.language = None
        self.eng_whitespace_pos_list = []
        self.low_char_index_dct = {}
        self.low_charset_level_list = []

    



class JAPCorpus(Corpus):


    def load_chars(self):
        self.chars = set()
        for line in open("./data/chars/japeng.txt"):
            self.chars.add(line.strip('\r\n'))
        #print(self.chars)
        print ("Load JAPENG CHARS : ", len(self.chars))

    def strQ2B(self, ustring):
        """全角转半角"""
        rstring = ""
        for uchar in ustring:
            inside_code=ord(uchar)
            if inside_code == 12288:                              #全角空格直接转换            
                inside_code = 32 
            elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
                inside_code -= 65248

            rstring += chr(inside_code)
        return rstring

    def iseng(self, line):
        #数据很大，就一行，只看前10个
        line = line[0:30]
        alpha_num = 0
        for c in line:
            if c <= 'z' and c >= 'a' or c >= 'A'  and c <= 'Z':
                alpha_num += 1
        if alpha_num * 2 >= len(line):
            return True
        return False


    def isalphnum(self, c):
        if c <= 'z' and c >= 'a' or c >= 'A'  and c <= 'Z':
            return True
        if c <= '9' and c >= '0':
            return True
        if c == '.' or c == ',':
            return True
        return False

    def ischinese(self, word):
        for ch in word:
            if '\u4e00' <= ch <= '\u9fff':
                return True
        return False

    def isjap(self, word):
        jap_num = 0 
        for ch in word:
            #if '\u0800' <= ch <= '\u4e00':
            #    jap_num += 1
            if '\u3040' <= ch <= '\u309F': #Hiragana
                return True
            if '\u30A0' <= ch <= '\u30FF': #Katakana
                return True
            #与汉字重叠了，只看片假名吧
            #if '\u4E00' <= ch <= '\u9FBF': #Kanji
                #jap_num += 1
        return False

    def prob(self, probability):
        r = random.randint(0, 100)
        #print ("Prob : ", r)
        if r <= probability * 100:
            return True
        else:
            return False



    def load_balanced_sample(self):
        self.single_words_list = []
        for line in open("./data/corpus/singleword_jap.dat"):
            parts = line.strip('\r\n ').split(' ')
            self.single_words_list.append(parts[0])
        print ("Load Single Word List : ", len(self.single_words_list))
    def load_subscript(self):
        self.up_subscript_list = []
        self.down_subscript_list = []
        for line in open('./data/corpus/suscripts.dat'):
            parts = line.strip('\r\n ').split(' ')[0]
            # print(parts)
            if parts not in self.charsets:
                continue
            if '▵' in parts:
                self.up_subscript_list.append(parts)
            elif '▿' in parts:
                self.down_subscript_list.append(parts)

        # print(self.subscript_list)
        print("Load up_subscripts List : ", len(self.up_subscript_list))
        print("Load down_subscripts List : ", len(self.down_subscript_list))


    def load(self):
        """
        Load one corpus file as one line , and get random {self.length} words as result
        """

        self.load_chars()
        self.load_balanced_sample()
        self.load_subscript()
        self.load_corpus_path()
        #self.load_balanced_sample()
        filter_corpus_path = []

        for i in self.corpus_path:

            if 'ja' in i.split('/')[-1] or 'eng' in i.split('/')[-1] :
                filter_corpus_path.append(i)
        self.corpus_path = filter_corpus_path
        for i, p in enumerate(self.corpus_path):
            print_end = '\n' if i == len(self.corpus_path) - 1 else '\r'
            print("Loading chn corpus: {}/{}".format(i + 1, len(self.corpus_path)), end=print_end)
            with open(p, encoding='utf-8') as f:
                    data = f.readlines()

            lines = []
            for line in data:
                line_striped = line.strip()
                line_striped = line_striped.replace('\u3000', ' ')
                line_striped = line_striped.replace('&nbsp', '')
                line_striped = line_striped.replace("\00", "")
                line_striped = line_striped.replace("()", "")
                line_striped = line_striped.replace("（）", "")
                line_striped = line_striped.replace("[]", "")
                line_striped = line_striped.replace("「」", "")
                #line_striped = self.strQ2B(line_striped)
                if line_striped != u'' and len(line.strip()) > 1:
                    lines.append(line_striped)

            # 所有行合并成一行
            split_chars = ['']
            splitchar = random.choice(split_chars)
            whole_line = splitchar.join(lines)
            '''
            total_len = 0    
            for line in lines:
                filtered = ''.join(filter(lambda x: x in self.charsets, line))
                #少于10个字的直接PASS
                if len(filtered ) < 10:
                    continue
                self.corpus.append(filtered)
                total_len += len(filtered)

            self.probability = [len(l) / float(total_len) for l in self.corpus]
            '''
            # 在 crnn/libs/label_converter 中 encode 时还会进行过滤
            #whole_line = ''.join([i for i in whole_line if i in self.chars])
            whole_line = ''.join(filter(lambda x: x in self.chars, whole_line))

            
            if len(whole_line) > self.length:
                #self.corpus.append(whole_line)
                eng_whitespace_pos_list = []
                language = 'jap'
                if self.iseng(whole_line):
                    language = 'eng'
                    for index in range(0, len(whole_line)):
                        if whole_line[index] == ' ':
                            eng_whitespace_pos_list.append(index)
                corpus = DIPCorpus()
                corpus.content = whole_line
                corpus.eng_whitespace_pos_list = eng_whitespace_pos_list
                corpus.low_char_index_dct = None
                corpus.low_charset_level_list = None
                corpus.language = language
                self.corpus.append(corpus)

    #尝试找到一个完整单词的界限，尽量不要截断单词
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


    #从一个语料中抽取一截
    def choose_line(self, corpus):
        line = corpus.content
        language = corpus.language
        eng_whitespace_pos_list = corpus.eng_whitespace_pos_list
        length = self.length
        #if self.iseng(line):
        #汉字算长度2，英文算1
        length = 2 * self.length
        ##尝试找到一个完整单词的界限，尽量不要截断单词，最多尝试6步
        max_step = 6

        if language == 'eng':
            pos = np.random.randint(0, len(eng_whitespace_pos_list) - 1)
            start = eng_whitespace_pos_list[pos]
            start += 1
        else:
            start = np.random.randint(0, len(line) - length - max_step)
            length = length  +  (random.randint(0, 8) - 4)

        return self.get_content_of_len_from_pos(line, length, start, max_step)



    def get_sample(self, img_index):
        # 每次 gen_word，随机选一个预料文件，随机获得长度为 word_length 的字符

        #补充一下单字，特别是那种频次特别低的单字
        #r = random.randint(0, 30)
        #print (r, len(self.single_words_list))
        if self.prob(0.02) and len(self.single_words_list) > 0:
            word = ''
            for i in range(0, self.length):
                r_i = random.randint(0, len(self.single_words_list) - 1)   
                word += self.single_words_list[r_i]

            return word, 'jap'

        corpus = random.choice(self.corpus)
        #减少一些英文的比例
        if corpus.language == 'eng' and  self.prob(0.2):
            corpus = random.choice(self.corpus)

        word = self.choose_line(corpus)
        language = corpus.language
        if language == 'eng' and self.prob(0.02):
            #有一定的几率全大写
            word = word.upper()

            #有一定的几率首字母大写 TODO 
            #if self.prob(0.02):
            #    word 
        #print (line[0:10], language)
        #word = line[start:start + length]
        #不能让文本的开始和结束有空格的出现

        return word.strip(' '), language


    def get_sample_add_script(self, img_index):
        # 每次 gen_word，随机选一个预料文件，随机获得长度为 word_length 的字符

        #补充一下单字，特别是那种频次特别低的单字
        #r = random.randint(0, 30)
        #print (r, len(self.single_words_list))
        if self.prob(0.02) and len(self.single_words_list) > 0:
            word = ''
            for i in range(0, self.length):
                r_i = random.randint(0, len(self.single_words_list) - 1)
                word += self.single_words_list[r_i]
            # if self.prob(1):   #0.3的概率随机组合角标
            #     subscript_index_list = np.random.randint(0, len(word), (np.random.randint(len(word) // 2)))
            #     for subscript_index in subscript_index_list:
            #         word = list(word)
            #         word[subscript_index] = np.random.choice(self.subscript_list)
            #         word = ''.join(word)
            return word, 'jap'

        corpus = random.choice(self.corpus)
        #减少一些英文的比例
        if corpus.language == 'eng' and  self.prob(0.2):
            corpus = random.choice(self.corpus)

        word = self.choose_line(corpus)
        language = corpus.language

        if corpus.language == 'jap' and self.prob(0.006) and self.ischinese(word[-1]) :

            str_list_right = '】」。、〕》〉'
            prob = [0.001433581650154878, 0.0036351534700355837, 0.9386887847835548, 0.00325115838517267, 0.040652279650820466, 0.001740777718045209, 0.01059826434221642]
            tmp_word_1 = np.random.choice(list(str_list_right),1,p=prob)

            #tmp_word_1= random.choice(list(str_list_right))
            word = word.strip(' ')+tmp_word_1[0]
        if corpus.language == 'jap' and self.prob(0.005) and self.ischinese(word[0]) :
            str_list_left = '「【《〈〔'
            prob = [0.11397058823529412, 0.025735294117647058, 0.265625, 0.07261029411764706, 0.5220588235294118]
            #str_list_right = '｠！？〉】〕」‘’：“”】]。、'
            tmp_word_1 = np.random.choice(list(str_list_left),1,p=prob)
            #tmp_word_1= random.choice(list(str_list_left))
            word = tmp_word_1[0] + word.strip(' ')
        #print('word',word)
        if language == 'eng' and self.prob(0.02) :
            #有一定的几率全大写
            word = word.upper()

            #有一定的几率首字母大写 TODO
            #if self.prob(0.02):
            #    word
        #print (line[0:10], language)
        #word = line[start:start + length]
        #不能让文本的开始和结束有空格的出现
        if (language == 'eng' and self.prob(0.15)) or (language == 'jap' and self.prob(0.05)):
            # print(language)
            #  有一定的几率将word中的字母随机替换成角标

            word_lsit = list(word)
            subscript_index_list = []
            for i in range(np.random.randint(3 )):
                tmp_i = np.random.randint(0,len(word_lsit))
                if tmp_i not in subscript_index_list:
                    subscript_index_list.append(tmp_i)



            #subscript_index_list = np.random.randint(0, len(word_lsit), (np.random.randint(len(word_lsit) )))
            #word = list(word)

            #print(add_scripts)
            for subscript_index in subscript_index_list:
                num_list = [1, 1, 2, 2, 3, 4]
                num = random.choice(num_list)
                scripts = random.choice([self.down_subscript_list, self.up_subscript_list])
                add_scripts = ''

                for i in range(num):
                    add_scripts += np.random.choice(scripts)
                if word_lsit[subscript_index]!= ' ':
                    word_lsit[subscript_index] = word_lsit[subscript_index] + add_scripts
            word = ''.join(word_lsit)
        return word.strip(' '), language
