import random
import numpy as np

from textrenderer.corpus.corpus import Corpus


class ChnCorpus(Corpus):

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
        line = line[0:10]
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


    def load_balanced_sample(self):
        self.single_words_list = []
        for line in open("./data/corpus/singleword.dat"):
            parts = line.split(' ')
            self.single_words_list.append(parts[0])
        print ("Load Single Word List : ", len(self.single_words_list))

    def load(self):
        """
        Load one corpus file as one line , and get random {self.length} words as result
        """
        self.load_corpus_path()
        self.load_balanced_sample()
        self.has_been_created_text = {}
        for i, p in enumerate(self.corpus_path):
            print_end = '\n' if i == len(self.corpus_path) - 1 else '\r'
            print("Loading chn corpus: {}/{}".format(i + 1, len(self.corpus_path)), end=print_end)
            with open(p, encoding='utf-8') as f:
                    data = f.readlines()

            lines = []
            for line in data:
                line_striped = line.strip()
                if len(line_striped) < 5:
                    continue
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
            if len(lines) > 0:
                if self.iseng(lines[0]):
                    split_chars = [' ']
            
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
            whole_line = ''.join(filter(lambda x: x in self.charsets, whole_line))
            #print (whole_line[0 : 500])
            if len(whole_line) > self.length:
                self.corpus.append(whole_line)
            #如果是英文的话，计算一下所有空格的位置
            if self.iseng(whole_line):
                self.eng_whitespace_pos_list = []
                for i in range(0, len(whole_line)):
                    if whole_line[i] == ' ':
                        self.eng_whitespace_pos_list.append(i)
            

    def get_sample(self, img_index):
        # 每次 gen_word，随机选一个预料文件，随机获得长度为 word_length 的字符

        #补充一下单字，特别是那种频次特别低的单字
        r = random.randint(0, 8)
        #print (r, len(self.single_words_list))
        if r == 0 and len(self.single_words_list) > 0:
            word = ''
            for i in range(0, self.length):
                r_i = random.randint(0, len(self.single_words_list) - 1)   
                word += self.single_words_list[r_i]
            #如果已经出现过了，那么Continue掉
            if word in self.has_been_created_text:
                return None
            self.has_been_created_text[word] = 1
            return word, self.iseng(word)

        line = random.choice(self.corpus)
        
        length = self.length
        #if self.iseng(line):
        length = 2 * self.length
        #尝试找到一个完整单词的界限，尽量不要截断单词
        max_step = 6
        if self.iseng(line):
            pos = np.random.randint(0, len(self.eng_whitespace_pos_list) - 1)
            start = self.eng_whitespace_pos_list[pos]
            start += 1
        else:
            start = np.random.randint(0, len(line) - length - max_step)
            length = length  +  (random.randint(0, 8) - 4)
        word = ''
        cur_len = 0
        #rand_len = length  +  (random.randint(0, 8) - 4)
        #length = rand_len
        while cur_len < length and start < len(line):
            c = line[start]
            if self.ischinese(c):
                cur_len += 2
            else:
                cur_len += 1
            word += line[start]
            start += 1
        isalpha = lambda  x: x>= 'a' and x<='z' or x >= 'A' and x <= 'Z'
        #如果结尾是个单词，那么往后继续查， 直到找到空格，尽量保证单词的完整性
        if isalpha(word[len(word) - 1]):
            while cur_len < length + max_step and start < len(line):
                c = line[start]
                if c == ' ':
                    break
                if self.ischinese(c):
                    cur_len += 2
                else:
                    cur_len += 1
                word += line[start]
                start += 1
        word = word.strip(' ')
        if word in self.has_been_created_text:
            return None
        self.has_been_created_text[word] = 1
        language = 'chn'
        if self.iseng(line):
            language = 'eng'
        #print ("Choose Word : [", word , "]" , len(word), language)
        #word = line[start:start + length]
        #不能让文本的开始和结束有空格的出现
        return word.strip(' '), language
