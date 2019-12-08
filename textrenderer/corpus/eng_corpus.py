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


class EngCorpus(Corpus):

    def strQ2B(self, ustring):
        """全角转半角"""
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248

            rstring += chr(inside_code)
        return rstring

    def iseng(self, line):
        # 数据很大，就一行，只看前10个
        line = line[0:30]
        alpha_num = 0
        for c in line:
            if c <= 'z' and c >= 'a' or c >= 'A' and c <= 'Z':
                alpha_num += 1
        if alpha_num * 2 >= len(line):
            return True
        return False

    def isalphnum(self, c):
        if c <= 'z' and c >= 'a' or c >= 'A' and c <= 'Z':
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

    def prob(self, probability):
        r = random.randint(0, 100)
        # print ("Prob : ", r)
        if r <= probability * 100:
            return True
        else:
            return False


    def load_subscript(self):
        self.up_subscript_list = []
        self.down_subscript_list = []
        for line in open('./data/corpus/suscripts.dat'):
            parts = line.strip('\r\n ').split(' ')[0]
            #print(parts)
            if parts not in self.charsets:
                continue
            if '▵' in parts:
                self.up_subscript_list.append(parts)
            elif '▿' in parts:
                self.down_subscript_list.append(parts)
        self.scripts_symbol = ['▵+', '▿+', '▵-', '▿-', '▵=', '▿=']
        self.super_scripts_num = ['▵(' + i + '▵)' for i in self.up_subscript_list if
                             ('▿' not in i and i != '▵(' and i != '▵)' and i != '▵=' and i != '▵-' and i != '▵+')]
        self.sub_scripts_num = ['▿(' + i + '▿)' for i in self.down_subscript_list if
                           ('▵' not in i and i != '▿(' and i != '▿)' and i != '▿=' and i != '▿-' and i != '▿+')]
        self.super_scripts_num_1 = [i + '▵+' for i in self.up_subscript_list if
                               ('▿' not in i and i != '▵(' and i != '▵)' and i != '▵=' and i != '▵-' and i != '▵+')]
        self.super_scripts_num_2 = [i + '▵-' for i in self.up_subscript_list if
                               ('▿' not in i and i != '▵(' and i != '▵)' and i != '▵=' and i != '▵-' and i != '▵+')]
        # scripts_single = [i for i in scripts if
        #                   (i != '▵=' and i != '▿=' and i != '▿(' and i != '▿)' and i != '▵(' and i != '▵)')]
        # self.up_scripts_single = [i for i in self.up_subscript_list if i not in self.scripts_symbol]
        # self.down_scripts_single = [i for i in self.down_subscript_list if i not in self.scripts_symbol]
        #print(self.subscript_list)
        print("Load up_subscripts List : ", len(self.up_subscript_list))
        print("Load down_subscripts List : ", len(self.down_subscript_list))
    def load_balanced_sample(self):
        self.single_words_list = []
        for line in open("./data/corpus/eng_single_word.dat"):

            parts = line.strip('\r\n ').split(' ')[0]
            if parts not in self.charsets:
                continue
            self.single_words_list.append(parts)

            parts = line.strip('\r\n ').split(' ')
            if parts[0] not in self.charsets:
                continue
            self.single_words_list.append(parts[0])

        print ("Load Single Word List : ", len(self.single_words_list))
    def load_charset_level(self):
        self.low_charset_level = set()
        self.mid_charset_level = set()
        self.high_charset_level = set()
        for line in open("./data/chars/high_charset"):
            # 本身字符集里面就用空格
            line = line.strip('\r\n')
            idx = line.rindex(' ')
            if idx <= 0:
                continue

            # self.low_charset_level.append()
            # self.mid_charset_level.append()
            self.high_charset_level.add(line[0: idx])
            # self.single_words_list.append(parts[0])
        print("Load high_charset List : ", len(self.high_charset_level))
        for line in open("./data/chars/mid_charset"):
            line = line.strip('\r\n')
            idx = line.rindex(' ')
            if idx <= 0:
                continue
            # self.low_charset_level.append()
            # self.mid_charset_level.append()
            self.mid_charset_level.add(line[0: idx])
            # self.single_words_list.append(parts[0])
        print("Load mid_charset List : ", len(self.mid_charset_level))
        for line in open("./data/chars/low_charset"):
            line = line.strip('\r\n')
            idx = line.rindex(' ')
            if idx <= 0:
                continue

            # self.low_charset_level.append()
            # self.mid_charset_level.append()
            self.low_charset_level.add(line[0: idx])
            # self.single_words_list.append(parts[0])

        print("Load low_charset_level List : ", len(self.low_charset_level))

    def load(self):
        """
        Load one corpus file as one line , and get random {self.length} words as result
        """
        self.load_corpus_path()
        #self.load_charset_level()
        self.load_balanced_sample()
        self.load_subscript()
        self.has_been_created_text = {}
        # 记住这里多个corpus，这样的话，需要
        # self.eng_whitespace_pos_list_dct = {}
        filter_corpus_path = []

        for i in self.corpus_path:
            if 'eng' in i.split('/')[-1]:
            # if 'chn' in i.split('/')[-1] or 'eng' in i.split('/')[-1]:
                filter_corpus_path.append(i)
        self.corpus_path = filter_corpus_path
        for i, p in enumerate(self.corpus_path):

            print_end = '\n' if i == len(self.corpus_path) - 1 else '\r'
            print("Loading eng corpus: {}/{}".format(i + 1, len(self.corpus_path)), end=print_end)
            with open(p, encoding='utf-8') as f:
                data = f.readlines()

            lines = []
            for line in data:
                line_striped = line.strip('\r\n ')
                # line_striped = line.strip()
                if len(line_striped) < 5:
                    continue
                line_striped = line_striped.replace('\u3000', ' ')
                line_striped = line_striped.replace('&nbsp', '')
                line_striped = line_striped.replace("\00", "")
                line_striped = line_striped.replace("()", "")
                line_striped = line_striped.replace("（）", "")
                line_striped = line_striped.replace("[]", "")
                line_striped = line_striped.replace("「」", "")
                if len(line_striped) < 5:
                    continue
                # line_striped = self.strQ2B(line_striped)
                if line_striped != u'' and len(line.strip()) > 1:
                    lines.append(line_striped)

            # 所有行合并成一行
            split_chars = ['']
            if len(lines) > 0:
                if self.iseng(lines[0]):
                    split_chars = [' ']

            splitchar = random.choice(split_chars)
            whole_line = ' '.join(lines)
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
            # print (whole_line[0 : 500])
            if len(whole_line) > self.length:
                # 计算Corpus语言
                # 如果是英文的话，计算一下所有空格的位置
                eng_whitespace_pos_list = []
                language = 'chn'
                if self.iseng(whole_line):
                    language = 'eng'
                    for index in range(0, len(whole_line)):
                        if whole_line[index] == ' ':
                            eng_whitespace_pos_list.append(index)
                # 计算每个稀缺的字的位置
                # self.mid_char_index_dct = {}
                low_char_index_dct = {}
                for index in range(0, len(whole_line)):
                    c = whole_line[index]
                    # 如果不是稀缺字，那么手动886
                    # if c not in self.low_charset_level:
                    #     continue
                    if c in low_char_index_dct:
                        low_char_index_dct[c].append(index)
                    else:
                        low_char_index_dct[c] = [index]

                low_charset_level_list = [e for e in low_char_index_dct]

                corpus = DIPCorpus()
                corpus.content = whole_line
                corpus.eng_whitespace_pos_list = eng_whitespace_pos_list
                corpus.low_char_index_dct = low_char_index_dct
                corpus.low_charset_level_list = low_charset_level_list
                corpus.language = language
                self.corpus.append(corpus)

    # 尝试找到一个完整单词的界限，尽量不要截断单词
    def get_content_of_len_from_pos(self, content, length, pos, max_step=6):
        word = ''
        cur_len = 0
        start = pos
        # rand_len = length  +  (random.randint(0, 8) - 4)
        # length = rand_len
        while cur_len < length and start < len(content):
            c = content[start]
            if self.ischinese(c):
                cur_len += 2
            else:
                cur_len += 1
            word += content[start]
            start += 1
        isalpha = lambda x: x >= 'a' and x <= 'z' or x >= 'A' and x <= 'Z'
        # 如果结尾是个单词，那么往后继续查， 直到找到空格，尽量保证单词的完整性
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

    # 从一个语料中抽取一截
    def choose_line(self, corpus):
        line = corpus.content
        language = corpus.language
        eng_whitespace_pos_list = corpus.eng_whitespace_pos_list
        length = self.length
        # if self.iseng(line):
        # 汉字算长度2，英文算1
        length = 2 * self.length
        length = 20
        ##尝试找到一个完整单词的界限，尽量不要截断单词，最多尝试6步
        max_step = 5

        if language == 'eng':
            pos = np.random.randint(0, len(eng_whitespace_pos_list) - 1)
            start = eng_whitespace_pos_list[pos]
            start += 1
        else:
            start = np.random.randint(0, len(line) - length - max_step)
            length = length + (random.randint(0, 8) - 4)

        return self.get_content_of_len_from_pos(line, length, start, max_step)

    # 判断这个数据是不是高频句子
    def balanced_sample(self, candidate_word, language):
        # 如果汉字全是高频词
        all_high_freq = True
        if language == 'chn':
            for c in candidate_word:
                if c not in self.high_charset_level:
                    all_high_freq = False
                    break
        else:
            return True
        if all_high_freq:
            return False
        return True

    def get_sample(self, img_index):
        # 每次 gen_word，随机选一个预料文件，随机获得长度为 word_length 的字符

        # 补充一下单字，特别是那种频次特别低的单字
        # r = random.randint(0, 30)
        # print (r, len(self.single_words_list))
        if len(self.single_words_list) > 0 and self.prob(0.02):
            word = ''
            for i in range(0, self.length):
                r_i = random.randint(0, len(self.single_words_list) - 1)
                word += self.single_words_list[r_i]
            return word, 'eng'

        corpus = random.choice(self.corpus)
        # print(corpus.content)
        # corpus = self.corpus[1]

        word = self.choose_line(corpus)
        language = corpus.language

        if language == 'eng' and self.prob(0.02):
            # 有一定的几率全大写
            word = word.upper()

            # 有一定的几率首字母大写 TODO
            # if self.prob(0.02):
            #    word

        return word.strip(' '),language

    def get_scripts(self,on_left = False):

        scripts = random.choice([self.down_subscript_list, self.up_subscript_list])
        scripts_word = ['▵s▵t','▵r▵d','▵n▵d','▵t▵h','▵T▵M']
        gen_method = np.random.randint(0, 9)
        if gen_method == 1:
            add_scripts = random.choice(self.sub_scripts_num)
        elif gen_method == 2:  # self.super_scripts_num_1
            add_scripts = random.choice(self.super_scripts_num)
        elif gen_method == 3:
            add_scripts = random.choice(self.super_scripts_num_1)
        elif gen_method == 4:
            add_scripts = random.choice(self.super_scripts_num_2)
        elif gen_method == 5:
            add_scripts = random.choice(scripts_word)

        else:
            add_scripts = ''
            num_list = [1, 1, 2, 2, 3, 4]
            num = random.choice(num_list)  # 随机放置几个连续角标
            one_more_time = True
            for i in range(num):
                tmp_script = np.random.choice(scripts)
                if ')' in tmp_script and ('(' not in add_scripts) and (i != num - 1):
                    continue
                if tmp_script in self.scripts_symbol:
                    one_more_time = False
                if one_more_time:

                    add_scripts += np.random.choice(scripts)
                else:
                    if tmp_script not in self.scripts_symbol:
                        add_scripts += np.random.choice(scripts)
            if add_scripts.count('▵(') % 2 == 1:
                add_scripts += '▵)'
            if add_scripts.count('▿(') % 2 == 1:
                add_scripts += '▿)'

        if on_left:
            add_scripts = add_scripts.replace('▵+','')
            add_scripts = add_scripts.replace('▵-', '')
            add_scripts = add_scripts.replace('▿+', '')
            add_scripts = add_scripts.replace('▿-', '')
            if self.prob(0.85) and '▿' in add_scripts:
                add_scripts = ''
        return add_scripts

    def get_scripts_index_list(self,word_list):
        subscript_index_list = []
        for i in range(np.random.randint(3)):  # 随机取放置角标位置
            tmp_i = np.random.randint(0, len(word_list))
            if tmp_i not in subscript_index_list:  # 避免出现重复位置
                subscript_index_list.append(tmp_i)
        return subscript_index_list

    def get_word_list_index_value(self,word_list,subscript_index):
        if np.random.randint(1, 7) == 1:
            add_scripts = self.get_scripts(on_left=True)
            return add_scripts + word_list[subscript_index]
        else:
            add_scripts = self.get_scripts()
            return word_list[subscript_index] + add_scripts


    def get_sample_add_script(self, img_index):
        # 每次 gen_word，随机选一个预料文件，随机获得长度为 word_length 的字符

        # 补充一下单字，特别是那种频次特别低的单字
        # r = random.randint(0, 30)
        # print (r, len(self.single_words_list))
        if len(self.single_words_list) > 0 and self.prob(0.02):
            word = ''
            for i in range(0, self.length):
                r_i = random.randint(0, len(self.single_words_list) - 1)
                word += self.single_words_list[r_i]
            return word,'eng'

        corpus = random.choice(self.corpus)
        # print(corpus.content)
        # corpus = self.corpus[1]

        word = self.choose_line(corpus)
        language = corpus.language

        if language == 'eng' and self.prob(0.04):
            # 有一定的几率全大写
            word = word.upper()

            # 有一定的几率首字母大写 TODO
            # if self.prob(0.02):
            #    word
        # print (line[0:10], language)
        # word = line[start:start + length]
        # 不能让文本的开始和结束有空格的出现
        if language == 'eng' and self.prob(1):
            #print(language)
            #  有一定的几率将word中的字母随机替换成角标

            if self.prob(0.2):
                word_list = list(word)
                subscript_index_list = []
                for i in range(np.random.randint(3 )):  #随机取放置角标位置
                    tmp_i = np.random.randint(0,len(word_list))
                    if tmp_i not in subscript_index_list:   #避免出现重复位置
                        subscript_index_list.append(tmp_i)

                for subscript_index in subscript_index_list:
                    if subscript_index + 1 < len(word_list):
                        if word_list[subscript_index] != ' ' and word_list[subscript_index + 1] != ' ':  # 前后不能都是空格
                            word_list[subscript_index] = self.get_word_list_index_value(word_list, subscript_index)
                    else:
                        word_list[subscript_index] = self.get_word_list_index_value(word_list, subscript_index )
                word = ''.join(word_list)
            else:

                word_list = word.split(' ')
                subscript_index_list = self.get_scripts_index_list(word_list)

                for subscript_index in subscript_index_list:
                    if word_list[subscript_index] != '':
                        aaa = ',.!;:'
                        if word_list[subscript_index][-1] in aaa :
                            if self.prob(0.1):
                                word_list[subscript_index] = self.get_word_list_index_value(word_list,subscript_index)
                        else:
                            word_list[subscript_index] = self.get_word_list_index_value(word_list, subscript_index)
                word = ' '.join(word_list)
        return word.strip(' '), language

