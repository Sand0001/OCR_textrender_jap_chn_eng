#coding=utf8
import os
import sys

dct = {}

dup_cnt = 0
word_dct = {}
char_dct = {}
for line in open('data/chars/chn.txt'):
	chars = line.strip('\r\n ')
	char_dct [chars] = 1

#输入语料，计算语料中的分布
for line in open(sys.argv[1]):
	line = line.strip('\r\n')
	if len(line) == 0:
		continue
	idx = line.index(' ')
	label = line[idx + 1 : ]
	for part in line:
		if part not in char_dct:
			continue
		if part in word_dct:
			word_dct[part] += 1
		else:
			word_dct[part] = 1


balance_word_list = []
#最少也得要200
#最多500
min_freq = 200
max_freq = 500
cnt = 0
for part in word_dct:
	cur_freq = word_dct[part]
	if cur_freq < max_freq:
		cnt += 1
		balance_word_list.append((part, cur_freq))

balance_word_list.sort(	key = lambda x : x[1])
final_word_list = []
final_freq_list = []
total_freq = 0
if cnt > 0:
	#线性回归
	incre = (max_freq - min_freq) / float(cnt)
	for i in range(0, len(balance_word_list)):
		word, cur_freq = balance_word_list[i]
		should_freq = 200 + i * incre - cur_freq
		print (word, cur_freq, should_freq)
		if should_freq < 0:
			should_freq = 0
		final_word_list.append(word)
		final_freq_list.append(should_freq)
		total_freq += should_freq

final_prob_list = [freq / total_freq for freq in final_freq_list]

import numpy as np
with open("cc", "w") as f:
	#for i in range(0, 100000):
	sample_all = np.random.choice(a=final_word_list, size=500000, replace=True, p=final_prob_list)
	for sample in sample_all:
		f.write(sample + '\n')




