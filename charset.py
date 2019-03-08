#coding=utf8
import os
import sys

dct  = {}

for line in open('data/dataset.txt'):
    line = line.strip('\r\n')
    for c in line:
        if c in dct : 
            dct [c] += 1
        else:
            dct [c] = 1


fout = open('charset', 'w')

for c in dct:
    if dct[c] >= 10:
        fout.write ("%s\t%d\n" %(c, dct[c]))