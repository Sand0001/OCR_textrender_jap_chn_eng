#coding=utf8
import os
import sys
import xml.etree.ElementTree as etree
import os

fp = 'zhwiki-latest-abstract-zh-cn1.xml'
#fp = 'test.xml'
root = etree.parse(fp)
#print("xmldom.parse:", type(root))
elementobj = root.findall("//abstract")
f = open("dataset.txt", 'w') 
for node in elementobj:
    #print (node.text)
    text = node.text
    if text is not None and len(text) >= 40 and text[0] != '|':
        f.write(text + '\n')
f.close()
#print ("getElementsByTagName:", type(subElementObj))