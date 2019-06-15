#-*- coding:utf-8 -*-
import os
import sys
import numpy as np
from imp import reload
from PIL import Image, ImageOps

from keras.layers import Input
from keras.models import Model
# import keras.backend as K
from keras.utils import multi_gpu_model
import dl_resnet_crnn as densenet
#import densenet
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
GPU_NUM = 2
#reload(densenet)

encode_dct =  {}
char_set = open('chn.txt', 'r', encoding='utf-8').readlines()
#char_set = open('japchn.txt', 'r', encoding='utf-8').readlines()
for i in range (0, len(char_set)):
	c = char_set[i].strip('\n')
	encode_dct[c] = i

char_set.append('卍')
#char_set = ''.join([ch.strip('\n') for ch in char_set] + ['卍'])

#characters = ''.join([chr(i) for i in range(32, 127)] + ['卍'])
nclass = len(char_set)


mult_model, basemodel = densenet.get_model(False, 32, nclass)
#input = Input(shape=(32, None, 1), name='the_input')
#y_pred= densenet.dense_cnn(input, nclass)
#basemodel = Model(inputs=input, outputs=y_pred)

#model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)

mp = 'weights_densenet-02-1.29.h5'
modelPath = os.path.join(os.getcwd(), './models/' + mp)
#modelPath = sys.argv[1]
'''
load models
'''

def load_model_weights(modelPath):
	if not os.path.exists(modelPath):
		print ("ERROR Load Model : ", modelPath)
		return None

	mult_model, basemodel = densenet.get_model(False, 32, nclass)
	multi_model = multi_gpu_model(basemodel, gpus=GPU_NUM)
	multi_model.load_weights(modelPath)
	weights = basemodel.get_weights()
	return weights

'''
modelPaths = [
'weights_chn_eng_eroded_v3_resnet-08-1.13.h5',
'weights_chn_eng_eroded_v3_resnet-09-1.21.h5',
'weights_chn_eng_eroded_v3_resnet-10-1.16.h5',
'weights_chn_eng_eroded_v3_resnet-11-1.14.h5',
'weights_chn_eng_eroded_v3_resnet-12-1.20.h5',
'weights_chn_eng_eroded_v3_resnet-13-1.11.h5'
]
'''

modelPaths = [
'weights_chn_eng_eroded_v7_resnet-03-1.14.h5',
'weights_chn_eng_eroded_v7_resnet-04-1.15.h5'
]



weights_list = []
for modelPath in modelPaths:
	weights_list.append(load_model_weights("./models/" + modelPath) )

print (weights_list[0])

new_weights = list()

for weights_list_tuple in zip(*weights_list):
	new_weights.append(
		[np.array(weights_).mean(axis=0)\
			for weights_ in zip(*weights_list_tuple)])


multi_model, basemodel = densenet.get_model(False, 32, nclass)
basemodel.set_weights(new_weights)
basemodel.save("avg_model.h5")



	#basemodel.save(sys.argv[2])
	#basemodel.save("./new_model.h5")
	#basemodel = multi_model
	#model.load_weights(modelPath)

