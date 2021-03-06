import sys
import math
import random
import numpy as np
import cv2
from PIL import ImageFont, Image, ImageDraw
from tenacity import retry
import time
import libs.math_utils as math_utils
from libs.utils import draw_box, draw_bbox, prob, apply
from libs.timer import Timer
from textrenderer.liner import Liner
from textrenderer.noiser import Noiser
import libs.font_utils as font_utils
from textrenderer.background_generator import BackgroundGenerator
import re
import os
# noinspection PyMethodMayBeStatic
from textrenderer.remaper import Remaper
import traceback
import matplotlib.pyplot as plt

#import pysnooper

class Renderer(object):
    def __init__(self, corpus, fonts, bgs,fgs, cfg, width=256, height=32,
                 clip_max_chars=False, debug=False, gpu=False, strict=True):
        self.corpus = corpus
        self.fonts = fonts
        self.bgs = bgs
        self.fgs = fgs
        self.out_width = width
        self.out_height = height
        self.clip_max_chars = clip_max_chars
        self.max_chars = math.floor(width / 4) - 1
        self.debug = debug
        self.gpu = gpu
        self.strict = strict
        self.cfg = cfg

        self.timer = Timer()
        self.liner = Liner(cfg)
        self.noiser = Noiser(cfg)
        self.remaper = Remaper(cfg)

        self.create_kernals()
        self.p1 = self.polyfit()

        if self.strict:

            self.font_unsupport_chars = font_utils.get_unsupported_chars(self.fonts, corpus.chars_file)

        self.show = False

        self.showEffect = True
        self.random_symbel = True

    def polyfit(self):
        x = np.array([i for i in range(22, 48) if i % 2 == 0])
        y1 = np.array([12, 15, 16, 17, 19, 20, 21, 22, 23, 26, 27, 28, 29])
        f1 = np.polyfit(x, y1, 2)
        p1 = np.poly1d(f1)
        return p1

    def start(self):
        return time.time()
    
    def end(self, t , msg = ""):
        return
        #print(msg + " took {:.3f}s".format(time.time() - t))

    def prob(self, probability):
        r = random.randint(0, 100)
        #print ("Prob : ", r)
        if r <= probability * 100:
            return True
        else:
            return False
    def plt_show_list (self,word_img, text_box_pnts_list = None, title = None):
        test_img = np.clip(word_img, 0., 255.)
        i = 0
        if text_box_pnts_list is not None:
            for text_box_pnts in text_box_pnts_list:

                test_img = draw_box(test_img, text_box_pnts, (0, 255, i * 255 ))
                i += 1
        #print (test_img)
        test_img = Image.fromarray(test_img.astype('uint8')).convert('RGB')
        if title is not None:
            plt.title(title,fontsize='large',fontweight='bold')
        plt.imshow(test_img)
        plt.show()


    def plt_show (self,word_img, text_box_pnts = None, title = None):
        test_img = np.clip(word_img, 0., 255.)
        if text_box_pnts is not None:
            test_img = draw_box(test_img, text_box_pnts, (0, 255, 155))
        #print (test_img)
        test_img = Image.fromarray(test_img.astype('uint8')).convert('RGB')
        if title is not None:
            plt.title(title,fontsize='large',fontweight='bold')
        plt.imshow(test_img)
        plt.show()

    def stretch_img_w(self, img, text_box_pnts):
        '''
        宽度方向进行拉伸
        '''
        min = self.cfg.stretch.min
        max = self.cfg.stretch.max
        scale = np.random.uniform(min, max)


        img = cv2.resize(img, None, fx=scale, fy=1.0, interpolation=cv2.INTER_CUBIC)
        text_box_pnts[0][0] = int(text_box_pnts[0][0] * scale)
        text_box_pnts[1][0] = int(text_box_pnts[1][0] * scale)
        text_box_pnts[2][0] = int(text_box_pnts[2][0] * scale)
        text_box_pnts[3][0] = int(text_box_pnts[3][0] * scale)
        return img, text_box_pnts

    def find_binary_threth(self,word_img):
        num = 0
        # word_img = cv2.GaussianBlur(word_img, (3, 5), 0)
        word_img_mean = word_img.mean()
        if word_img_mean > 200:
            bg_type = 'white'
        elif word_img_mean < 100:
            bg_type = 'black'
        else:
            print('分不清背景  None')
            return None
        word_img_copy = word_img.copy()
        threth = None
        gama = random.uniform(0.1,0.3)

        if bg_type == 'white':
            bg_pixel_num = len(np.where(word_img == 255)[0])
            fg_pixel_num = word_img.shape[0] * word_img.shape[1] - bg_pixel_num
            word_img_copy[word_img_copy == 255] = 0
            word_img_mean = word_img_copy.sum() / fg_pixel_num
            for j in range(int(word_img_mean)):
                i = int(word_img_mean) - j
                if num < (fg_pixel_num) * gama:
                    num += len(np.where(word_img == i)[0])
                else:
                    threth = i
                    break
        else:
            bg_pixel_num = len(np.where(word_img == 0)[0])
            fg_pixel_num = word_img.shape[0] * word_img.shape[1] - bg_pixel_num

            word_img_mean = word_img_copy.sum() / fg_pixel_num
            for j in range(int(word_img_mean), 256):
                i = j

                if num < (fg_pixel_num) * gama:
                    num += len(np.where(word_img == i)[0])
                else:
                    threth = i
                    break
        return threth

    def split_thin(self, word_img, mask_img, bg, lock=None):
        word_img = np.array(word_img)
        mask_img = np.clip(mask_img, 0., 255.).astype(np.int16)
        mask_img_no_blur = mask_img.copy()
        mask_img = self.apply_gauss_blur(mask_img, ks=[3,5], lock=lock)
        threth = self.find_binary_threth(mask_img)
        word_img[mask_img > threth] = bg[mask_img > threth]

        return word_img


    def gen_img(self, img_index):
        t = self.start()
        lock = None

        word, font, word_size,font_little,language,font_name = self.pick_font(img_index)
        #word = 't_he age-adjusted incidence undefined'

        self.end(t, "pick_font")
        self.dmsg("after pick font")
        self.dmsg ("***********************")
        self.dmsg ("Text : ", word)
        # Background's height should much larger than raw word image's height,
        # to make sure we can crop full word image after apply perspective
        t = self.start()
        if self.show:
            print ("Word_Size :", word_size,' WORD:', word)
        #如果Wordsize特别小，乘以一个系数也是有问题的

        bg = self.gen_bg(width=word_size[0] + 280, height=word_size[1]  +  96, lock = lock)

        if ('▵' in word) or ('▿' in word):
        #if apply(self.cfg.add_script):
            word_img, text_box_pnts, word_color = self.draw_add_script_text_on_bg(word, font, bg,font_little)
            sp_symbol = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩','®','©','*','∞','※']
            for sp in sp_symbol:
                word = word.replace('▿'+sp,'')
                word = word.replace('▵'+sp,'')


        else:
            word_img, text_box_pnts, word_color,word = self.draw_text_on_bg(word, font, bg,language,lock = lock)

        if self.show:
            print ("BG SHAPE : ", bg.shape)
            print ("Word Image : ", word_img.shape)
            print ("text_box_pnts : ", text_box_pnts)
        self.end(t, "gen_bg & draw_text_on_bg : " + str(word_size))
        #print ("Before Apply", word_size, word_img.shape)
        self.dmsg("After draw_text_on_bg")
        t = self.start()
        if (apply(self.cfg.stretch)):
            word_img, text_box_pnts = self.stretch_img_w(word_img, text_box_pnts)
        if apply(self.cfg.crop):
            text_box_pnts = self.apply_crop(text_box_pnts, self.cfg.crop)
        self.end(t, "apply crop ")
        if self.show:
            self.plt_show(word_img, text_box_pnts, title = "before line")
        if apply(self.cfg.line):
            word_img, text_box_pnts = self.liner.apply(word_img, text_box_pnts, word_color,word,font)
            self.dmsg("After draw line")
        if self.show:
            self.plt_show(word_img, text_box_pnts, title = "after line")
        #print ("After Apply Line", text_box_pnts, word_img.shape, type(word_img))
        #test_image = draw_box(word_img, text_box_pnts, (0, 255, 155))
        #plt.imshow(test_image)
        #plt.show()
        if apply(self.cfg.seamless_clone):  #seal 做前景融合
            word_img = self.apply_seamless_cloe_add_foreground(word_img)
        if self.debug:
            word_img = draw_box(word_img, text_box_pnts, (0, 255, 155))
        if self.show:
            print('text_box_pnt_before Transform',text_box_pnts)
            self.plt_show(word_img, text_box_pnts, title = "before Transform")

        if apply(self.cfg.curve):
            word_img, text_box_pnts = self.remaper.apply(word_img, text_box_pnts, word_color)
            self.dmsg("After remapping")

        if self.debug:
            word_img = draw_box(word_img, text_box_pnts, (155, 255, 0))

        t = self.start()
        #print ("Before transform word_img.shape", word_img.shape)
        # if isinstance(word, str):
        
        word_img, bg_pnts_transformed, text_box_pnts_transformed = \
            self.apply_perspective_transform(word_img, text_box_pnts,
                                             max_x=self.cfg.perspective_transform.max_x,
                                             max_y=self.cfg.perspective_transform.max_y,
                                             max_z=self.cfg.perspective_transform.max_z,
                                             gpu=self.gpu)
        if self.show:
            print ("text_box_pnts_transformed : ", text_box_pnts_transformed)
            print ("img_pnts_transformed : ", bg_pnts_transformed)
            self.plt_show(word_img, text_box_pnts_transformed, title= "After Transform")
            self.plt_show(word_img,bg_pnts_transformed, title = "After Transform img_pnts")
        self.end(t, "apply apply_perspective_transform ")
        #plt.imshow(word_img)
        #plt.show()
        self.dmsg("After perspective transform")
        t = self.start()

        if self.debug:
            #bg_pnts_transformed表示的是，背景的四个顶点，在Transform后，图片会扩大，背景色为黑色，必须要限定裁剪在背景范围内，否则会出现多余的黑色
            _, crop_bbox = self.crop_img(word_img, text_box_pnts_transformed,bg_pnts_transformed)
            word_img = draw_bbox(word_img, crop_bbox, (255, 0, 0))
        else:
            #all bad comes from here, why leaving some padding?
            word_img, crop_bbox = self.crop_img(word_img, text_box_pnts_transformed, bg_pnts_transformed)
        #if apply(self.cfg.seamless_clone):
        #print('word_img.shape',word_img.shape)

        if self.show:
            print ("AFTER CROP")
            #左下, 右下, 右上，左上
            startx = crop_bbox[0]
            starty = crop_bbox[1]
            xwidth = crop_bbox[2]
            yheight = crop_bbox[3]
            text_box_pnts = [[startx, starty], [startx + xwidth, starty], [startx + xwidth, starty + yheight], [startx, starty + yheight]]
            self.plt_show(word_img, text_box_pnts, title = "After Crop")
            self.end(t, "apply crop_img ")
            self.dmsg("After crop_img")

        if apply(self.cfg.noise):
            word_img = np.clip(word_img, 0., 255.)
            word_img = self.noiser.apply(word_img)
            if self.show:
                self.plt_show(word_img, title = 'After noiser')
            self.dmsg("After noiser")

        blured = False
        if apply(self.cfg.blur):
            if not (('▵' in word) or ('▿' in word)):
                blured = True
                word_img = self.apply_blur_on_output(word_img, lock)
                self.dmsg("After blur")
                if self.show:
                    self.plt_show(word_img, title = 'After blur')

        prydown_scale = 1.0
        if not blured:
            if apply(self.cfg.prydown):
                word_img, prydown_scale = self.apply_prydown(word_img)
                self.dmsg("After prydown")
                if self.show:
                    self.plt_show(word_img, title = 'After prydown')
       
        t = self.start()
        word_img = np.clip(word_img, 0., 255.)
        #print (word_img.shape)
        isReversed = False
        if apply(self.cfg.reverse_color):
            self.dmsg ("-*- APPLY reverse_color ")
            word_img = self.reverse_img(word_img)
            isReversed = True
            self.dmsg("After reverse_img")
            if self.show:
                self.plt_show(word_img, title = 'After reverse_color')

        if apply(self.cfg.emboss):
            self.dmsg ("-*- APPLY emboss ")
            word_img = self.apply_emboss(word_img)
            self.dmsg("After emboss")
            if self.show:
                self.plt_show(word_img, title = 'After emboss')
        #如果resize的太过分了，就别sharp和erode了
        if apply(self.cfg.sharp) and prydown_scale < 1.3:
            self.dmsg ("-*- APPLY sharp ")
            #if not isReversed:
            #    word_img = self.reverse_img(word_img)
            word_img = self.apply_sharp(word_img)
            #if not isReversed:
            #    word_img = self.reverse_img(word_img)
            self.dmsg("After sharp")
            if self.show:
                self.plt_show(word_img, title = 'After sharp')

        if apply(self.cfg.erode) and prydown_scale < 1.3:

            word_img = self.add_erode(word_img,font,word)
            if self.show:
                self.plt_show(word_img, title = 'After erode')

        if apply(self.cfg.dilate):
            self.dmsg ("-*- APPLY dilate ")
            word_img = self.add_dilate(word_img)

        #word_img = cv2.resize(word_img, (self.out_width, self.out_height), interpolation=cv2.INTER_CUBIC)
        if self.show:
            self.plt_show(word_img, title = 'After Resize')

        self.end(t, "apply2 ")
        self.dmsg ("***********************END*****************")
        return word_img, word,font_name

    def dmsg(self, *msg):
        if self.debug:
            print(msg)

    def random_xy_offset(self, src_height, src_width, dst_height, dst_width):
        """
        Get random left-top point for putting a small rect in a large rect.
        Normally dst_height>src_height and dst_width>src_width
        """
        y_max_offset = 0
        #20%的样本, 增加1像素的扰动
        #if (prob(0.2)):
        max_step_create_rnd_crop = 2
        #else:
        #max_step_create_rnd_crop = 0
        if dst_height > src_height:
            y_max_offset = dst_height - src_height + max_step_create_rnd_crop

        x_max_offset = 0
        if dst_width > src_width:
            x_max_offset = dst_width - src_width + max_step_create_rnd_crop

        y_offset = 0
        if y_max_offset >= 0:
            y_offset = random.randint(0, y_max_offset) - max_step_create_rnd_crop // 2

        x_offset = 0
        if x_max_offset >= 0:
            x_offset = random.randint(0, x_max_offset) - max_step_create_rnd_crop // 2
        #print("OFFSET : ", x_offset, y_offset, x_max_offset, y_max_offset)
        return x_offset, y_offset

    def crop_img(self, img, text_box_pnts_transformed, bg_pnts_transformed):
        """
        Crop text from large input image
        :param img: image to crop
        :param text_box_pnts_transformed: text_bbox_pnts after apply_perspective_transform
        :return:
            dst: image with desired output size, height=32, width=flags.img_width
            crop_bbox: bounding box on input image
        """
        bbox = cv2.boundingRect(text_box_pnts_transformed)
        bbox_width = bbox[2]
        bbox_height = bbox[3]

        # Output shape is (self.out_width, self.out_height)
        # We randomly put bounding box of transformed text in the output shape
        # so the max value of dst_height is out_height

        # TODO: If rotate angle(z) of text is too big, text will become very small,
        # we should do something to prevent text too small

        # dst_height and dst_width is used to leave some padding around text bbox

        dst_height = random.randint(self.out_height // 8 * 7, self.out_height)

        #dst_height = random.randint(self.out_height // 4 * 3, self.out_height)

        if self.out_width == 0:
            scale = bbox_height / dst_height
        else:
            dst_width = self.out_width
            scale = max(bbox_height / dst_height, bbox_width / self.out_width)
       
        #print("dst_width : ", dst_width, " out_width : ", self.out_width, "bbox_width : ", bbox_width, "bbox_height : ", bbox_height, "scale : " , scale)
        s_bbox_width = math.ceil(bbox_width / scale)
        s_bbox_height = math.ceil(bbox_height / scale)
            
        if self.out_width == 0:
            padding = random.randint(s_bbox_width // 10, s_bbox_width // 8)
            dst_width = s_bbox_width + padding * 2

        s_bbox = (np.around(bbox[0] / scale),
                  np.around(bbox[1] / scale),
                  np.around(bbox[2] / scale),
                  np.around(bbox[3] / scale))

        

        #增加一下随机，变成裁剪效果，这样才能精确的控制裁剪到字体
        x_offset, y_offset = self.random_xy_offset(s_bbox_height, s_bbox_width, self.out_height, dst_width)
        #y_offset = 0
        #这里会出现极端情况可能出现负值
        if (s_bbox[0] - x_offset) < 0:
            print ("FUCKING ERROR")
            return None
        if (s_bbox[1] - y_offset < 0):
            print ("FUCKING Y_ERROR")
            return None

        dst_bbox = (
            self.int_around((s_bbox[0] - x_offset) * scale),
            self.int_around((s_bbox[1] - y_offset) * scale),
            self.int_around(dst_width * scale),
            self.int_around(self.out_height * scale)
        )
        #左右x坐标上的边界会不会超出
        maxBGLeftX = max(bg_pnts_transformed[0][0], bg_pnts_transformed[3][0])
        minBGRightX = min(bg_pnts_transformed[1][0], bg_pnts_transformed[2][0])
        if dst_bbox[0] + dst_bbox[2] > minBGRightX:
            print ("FUCKING minBGRightX_ERROR", dst_bbox[0], dst_bbox[2], minBGRightX)
            return None
        if dst_bbox[0] < maxBGLeftX:
            print ("FUCKING maxBGLeftX_ERROR", dst_bbox[0], maxBGLeftX)
            return None

        #上下Y坐标上的边界会不会超出, 这里的坐标系是以左上角为原点，所以Y轴需要倒转一下
        maxY = min(bg_pnts_transformed[0][1], bg_pnts_transformed[1][1])
        minY = max(bg_pnts_transformed[2][1], bg_pnts_transformed[3][1])
        

        if dst_bbox[1] + dst_bbox[3] > maxY:
            print ("FUCKING maxY_ERROR", dst_bbox[1], dst_bbox[3], maxY)
            return None
        if dst_bbox[1] < minY:
            print ("FUCKING minY_ERROR", dst_bbox[1], minY)
            return None
        '''
        if (dst_bbox[0] + dst_bbox[2] > bbox[0] + bbox[2]):
            #超出边界了呀
            print ("FUCKING ERROR XWidth ", dst_bbox[0] ,dst_bbox[2], bbox[0],  bbox[2])
        if (dst_bbox[1] + dst_bbox[3] > bbox[1] + bbox[3]):
            #超出边界了呀
            print ("FUCKING ERROR YHeight", dst_bbox[1], dst_bbox[3] , bbox[1] , bbox[3])
        '''
        if self.show:
            print ("out_height", self.out_height, "rand dst_height : ", dst_height, "bbox_height : ", bbox_height,  " bbox_width : ", bbox_width, " src aspo : ", bbox_width / float(bbox_height))
            print ("s_bbox_width : ", s_bbox_width, " s_bbox_height : ", s_bbox_height, " s_bbox : ", s_bbox)
            print ("h_scale : ", bbox_height / dst_height, " w_scale : ", bbox_width / self.out_width,  " Sacle", scale, "s_bbox : ", s_bbox)
            print ("x_offset : ", x_offset, " y_offset : ", y_offset)
            print ("bbox : ", bbox, "dst_bbox : ", dst_bbox)
            text_box_pnts = [
                            [dst_bbox[0], dst_bbox[1]] , 
                            [dst_bbox[0] + dst_bbox[2] , dst_bbox[1]], 
                            [dst_bbox[0] + dst_bbox[2] , dst_bbox[1] + dst_bbox[3]],
                            [dst_bbox[0] , dst_bbox[1] + dst_bbox[3]]
                            ]
            self.plt_show_list(img, [text_box_pnts_transformed,text_box_pnts], title= "before crop_resize")
            
        # It's important do crop first and than do resize for speed consider
        #img 是 宽 * 高
        dst = img[dst_bbox[1]:dst_bbox[1] + dst_bbox[3], dst_bbox[0]:dst_bbox[0] + dst_bbox[2]]
        #dst = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        if self.show:
            print ("Before Resize : ", dst.shape)
            self.plt_show(dst)
         #这里务必选择LINEAR，否则会出现白边
        dst = cv2.resize(dst, (dst_width, self.out_height), interpolation=cv2.INTER_LINEAR)

        return dst, dst_bbox

    def int_around(self, val):
        return int(np.around(val))

    def get_word_color(self, bg, text_x, text_y, word_height, word_width):
        """
        Only use word roi area to get word color
        """
        #offset = 10
        offset = 0
        ymin = text_y - offset
        ymax = text_y + word_height + offset
        xmin = text_x - offset
        xmax = text_x + word_width + offset

        word_roi_bg = bg[ymin: ymax, xmin: xmax]

        #print (word_roi_bg[word_roi_bg < 32].shape)
        #黑色的太多了，那么PASS掉
        if word_roi_bg[word_roi_bg < 32].shape[0] > 200 :
            raise Exception()
            return None
            #plt.imshow(bg)
            #plt.show()

        #出现一些更不清晰的样本
        bg_mean = int(np.mean(word_roi_bg) * (2 / 3))
        #if bg_mean < 

        word_color = random.randint(0, bg_mean)
        #print ("bg_mean : ", bg_mean, " np.mean(word_roi_bg) : ", np.mean(word_roi_bg), "word_color : ", word_color)
        return word_color

    def draw_text_on_bg(self, word, font, bg,language,lock = None):
        """
        Draw word in the center of background
        :param word: word to draw
        :param font: font to draw word
        :param bg: background numpy image
        :return:
            np_img: word image
            text_box_pnts: left-top, right-top, right-bottom, left-bottom
        """


        bg_height = bg.shape[0]
        bg_width = bg.shape[1]

        word_size = self.get_word_size(font, word)
        word_height = word_size[1]
        word_width = word_size[0]

        offset = font.getoffset(word)
        word_list = []
        text_x = int((bg_width - word_width) / 2)
        text_y = int((bg_height - word_height) / 2)
        word_color = self.get_word_color(bg, text_x, text_y, word_height, word_width)




        pil_img = Image.fromarray(np.uint8(bg))
        draw = ImageDraw.Draw(pil_img)
        if self.show:
            self.plt_show(bg, title = 'bg')
        #Draw text in the center of bg


        if self.show:
            print ("BG_H_W : ( ", bg_height, bg_width,  ")", " Offset : (" , offset , ")", " WordSize : (", word_size, ")", "Text_x", text_x, "Text_y", text_y)
        if word_color is None:
            raise Exception
        if apply(self.cfg.random_space):
            text_x, text_y, word_width, word_height = self.draw_text_with_random_space(draw, font, word, word_color,
                                                                                       bg_width, bg_height)
            np_img = np.array(pil_img).astype(np.float32)
        else:
            # if apply(self.cfg.seamless_clone):
            #     np_img = self.draw_text_seamless(font, bg, word, word_color, word_height, word_width, offset)
            # else:
                word_width = self.draw_text_wrapper(draw, word, text_x - offset[0], text_y - offset[1], font, word_color)
                # draw.text((text_x - offset[0], text_y - offset[1]), word, fill=word_color, font=font)

                # np_img = np.array(pil_img).astype(np.int16)
                if (apply(self.cfg.split_thin_font)):
                    word_size = self.get_word_size(font, word)

                    mask_bg = np.ones((bg_height, bg_width)) * 255
                    mask_bg = mask_bg.astype(np.uint8)
                    # word_list = [word,'split']
                    # 将mask 背景生成
                    mask_img = Image.fromarray(np.uint8(mask_bg))
                    mask_draw = ImageDraw.Draw(mask_img)

                    word_width = self.draw_text_wrapper(mask_draw, word, text_x - offset[0], text_y - offset[1], font, 0)
                    mask_img = np.array(mask_img)
                    pil_img = self.split_thin(pil_img,mask_img,bg,lock = lock)




                np_img = np.array(pil_img).astype(np.float32)








        if language == 'chn':
            str_list_left = '《〈【〔「'

            str_list_right = '》！？〉】〕」：】。、'
            if word[-1] in str_list_right :
            #if word[-1] == '。' or word[-1] == '、':
                tmp_right_offset = np.random.randint(font.size//3,font.size//3*2)
                word_width = word_width-tmp_right_offset

            if word[0] in str_list_left:
                tmp_left_offset = np.random.randint(font.size // 3, font.size // 3*2)
                text_x = text_x+tmp_left_offset
        elif language == 'jap':
            str_list_left = '「【《〈〔'
            str_list_right = '】」。、〕》〉'
            if word[-1] in str_list_right:
                # if word[-1] == '。' or word[-1] == '、':
                tmp_right_offset = np.random.randint(font.size // 3, font.size // 3*2)
                word_width = word_width - tmp_right_offset

            if word[0] in str_list_left:
                tmp_left_offset = np.random.randint(font.size // 3, font.size // 2)
                text_x = text_x + tmp_left_offset





            #print(tmp_offset)
        text_box_pnts = [
            [text_x, text_y],
            [text_x + word_width, text_y],
            [text_x + word_width, text_y + word_height],
            [text_x, text_y + word_height]
        ]
        if word_list != []:


            return np_img, text_box_pnts, word_color,word_list
        else:
            return np_img, text_box_pnts, word_color,word


    def draw_add_script_text_on_bg(self, word, font, bg,font_little):
        """
        Draw word in the center of background
        :param word: word to draw
        :param font: font to draw word
        :param bg: background numpy image
        :return:
            np_img: word image
            text_box_pnts: left-top, right-top, right-bottom, left-bottom
        """
        bg_height = bg.shape[0]
        bg_width = bg.shape[1]

        word_size = self.get_word_size(font, word)#
        word_height = word_size[1]
        word_width = word_size[0]

        offset = font.getoffset(word)

        pil_img = Image.fromarray(np.uint8(bg))
        draw = ImageDraw.Draw(pil_img)
        if self.show:
            self.plt_show(bg, title='bg')
        # print ("BG_H_W : ( ", bg_height, bg_width,  ")", " Offset : (" , offset , ")", " WordSize : (", word_size, ")", "Text_x", text_x, "Text_y", text_y)
        # Draw text in the center of bg
        text_x = int((bg_width - word_width) / 2)
        text_y = int((bg_height - word_height) / 2)
        if self.show:
            print("BG_H_W : ( ", bg_height, bg_width, ")", " Offset : (", offset, ")", " WordSize : (", word_size, ")",
                  "Text_x", text_x, "Text_y", text_y)
        word_color = self.get_word_color(bg, text_x, text_y, word_height, word_width)

        if word_color is None:
            raise Exception

        if apply(self.cfg.random_space):
            text_x, text_y, word_width, word_height = self.draw_text_with_random_space(draw, font, word, word_color,
                                                                                       bg_width, bg_height)


            np_img = np.array(pil_img).astype(np.float32)
        else:
                word_width,word_height,text_y,text_x = self.draw_text_add_script(draw, word, text_x - offset[0], text_y - offset[1], font, word_color,font_little)
                # draw.text((text_x - offset[0], text_y - offset[1]), word, fill=word_color, font=font)


                np_img = np.array(pil_img).astype(np.float32)

        text_box_pnts = [
            [text_x, text_y],
            [text_x + word_width, text_y],
            [text_x + word_width, text_y + word_height],
            [text_x, text_y + word_height]
        ]
        # text_box_pnts = [
        #     [text_x - offset[0], text_y - offset[1]],
        #     [text_x - offset[0] + word_width, text_y - offset[1]],
        #     [text_x - offset[0] + word_width, text_y - offset[1] + word_height],
        #     [text_x - offset[0], text_y - offset[1] + word_height]
        # ]
        #print('text_box_pnts_add_script',text_box_pnts)
        return np_img, text_box_pnts, word_color

    def draw_text_seamless(self, font, bg, word, word_color, word_height, word_width, offset):
        # For better seamlessClone
        seamless_offset = 6

        # Draw text on a white image, than draw it on background
        white_bg = np.ones((word_height + seamless_offset, word_width + seamless_offset)) * 255
        text_img = Image.fromarray(np.uint8(white_bg))
        draw = ImageDraw.Draw(text_img)

        # draw.text((0 + seamless_offset // 2, 0 - offset[1] + seamless_offset // 2), word,
        #           fill=word_color, font=font)

        self.draw_text_wrapper(draw, word,
                               0 + seamless_offset // 2,
                               0 - offset[1] + seamless_offset // 2,
                               font, word_color)

        # assume whole text_img as mask
        text_img = np.array(text_img).astype(np.uint8)
        text_mask = 255 * np.ones(text_img.shape, text_img.dtype)

        # This is where the CENTER of the airplane will be placed
        center = (bg.shape[1] // 2, bg.shape[0] // 2)

        # opencv seamlessClone require bgr image
        text_img_bgr = np.ones((text_img.shape[0], text_img.shape[1], 3), np.uint8)
        bg_bgr = np.ones((bg.shape[0], bg.shape[1], 3), np.uint8)
        cv2.cvtColor(text_img, cv2.COLOR_GRAY2BGR, text_img_bgr)
        cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR, bg_bgr)

        flag = np.random.choice([
            cv2.NORMAL_CLONE,
            cv2.MIXED_CLONE,
            cv2.MONOCHROME_TRANSFER
        ])
        #flag = cv2.MIXED_CLONE
        mixed_clone = cv2.seamlessClone(text_img_bgr, bg_bgr, text_mask, center, flag)

        np_img = cv2.cvtColor(mixed_clone, cv2.COLOR_BGR2GRAY)

        return np_img
    def draw_text_with_random_space(self, draw, font, word, word_color, bg_width, bg_height):
        """ If random_space applied, text_x, text_y, word_width, word_height may change"""
        width = 0
        height = 0
        chars_size = []
        y_offset = 10 ** 5
        for c in word:
            size = font.getsize(c)
            chars_size.append(size)

            width += size[0]
            # set max char height as word height
            if size[1] > height:
                height = size[1]

            # Min chars y offset as word y offset
            # Assume only y offset
            c_offset = font.getoffset(c)
            if c_offset[1] < y_offset:
                y_offset = c_offset[1]

        char_space_width = int(height * np.random.uniform(self.cfg.random_space.min, self.cfg.random_space.max))

        width += (char_space_width * (len(word) - 1))

        text_x = int((bg_width - width) / 2)
        text_y = int((bg_height - height) / 2)

        c_x = text_x
        c_y = text_y

        for i, c in enumerate(word):
            # self.draw_text_wrapper(draw, c, c_x, c_y - y_offset, font, word_color, force_text_border)
            draw.text((c_x, c_y - y_offset), c, fill=word_color, font=font)

            c_x += (chars_size[i][0] + char_space_width)

        return text_x, text_y, width, height

    def draw_text_wrapper(self, draw, text, x, y, font, text_color):
        """
        :param x/y: 应该是移除了 offset 的
        """
        if apply(self.cfg.text_border):
            self.dmsg ("draw border")
            self.draw_border_text(draw, text, x, y, font, text_color)
            width = self.get_word_size(font, text)[0]
        else:
            if self.prob(0.45) and 'Heiti'.upper() not in font.getname()[0].upper():
                width = x
                text_tmp = ''
                zhong_symbol_right = '》！？〉）】〕」：；】，。、'
                zhong_symbol_left = '《〈【（〔「'
                for t in text[:-1]:
                    if t not in zhong_symbol_right and t not in zhong_symbol_left:
                        text_tmp += t
                    else:
                        if t in zhong_symbol_right:
                            text_tmp += t
                            draw.text((width, y), text_tmp, fill=text_color, font=font)
                            width += font.getsize(text_tmp)[0] - random.randint(0,font.getsize(t)[0] // 2)
                        else:
                            draw.text((width, y), text_tmp, fill=text_color, font=font)
                            width += font.getsize(text_tmp)[0] - - random.randint(0,font.getsize(t)[0] // 2)
                            draw.text((width, y), t, fill=text_color, font=font)
                            width += font.getsize(t)[0]
                        text_tmp = ''
                if text_tmp != '':
                    draw.text((width, y), text_tmp, fill=text_color, font=font)
                    width += font.getsize(text_tmp)[0]
                draw.text((width, y), text[-1], fill=text_color, font=font)
                width += font.getsize(text[-1])[0] - x
            else:
                draw.text((x, y), text, fill=text_color, font=font)
                width = self.get_word_size(font,text)[0]

        return width

    def find_superscript_y(self,y,text_char,superscript_char,font,font_little):
        script_max_y = y  + ((font.getsize(text_char)[1] - font.getoffset(text_char)[1]) // 2) + \
                       font.getoffset(text_char)[1] - (font_little.getsize(superscript_char)[1])  # 上标位置随机
        script_min_y = y  +font_little.getoffset(superscript_char)[1] - (font_little.getsize(superscript_char)[1])
        # script_min_y = y   - (font_little.getsize(superscript_char)[1])

        if script_max_y <= script_min_y:
            y_superscript = script_max_y
        else:

            y_superscript = np.random.randint(script_min_y, script_max_y)  # TODO  2有问题
        return y_superscript,script_max_y

    def finde_subscript_y(self,y,text_char,subscript_char,font,font_little):
        script_min_y = y  -((font.getsize(text_char)[1] - font.getoffset(text_char)[1]) // 5*3) + \
                       font.getsize(text_char)[1]-font_little.getoffset(subscript_char)[1] # 上标位置随机
        # script_max_y = y   - font_little.getoffset(subscript_char)[1]
        script_max_y = y  +font.getsize(text_char)[1] - font_little.getoffset(subscript_char)[1]

        if script_min_y >= script_max_y:
            y_subscript = script_max_y
        else:
            y_subscript = np.random.randint(script_min_y, script_max_y)  # TODO  2有问题
        return y_subscript

    def find_test_char(self,tmp_text):
        for j, c in enumerate(tmp_text):
            if j != 0:
                if c != '▵' and c != '▿' and tmp_text[j - 1] != '▵' and tmp_text[j - 1] != '▿' and c != ' ':
                    test_char = c
                    break
            else:
                if c != ' ':
                    test_char = c
                    break
        return test_char

    def update_ymax_and_ymin(self,y_superscript,new_y_min,new_y_max,char,font_little):
        new_y_min = min(y_superscript + font_little.getoffset(char)[1], new_y_min)
        new_y_max = max(y_superscript + font_little.getsize(char)[1], new_y_max)
        return new_y_min,new_y_max

    def draw_normal_text(self,draw,x,y,text,text_color,font):
        draw.text((x, y), text, fill=text_color,
                  font=font)
        x += font.getsize(text)[0]
        return x

    def count_space_num_left(self,text):
        num = 0
        for i in range(1,len(text)+1):
            if text[-i]== ' ':
                num+=1
            else:
                return num,text[-i]
        return num,' '

    def count_space_num_right(self,text):
        num = 0
        for i in range(len(text)):
            if text[i] == ' ':
                num+=1
            else:
                return num,text[i]
        return num,' '


    def draw_text_add_script(self, draw, text, x, y, font, text_color, font_little):
        word_start = x
        random_offset = np.random.randint(-1, 2)
        y = y+random_offset
        start_index = 0
        new_y_min = 10000
        new_y_max = y
        tmp_script_index_list = [i.start() for i in re.compile('▿|▵').finditer(text)]
        test_char = None
        for index, i in enumerate(tmp_script_index_list):
            tmp_script_index = i
            x = self.draw_normal_text(draw,x, y, text[start_index:tmp_script_index], text_color, font)
            new_y_min, new_y_max = self.update_ymax_and_ymin(y, new_y_min, new_y_max, text[start_index:],
                                                        font)
            text_tmp_1 = text[start_index:tmp_script_index]
            if text_tmp_1 != '':
                if text_tmp_1[-1] != ' ':
                    test_char = text_tmp_1.strip()[-1]
                else:
                    space_num_left, test_char_left = self.count_space_num_left(text_tmp_1)
                    space_num_right, test_char_right = self.count_space_num_right(text[tmp_script_index+2:])
                    if space_num_left > space_num_right:
                        test_char = test_char_right
                    elif space_num_left < space_num_right:
                        test_char = test_char_left
                    else:
                        test_char = np.random.choice([test_char_left, test_char_right])
            # if text[i] == '▵' and len(re.compile(r'([a-zA-Z0-9]+|[\(\)\-\=\+\®\©\]+)').findall(text[i + 1])) != 0:
            if text[i] == '▵' and text[i + 1]!= '':
                if index != 0 and text[tmp_script_index_list[index-1]] == '▵':
                    y_superscript = y_superscript
                else:
                    if test_char == None:
                        test_char = self.find_test_char(text[i+2:])   #  TODO:寻找作为标杆的大字母有些许问题
                    y_superscript,script_max_y = self.find_superscript_y(y,test_char,text[i+1],font,font_little)
                draw.text((x, y_superscript), text[i + 1], fill=text_color, font=font_little)
                x += font_little.getsize(text[i + 1])[0]    #更新x的位置
                new_y_min,new_y_max = self.update_ymax_and_ymin(y_superscript, new_y_min, new_y_max, text[i + 1], font_little)
                start_index = i + 2
                if index == len(tmp_script_index_list) - 1 and text[start_index:] != '':
                    x = self.draw_normal_text(draw,x, y, text[start_index:], text_color, font)
                    new_y_min, new_y_max = self.update_ymax_and_ymin(y, new_y_min, new_y_max, text[start_index:],
                                                                font)
            # elif text[i] == '▿' and len(re.compile(r'([a-zA-Z0-9]+|[\(\)\-\=\+\®\©]+)').findall(text[i + 1])) != 0:
                #下标位置随机
            elif text[i] == '▿' and text[i + 1] != '':

                if index != 0 and text[tmp_script_index_list[index-1]] == '▿':
                    y_subscript = y_subscript
                else:
                    y_subscript = self.finde_subscript_y(y,test_char[-1],text[i+1],font,font_little)
                new_y_min,new_y_max = self.update_ymax_and_ymin(y_subscript, new_y_min, new_y_max, text[i + 1], font_little)   #更新ymax和ymin
                draw.text((x, y_subscript), text[i + 1], fill=text_color,
                          font=font_little)
                x += font_little.getsize(text[i + 1])[0]

                start_index = i + 2
                if index == len(tmp_script_index_list) - 1 and text[start_index:] != '':
                    x = self.draw_normal_text(draw,x, y, text[start_index:], text_color, font)
                    new_y_min, new_y_max = self.update_ymax_and_ymin(y, new_y_min, new_y_max, text[start_index:], font)
        word_height = new_y_max - new_y_min

        return x - word_start, word_height,new_y_min,word_start


    def draw_text_add_script_ori(self,draw, text, x, y, font, text_color,font_little):
        word_start = x
        word_height = 0
        random_offset =0 # np.random.randint(-1,1)


        for index, t in enumerate(text):

            if t == '▵' and len(re.compile(r'([a-zA-Z0-9]+|[\(\)\-\=\+]+)').findall(text[index + 1])) != 0: #判定上角标
                draw.text((x, y+random_offset), text[index + 1], fill=text_color, font=font_little)
                x += font_little.getsize(text[index + 1])[0]
                word_height = max(word_height,font_little.getsize(text[index + 1])[1]-font_little.getoffset(text[index+1])[1])
            elif t =='▿' and len(re.compile(r'([a-zA-Z0-9]+|[\(\)\-\=\+]+)').findall(text[index + 1])) != 0: #判定下角标
                draw.text((x, y + int(font_little.size)+random_offset+1), text[index + 1], fill=text_color, font=font_little)
                x += font_little.getsize(text[index + 1])[0]
                word_height = max(word_height,int(font_little.size)+1+font_little.getsize(text[index+1])[1])

            else:
                if (text[index - 1] == '▵' or text[index - 1] == '▿') and len(re.compile(r'([a-zA-Z0-9]+|[\(\)\-\=\+]+)').findall(t)) != 0:
                    continue
                draw.text((x, y+random_offset), t, fill=text_color, font=font)
                #mask, offset = font.getmask2(t, draw.mode)
                x += font.getsize(t)[0]
                #x += offset[0]
                if t == '_':
                    tmp_word_height = font.getsize(t)[1]-font.getoffset(t)[1]
                    if tmp_word_height < word_height:

                        word_height = max(word_height, font.getsize(t)[1])
                    else:
                        word_height = max(word_height, font.getsize(t)[1] - font.getoffset(t)[1])

                else:
                    word_height = max(word_height,font.getsize(t)[1])
        return x-word_start,word_height


    def draw_border_text(self, draw, text, x, y, font, text_color):
        """
        :param x/y: 应该是移除了 offset 的
        """
        # thickness larger than 1 may give bad border result
        thickness = 1

        choices = []
        p = []
        if self.cfg.text_border.light.enable:
            choices.append(0)
            p.append(self.cfg.text_border.light.fraction)
        if self.cfg.text_border.dark.enable:
            choices.append(1)
            p.append(self.cfg.text_border.dark.fraction)

        light_or_dark = np.random.choice(choices, p=p)

        if light_or_dark == 0:
            border_color = text_color + np.random.randint(0, 255 - text_color - 1)
        elif light_or_dark == 1:
            border_color = text_color - np.random.randint(0, text_color + 1)

        # thin border
        draw.text((x - thickness, y), text, font=font, fill=border_color)
        draw.text((x + thickness, y), text, font=font, fill=border_color)
        draw.text((x, y - thickness), text, font=font, fill=border_color)
        draw.text((x, y + thickness), text, font=font, fill=border_color)

        # thicker border
        draw.text((x - thickness, y - thickness), text, font=font, fill=border_color)
        draw.text((x + thickness, y - thickness), text, font=font, fill=border_color)
        draw.text((x - thickness, y + thickness), text, font=font, fill=border_color)
        draw.text((x + thickness, y + thickness), text, font=font, fill=border_color)

        # now draw the text over it
        draw.text((x, y), text, font=font, fill=text_color)

   
    def gen_bg(self, width, height, lock):
        if apply(self.cfg.img_bg):
            bg = self.gen_bg_from_image(int(width), int(height), lock)
            #print('img bg',bg.shape)
        else:
            bg = self.gen_rand_bg(int(width), int(height), lock)
            #print('random',bg.shape)
        return bg

    def gen_rand_bg(self, width, height, lock):
        """
        Generate random background
        """
        r = random.randint(2, 6)
        self.dmsg ("randbg : ", r)
        #r = 4
        if r == 1:
            bg = np.array(BackgroundGenerator().quasicrystal(height, width))
            bg = self.apply_gauss_blur(bg, lock=lock)
            #print('1',bg.shape)
            return bg
        if r == 2:   
            bg_high = random.uniform(220, 255)
            bg_low = bg_high - random.uniform(0, 128)
            bg = np.random.randint(bg_low, bg_high, (height, width)).astype(np.uint8)
            #if random.randint(1,4) < 4:
            bg = self.apply_gauss_blur(bg, lock=lock)
            #print('2', bg.shape)
            return bg
        if r == 3:  
            bg = np.array(BackgroundGenerator().gaussian_noise(height, width))
            bg = self.apply_gauss_blur(bg, lock = lock)
            #print('3', bg.shape)
            return bg

        if r >3 and r< 6:
            #noise_index = np.random.randint(225,254)
            bg = np.random.randint(220, 255, (height, width)).astype(np.uint8)
            # plt.figure('1')
            # plt.imshow(bg)
            # plt.show()
            #if random.randint(1,4) < 4:
            bg = self.apply_gauss_blur(bg, lock = lock)
            #print('4', bg.shape)
            return bg
           

    def gen_bg_from_image(self, width, height, lock):
        """
        Resize background, let bg_width>=width, bg_height >=height, and random crop from resized background
        """
        assert width > height

        bg = random.choice(self.bgs)

        scale = max(width / bg.shape[1], height / bg.shape[0])

        out = bg
        #out = cv2.resize(bg, None, fx=scale, fy=scale)

        x_offset, y_offset = self.random_xy_offset(height, width, out.shape[0], out.shape[1])

        out = out[y_offset:y_offset + height, x_offset:x_offset + width]
        #有33%的几率不做模糊处理
        if random.randint(0, 1) > 0:
            out = self.apply_gauss_blur(out, ks=[3, 5, 7, 9, 11, 13, 15, 17], lock = lock)
        #out = self.apply_gauss_blur(out, ks = 1)
        #bg_mean = int(np.mean(out))

        # TODO: find a better way to deal with background
        # alpha = 255 / bg_mean  # 对比度
        # beta = np.random.randint(bg_mean // 4, bg_mean // 2)  # 亮度
        # out = np.uint8(np.clip((alpha * out + beta), 0, 255))

        return out

    def choose_font(self,language,word ,font_dct):
        if language == 'eng':
            if '=' in word :
                font_path = random.choice(font_dct['eng_strict'])
            else:
                font_path = random.choice(font_dct['eng'])
            if '□' in word or '■' in word:
                font_path = random.choice(font_dct['eng_checkbox'])
                if '=' in word:
                    word = word.replace('=','')
        else:
            if language == 'jap':
                font_path = random.choice(font_dct['jap'])
                if '□' in word or '■' in word:
                    font_path = random.choice(font_dct['jap_checkbox'])
            else:
                if ',' in word or ';' in word:

                    font_path = random.choice(font_dct['chn_strict'])
                else:
                    font_path = random.choice(font_dct['chn'])
        return font_path,word

    def get_font_little_size(self,font_size):
        font_little_size = int(self.p1(font_size))
        font_little_size = np.random.randint(font_little_size - 1, font_little_size+2)
        return font_little_size

    @retry
    def pick_font(self, img_index):
        """
        :param img_index when use list corpus, this param is used
        :return:
            font: truetype
            size: word size, removed offset (width, height)
        """
        try:
            if apply(self.cfg.add_script):
                word, language = self.corpus.get_sample_add_script(img_index)
            else:
                word, language = self.corpus.get_sample(img_index)
            if self.clip_max_chars and len(word) > self.max_chars:
                word = word[:self.max_chars]
            font_dct = self.fonts
            #word = ', lal，l。a; lala: la；la.aaa'
            font_path,word = self.choose_font(language, word, font_dct)

            if self.strict:
                unsupport_chars = self.font_unsupport_chars[font_path]
                for c in word:
                    if c == ' ' or c == '▵' or c =='▿':
                        continue
                    if c in unsupport_chars:
                        print('Retry pick_font(), \'%s\' contains chars \'%s\' not supported by font %s' % (
                            word, c, font_path))
                        raise Exception
            # Font size in point
            font_size = random.randint(self.cfg.font_size.min, self.cfg.font_size.max)
            #font_path = '/fengjing/data_script/OCR_textrender/data/fonts/chn/STHeiti Medium.ttc'
            font_name = os.path.basename(font_path)
            if 'Capture_it.ttf' in font_path:
                word = word.upper()
            #word = '上海。北京、《附。件？上海、北）京、'

            font = ImageFont.truetype(font_path, font_size)
            font_little_size= self.get_font_little_size(font_size)
            font_little = ImageFont.truetype(font_path, font_little_size)
            return word, font, self.get_word_size(font, word),font_little,language,font_name

        except Exception as e:
            print("Retry pick_font: %s" % str(e))
            traceback.print_exc()
            #继续重试
            raise Exception

            

    def get_word_size(self, font, word):
        """
        Get word size removed offset
        :param font: truetype
        :param word:
        :return:
            size: word size, removed offset (width, height)
        """
        offset = font.getoffset(word)
        size = font.getsize(word)
        size = (size[0] - offset[0], size[1] - offset[1])
        return size

    def gray2rgb(self, imggray):
        '''

        :param imggray:
        :return:
        '''
        rgb_img = np.zeros((imggray.shape[0],imggray.shape[1],3))
        rgb_img[:, :, 2] = imggray
        rgb_img[:, :, 0] = imggray
        rgb_img[:, :, 1] = imggray

        return rgb_img

    def apply_seamless_cloe_add_foreground(self,img1):
        '''

        :param img1:
        :return:
        '''
        tmp_name = '/data1/fengjing/output/tmp/' + str(time.time()+random.random()) +'.jpg'
        # tmp_name = '/fengjing/data_script/OCR_textrender/output/tmp/' +'1.jpg'
        cv2.imwrite(tmp_name,img1)
        img1 = cv2.imread(tmp_name)
        img2  = random.choice(self.fgs)
        height, width = img1.shape[0:2]
        height_2, width_2 = img2.shape[0:2]
        # 最大crop img 的宽高
        crop_max_width = min(width, width_2)
        crop_max_height = min(height_2, height)
        # 实际crop img 宽高
        crop_height = random.randint(5, crop_max_height)
        crop_width = random.randint(5, crop_max_width)
        # crop img 随机裁剪的位置
        crop_x = random.randint(0, crop_max_width - crop_width)
        crop_y = random.randint(0, crop_max_height - crop_height)
        crop_img = img2[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        crop_h, crop_w = crop_img.shape[0:2]
        # 随机x，y 框放的位置
        range_x = random.randint(0, width - crop_w)
        range_y = random.randint(0, height - crop_h)

        # 由随机xy 计算center
        center = (range_x + crop_w // 2, range_y + crop_h // 2)
        mask = 255 * np.ones(crop_img.shape, crop_img.dtype)
        mixed_clone = cv2.seamlessClone(crop_img, img1, mask, (center[0], center[1]), cv2.MIXED_CLONE)
        ret, binary = cv2.threshold(~crop_img[:, :, 2], 30, 255, cv2.THRESH_BINARY)
        mask_coor = np.argwhere(binary > 200)

        for i in mask_coor:
            try:
                img1[range_y + i[0], range_x + i[1]] = mixed_clone[range_y + i[0], range_x + i[1]]
            except Exception as e:
                print(e)
                continue


        np_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        return np_img
    def apply_perspective_transform(self, img, text_box_pnts, max_x, max_y, max_z, gpu=False):
        """
        Apply perspective transform on image
        :param img: origin numpy image
        :param text_box_pnts: four corner points of text
        :param x: max rotate angle around X-axis
        :param y: max rotate angle around Y-axis
        :param z: max rotate angle around Z-axis
        :return:
            dst_img:
            dst_img_pnts: points of whole word image after apply perspective transform
            dst_text_pnts: points of text after apply perspective transform
        """

        x = math_utils.cliped_rand_norm(0, max_x)
        y = math_utils.cliped_rand_norm(0, max_y)
        #有一部分数据是正常的，没有经过旋转
        if (prob(0.2)):
            z = 0
        else:
            z = math_utils.cliped_rand_norm(0, max_z)

        self.dmsg("Transform , x: %f, y: %f, z: %f" % (x, y, z))

        transformer = math_utils.PerspectiveTransform(x, y, z, scale=1.0, fovy=50)

        dst_img, M33, dst_img_pnts = transformer.transform_image(img, gpu)
        dst_text_pnts = transformer.transform_pnts(text_box_pnts, M33)

        return dst_img, dst_img_pnts, dst_text_pnts

    def apply_blur_on_output(self, img,word, lock = None):
        #if prob(0.5):
        if prob(1):
            self.dmsg ("-*- APPLY blured 3,5 ")
            return self.apply_gauss_blur(img, [1, 2], lock = lock)
        else:
            self.dmsg ("-*- APPLY blured")
            return self.apply_norm_blur(img)

    def apply_gauss_blur(self, img, ks=None, lock=None):
        if ks is None :

            ks = [7, 9, 11, 13]

        ksize = random.choice(ks)

        sigmas = [0, 1, 2, 3, 4, 5, 6, 7]
        sigma = 0
        if ksize <= 3:
            sigma = random.choice(sigmas)
        if lock:
            with lock:
                if img.shape[0] > ksize and img.shape[1] > ksize:
                    img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        else:
            if img.shape[0] > ksize and img.shape[1] > ksize:
                img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        return img

    def apply_norm_blur(self, img, ks=None):
        # kernel == 1, the output image will be the same
        if ks is None:
            ks = [2, 3]
        kernel = random.choice(ks)
        img = cv2.blur(img, (kernel, kernel))
        return img

    def apply_prydown(self, img):
        """
        模糊图像，模拟小图片放大的效果
        """
        scale = random.uniform(1, self.cfg.prydown.max_scale)
        #scale = 2.2
        height = img.shape[0]
        width = img.shape[1]
        self.dmsg ("-*- APPLY prydown ", scale)
        out = cv2.resize(img, (int(width / scale), int(height / scale)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(out, (width, height), interpolation=cv2.INTER_LINEAR), scale

    def reverse_img(self, word_img):
        offset = np.random.randint(-10, 10)
        return 255 + offset - word_img

    def create_kernals(self):
        self.emboss_kernal = np.array([
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]
        ])
        '''
        self.sharp_kernel = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ])
        '''
        self.sharp_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
    def apply_emboss(self, word_img):
        return cv2.filter2D(word_img, -1, self.emboss_kernal)

    def apply_sharp(self, word_img):
        return cv2.filter2D(word_img, -1, self.sharp_kernel)

    def apply_crop(self, text_box_pnts, crop_cfg):
        """
        Random crop text box height top or bottom, we don't need image information in this step, only change box pnts
        :param text_box_pnts: bbox of text [left-top, right-top, right-bottom, left-bottom]
        :param crop_cfg:
        :return:
            croped_text_box_pnts
        """
        height = abs(text_box_pnts[0][1] - text_box_pnts[3][1])
        scale = float(height) / float(self.out_height)

        croped_text_box_pnts = text_box_pnts

        if prob(0.5):
            top_crop = int(random.randint(crop_cfg.top.min, crop_cfg.top.max) * scale)
            self.dmsg("top crop %d" % top_crop)
            croped_text_box_pnts[0][1] += top_crop
            croped_text_box_pnts[1][1] += top_crop
        else:
            bottom_crop = int(random.randint(crop_cfg.bottom.min, crop_cfg.bottom.max) * scale)
            self.dmsg("bottom crop %d " % bottom_crop)
            croped_text_box_pnts[2][1] -= bottom_crop
            croped_text_box_pnts[3][1] -= bottom_crop

        return croped_text_box_pnts


    def add_erode(self, img,font,word):
        if  ('▵' in word) or ('▿' in word):
            return img
        if 'thin' in font.getname()[0].lower() :

            radius = 1
        else:
            radius = random.randint(1,2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(radius, radius))
        #return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        img = cv2.dilate(img,kernel)
        self.dmsg ("-*- APPLY dilate ", radius)
        radius = random.randint(1,2)
        self.dmsg ("-*- APPLY erode ", radius)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(radius, radius))
        img = cv2.erode(img, kernel)
        return img


    def add_dilate(self,img):
        radius = random.randint(3,3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(radius, radius))
        img = cv2.dilate(img,kernel)
        return img
