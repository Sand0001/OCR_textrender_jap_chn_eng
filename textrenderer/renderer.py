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

# noinspection PyMethodMayBeStatic
from textrenderer.remaper import Remaper
import traceback
import matplotlib.pyplot as plt

#import pysnooper

class Renderer(object):
    def __init__(self, corpus, fonts, bgs, cfg, width=256, height=32,
                 clip_max_chars=False, debug=False, gpu=False, strict=False):
        self.corpus = corpus
        self.fonts = fonts
        self.bgs = bgs
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

        if self.strict:
            #print (self.fonts)
            self.font_unsupport_chars = font_utils.get_unsupported_chars(self.fonts, corpus.chars_file)


        self.show = False
        self.showEffect = True

    def start(self):
        return time.time()
    
    def end(self, t , msg = ""):
        return
        #print(msg + " took {:.3f}s".format(time.time() - t))


    def plt_show_list (self,word_img, text_box_pnts_list = None, title = None):
        test_img = np.clip(word_img, 0., 255.)
        i = 0
        if text_box_pnts_list is not None:
            for text_box_pnts in text_box_pnts_list:
                i += 1
                test_img = draw_box(test_img, text_box_pnts, (0, 255, i * 25 % 255))
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
  


    def gen_img(self, img_index, lock):
        t = self.start()
        word, font, word_size = self.pick_font(img_index)
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
        word_img, text_box_pnts, word_color = self.draw_text_on_bg(word, font, bg)
        if self.show:
            print ("BG SHAPE : ", bg.shape)
            print ("Word Image : ", word_img.shape)
            print ("text_box_pnts : ", text_box_pnts)
        self.end(t, "gen_bg & draw_text_on_bg : " + str(word_size))
        #print ("Before Apply", word_size, word_img.shape)
        self.dmsg("After draw_text_on_bg")
        t = self.start()
        if apply(self.cfg.crop):
            text_box_pnts = self.apply_crop(text_box_pnts, self.cfg.crop)
        self.end(t, "apply crop ")
        if apply(self.cfg.line):
            word_img, text_box_pnts = self.liner.apply(word_img, text_box_pnts, word_color)
            self.dmsg("After draw line")
        #print ("After Apply Line", text_box_pnts, word_img.shape, type(word_img))
        #test_image = draw_box(word_img, text_box_pnts, (0, 255, 155))
        #plt.imshow(test_image)
        #plt.show()
        if self.debug:
            word_img = draw_box(word_img, text_box_pnts, (0, 255, 155))
        if self.show:
            self.plt_show(word_img, text_box_pnts, title = "before Transform")

        if apply(self.cfg.curve):
            word_img, text_box_pnts = self.remaper.apply(word_img, text_box_pnts, word_color)
            self.dmsg("After remapping")

        if self.debug:
            word_img = draw_box(word_img, text_box_pnts, (155, 255, 0))

        #plt.imshow(word_img)
        #plt.show()
        t = self.start()
        #print ("Before transform ", word_img.shape)
        
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
            word_img = self.add_erode(word_img)
            if self.show:
                self.plt_show(word_img, title = 'After erode')


        if apply(self.cfg.dilate):
            self.dmsg ("-*- APPLY dilate ")
            word_img = self.add_dilate(word_img)

        #word_img = cv2.resize(word_img, (self.out_width, self.out_height), interpolation=cv2.INTER_CUBIC)
        if self.show:
            self.plt_show(word_img, title = 'After Resize')
        # #延后做
        # if not blured:
        #     if apply(self.cfg.prydown):
        #         word_img = self.apply_prydown(word_img)
        #         self.dmsg("After prydown")

        self.end(t, "apply2 ")
        self.dmsg ("***********************END*****************")
        return word_img, word

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
        if self.show:
            print ("Before Resize : ", dst.shape)
            self.plt_show(dst)
        dst = cv2.resize(dst, (dst_width, self.out_height), interpolation=cv2.INTER_CUBIC)

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
            
        


        bg_mean = int(np.mean(word_roi_bg) * (2 / 3))
        #if bg_mean < 

        word_color = random.randint(0, bg_mean)
        #print ("bg_mean : ", bg_mean, " np.mean(word_roi_bg) : ", np.mean(word_roi_bg), "word_color : ", word_color)
        return word_color

    def draw_text_on_bg(self, word, font, bg):
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
        
        pil_img = Image.fromarray(np.uint8(bg))
        draw = ImageDraw.Draw(pil_img)
        if self.show:
            self.plt_show(bg, title = 'bg')
        #text_x = random.randint(50, bg_width - word_width - 50)
        #text_y = random.randint(50, bg_height - word_height - 50)

        #print ("BG_H_W : ( ", bg_height, bg_width,  ")", " Offset : (" , offset , ")", " WordSize : (", word_size, ")", "Text_x", text_x, "Text_y", text_y)
        #Draw text in the center of bg
        text_x = int((bg_width - word_width) / 2)
        text_y = int((bg_height - word_height) / 2)
        if self.show:
            print ("BG_H_W : ( ", bg_height, bg_width,  ")", " Offset : (" , offset , ")", " WordSize : (", word_size, ")", "Text_x", text_x, "Text_y", text_y)
        word_color = self.get_word_color(bg, text_x, text_y, word_height, word_width)

        if word_color is None:
            raise Exception

        if apply(self.cfg.random_space):
            text_x, text_y, word_width, word_height = self.draw_text_with_random_space(draw, font, word, word_color,
                                                                                       bg_width, bg_height)
            np_img = np.array(pil_img).astype(np.float32)
        else:
            if apply(self.cfg.seamless_clone):
                np_img = self.draw_text_seamless(font, bg, word, word_color, word_height, word_width, offset)
            else:
                self.draw_text_wrapper(draw, word, text_x - offset[0], text_y - offset[1], font, word_color)
                # draw.text((text_x - offset[0], text_y - offset[1]), word, fill=word_color, font=font)

                np_img = np.array(pil_img).astype(np.float32)

        text_box_pnts = [
            [text_x, text_y],
            [text_x + word_width, text_y],
            [text_x + word_width, text_y + word_height],
            [text_x, text_y + word_height]
        ]

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
        else:
            draw.text((x, y), text, fill=text_color, font=font)

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
        else:
            bg = self.gen_rand_bg(int(width), int(height), lock)
        return bg

    def gen_rand_bg(self, width, height, lock):
        """
        Generate random background
        """
        r = random.randint(2, 4)
        self.dmsg ("randbg : ", r)
        #r = 1
        if r == 1:
            bg = np.array(BackgroundGenerator().quasicrystal(height, width))
            bg = self.apply_gauss_blur(bg, lock=lock)
            return bg
        if r == 2:   
            bg_high = random.uniform(220, 255)
            bg_low = bg_high - random.uniform(0, 128)
            bg = np.random.randint(bg_low, bg_high, (height, width)).astype(np.uint8)
            #if random.randint(1,4) < 4:
            bg = self.apply_gauss_blur(bg, lock=lock)
            return bg
        if r == 3:  
            bg = np.array(BackgroundGenerator().gaussian_noise(height, width))
            bg = self.apply_gauss_blur(bg, lock = lock)
            return bg 
        if r == 4: 
            bg = np.random.randint(220, 255, (height, width)).astype(np.uint8)
            #if random.randint(1,4) < 4:
            bg = self.apply_gauss_blur(bg, lock = lock)
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
        if random.randint(0, 2) > 0:
            out = self.apply_gauss_blur(out, ks=[3, 5, 7, 9, 11, 13, 15, 17], lock = lock)
        #out = self.apply_gauss_blur(out, ks = 1)
        bg_mean = int(np.mean(out))

        # TODO: find a better way to deal with background
        # alpha = 255 / bg_mean  # 对比度
        # beta = np.random.randint(bg_mean // 4, bg_mean // 2)  # 亮度
        # out = np.uint8(np.clip((alpha * out + beta), 0, 255))

        return out

    @retry
    def pick_font(self, img_index):
        """
        :param img_index when use list corpus, this param is used
        :return:
            font: truetype
            size: word size, removed offset (width, height)
        """
        try:
            word, language = self.corpus.get_sample(img_index)

            #if word is None:


            if self.clip_max_chars and len(word) > self.max_chars:
                word = word[:self.max_chars]
            font_dct = self.fonts
            #font_dct = random.choice(self.fonts)
            #different lang  should have different fonts
            #print(font_dct)
            if language == 'eng':
                font_path = random.choice(font_dct['eng'])
            else:
                if language == 'jap':
                    font_path = random.choice(font_dct['jap'])
                else:
                    font_path = random.choice(font_dct['chn'])
            #print (language, font_path)
            if self.strict:
                unsupport_chars = self.font_unsupport_chars[font_path]
                for c in word:
                    if c == ' ':
                        continue
                    if c in unsupport_chars:
                        print('Retry pick_font(), \'%s\' contains chars \'%s\' not supported by font %s' % (
                            word, c, font_path))
                        raise Exception

            # Font size in point
            font_size = random.randint(self.cfg.font_size.min, self.cfg.font_size.max)
            font = ImageFont.truetype(font_path, font_size)

            return word, font, self.get_word_size(font, word)
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

    def apply_blur_on_output(self, img, lock = None):
        if prob(0.5):
            self.dmsg ("-*- APPLY blured 3,5 ")
            return self.apply_gauss_blur(img, [3, 5], lock = lock)
        else:
            self.dmsg ("-*- APPLY blured")
            return self.apply_norm_blur(img)

    def apply_gauss_blur(self, img, ks=None, lock=None):
        if ks is None:
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
        out = cv2.resize(img, (int(width / scale), int(height / scale)), interpolation=cv2.INTER_AREA)
        return cv2.resize(out, (width, height), interpolation=cv2.INTER_AREA), scale

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


    def add_erode(self, img):
    
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
