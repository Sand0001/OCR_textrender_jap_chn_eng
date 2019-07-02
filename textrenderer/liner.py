import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from libs.utils import prob
import re
class LineState(object):
    tableline_x_offsets = range(8, 40)
    tableline_y_offsets = range(3, 10)
    tableline_thickness = [1, 2]

    # 0/1/2/3: 仅单边（左上右下）
    # 4/5/6/7: 两边都有线（左上，右上，右下，左下）
    tableline_options = range(0, 8)

    middleline_thickness = [1, 2, 3]
    middleline_thickness_p = [0.2, 0.7, 0.1]


class Liner(object):
    def __init__(self, cfg):
        self.linestate = LineState()
        self.cfg = cfg

    def apply(self, word_img, text_box_pnts, word_color,word,font):
        """
        :param word_img:  word image with big background
        :param text_box_pnts: left-top, right-top, right-bottom, left-bottom of text word
        :return:
        """
        line_p = []
        funcs = []

        if self.cfg.line.under_line.enable:
            line_p.append(self.cfg.line.under_line.fraction)
            funcs.append(self.apply_under_line)

        if self.cfg.line.table_line.enable:
            line_p.append(self.cfg.line.table_line.fraction)
            funcs.append(self.apply_table_line)

        if self.cfg.line.middle_line.enable:
            line_p.append(self.cfg.line.middle_line.fraction)
            funcs.append(self.apply_middle_line)

        if len(line_p) == 0:
            return word_img, text_box_pnts

        line_effect_func = np.random.choice(funcs, p=line_p)

        return line_effect_func(word_img, text_box_pnts, word_color,word,font)

    def apply_under_line(self, word_img, text_box_pnts, word_color,word,font,):
        y_offset = random.choice([0, 1])

        text_box_pnts[2][1] += y_offset
        text_box_pnts[3][1] += y_offset
        #print(word, font)




        line_color = word_color + random.randint(0, 10)
        # plt.figure('befor line')
        # plt.imshow(word_img)
        # plt.show()
        #if prob(0.5): #0.5的概率是虚线下划线
        leftBottomX, leftBottomY = text_box_pnts[3][0], text_box_pnts[3][1]
        rightBottomX, rightBottomY = text_box_pnts[2][0], text_box_pnts[2][1]
        #divded = len(word)
        chars_size = []
        leftBottomX_tmp = leftBottomX

        if prob(0): #0.3概率为部分虚线或者实线下划线
            word_split = re.split('\.|,|。|？|！|\n| |；|、|;|"：|。"|，"|!"',word)
            if len(word_split)>0:
                sub_word_index_list_in_word = []
                for i in range(np.random.randint(len(word_split))):  # 随机挑选n个word
                    a = np.random.randint(len(word_split)) #选取n个word的index
                    sub_word_index_list_in_word_sub =[(i.start(),i.end())for i in re.finditer(word_split[a],word)]
                    sub_index = np.random.randint(len(sub_word_index_list_in_word_sub))
                    sub_word_index_list_in_word.append(sub_word_index_list_in_word_sub[sub_index])
                    # if a not in random_word_index_list:
                    #     random_word_index_list.append(a)
                    # else:
                    #     continue

                thickness = np.random.randint(1, 3)
                for i in range(len(word)):
                    size = font.getsize(word[i])
                    chars_size.append(size)
                    #print(size)
                    # xStep = (rightBottomX - leftBottomX) // divded
                    # yStep = (rightBottomY - leftBottomY) // divded

                    # for i in range(0, divded - 1):
                    #print('draw cor',leftBottomX_tmp,leftBottomY)


                    if word[i]!= ' ':
                        if len(sub_word_index_list_in_word)!=0:
                            for index in sub_word_index_list_in_word:
                                if i >index[0]-1 and i <index[1]+1:
                                    if prob(0): #部分虚线
                                        draw_leftBottomX_tmp = leftBottomX_tmp+2
                                        draw_rightBottomX_tmp = leftBottomX_tmp + size[0] - 2
                                    else:
                                        draw_leftBottomX_tmp = leftBottomX_tmp
                                        draw_rightBottomX_tmp = leftBottomX_tmp + size[0]

                                    dst = cv2.line(word_img, (draw_leftBottomX_tmp , leftBottomY ),
                                               (draw_rightBottomX_tmp, leftBottomY ),
                                               color=line_color,
                                               thickness=thickness,
                                               lineType=cv2.LINE_AA)


                    leftBottomX_tmp = leftBottomX_tmp + size[0]
            # print(leftBottomX_tmp)
            # plt.figure('after line')
            # plt.imshow(word_img)
            # plt.show()
        else:  #全部虚线或者实线下划线
            thickness = np.random.randint(1, 3)
            for i in range(len(word)):
                size = font.getsize(word[i])
                chars_size.append(size)
                # print(size)
                # xStep = (rightBottomX - leftBottomX) // divded
                # yStep = (rightBottomY - leftBottomY) // divded

                # for i in range(0, divded - 1):
                # print('draw cor',leftBottomX_tmp,leftBottomY)
                if prob(0):#0.5概率是全部虚线
                    draw_leftBottomX_tmp = leftBottomX_tmp + 2
                    draw_rightBottomX_tmp = leftBottomX_tmp + size[0] - 2

                    if word[i] != ' ':
                        dst = cv2.line(word_img, (draw_leftBottomX_tmp, leftBottomY),
                                       (draw_rightBottomX_tmp, leftBottomY),
                                       color=line_color,
                                       thickness=thickness,
                                       lineType=cv2.LINE_AA)
                else: #全部实线
                    draw_leftBottomX_tmp = leftBottomX_tmp
                    draw_rightBottomX_tmp = leftBottomX_tmp + size[0]
                    dst = cv2.line(word_img, (draw_leftBottomX_tmp, leftBottomY),
                                   (draw_rightBottomX_tmp, leftBottomY),
                                   color=line_color,
                                   thickness=thickness,
                                   lineType=cv2.LINE_AA)
                leftBottomX_tmp = leftBottomX_tmp + size[0]




        return dst, text_box_pnts

    def apply_table_line(self, word_img, text_box_pnts, word_color,word,font):
        """
        共有 8 种可能的画法，横线横穿整张 word_img
        0/1/2/3: 仅单边（左上右下）
        4/5/6/7: 两边都有线（左上，右上，右下，左下）
        """
        dst = word_img
        option = random.choice(self.linestate.tableline_options)
        thickness = random.choice(self.linestate.tableline_thickness)
        line_color = word_color + random.randint(0, 10)

        top_y_offset = random.choice(self.linestate.tableline_y_offsets)
        bottom_y_offset = random.choice(self.linestate.tableline_y_offsets)
        left_x_offset = random.choice(self.linestate.tableline_x_offsets)
        right_x_offset = random.choice(self.linestate.tableline_x_offsets)

        def is_top():
            return option in [1, 4, 5]

        def is_bottom():
            return option in [3, 6, 7]

        def is_left():
            return option in [0, 4, 7]

        def is_right():
            return option in [2, 5, 6]

        if is_top():
            text_box_pnts[0][1] -= top_y_offset
            text_box_pnts[1][1] -= top_y_offset

        if is_bottom():
            text_box_pnts[2][1] += bottom_y_offset
            text_box_pnts[3][1] += bottom_y_offset

        if is_left():
            text_box_pnts[0][0] -= left_x_offset
            text_box_pnts[3][0] -= left_x_offset

        if is_right():
            text_box_pnts[1][0] += right_x_offset
            text_box_pnts[2][0] += right_x_offset

        if is_bottom():
            dst = cv2.line(dst,
                           (0, text_box_pnts[2][1]),
                           (word_img.shape[1], text_box_pnts[3][1]),
                           color=line_color,
                           thickness=thickness,
                           lineType=cv2.LINE_AA)

        if is_top():
            dst = cv2.line(dst,
                           (0, text_box_pnts[0][1]),
                           (word_img.shape[1], text_box_pnts[1][1]),
                           color=line_color,
                           thickness=thickness,
                           lineType=cv2.LINE_AA)

        if is_left():
            dst = cv2.line(dst,
                           (text_box_pnts[0][0], 0),
                           (text_box_pnts[3][0], word_img.shape[0]),
                           color=line_color,
                           thickness=thickness,
                           lineType=cv2.LINE_AA)

        if is_right():
            dst = cv2.line(dst,
                           (text_box_pnts[1][0], 0),
                           (text_box_pnts[2][0], word_img.shape[0]),
                           color=line_color,
                           thickness=thickness,
                           lineType=cv2.LINE_AA)

        return dst, text_box_pnts

    def apply_middle_line(self, word_img, text_box_pnts, word_color):
        y_center = int((text_box_pnts[0][1] + text_box_pnts[3][1]) / 2)

        img_mean = int(np.mean(word_img))
        thickness = np.random.choice(self.linestate.middleline_thickness, p=self.linestate.middleline_thickness_p)

        dst = cv2.line(word_img,
                       (text_box_pnts[0][0], y_center),
                       (text_box_pnts[1][0], y_center),
                       color=img_mean,
                       thickness=thickness,
                       lineType=cv2.LINE_AA)

        return dst, text_box_pnts
