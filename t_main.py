#!/usr/env/bin python3

"""
Generate training and test images.
"""
import traceback
import numpy as np

import multiprocessing as mp
from itertools import repeat
import os

import cv2

from libs.config import load_config
from libs.timer import Timer
from parse_args import parse_args
import libs.utils as utils
import libs.font_utils as font_utils
from textrenderer.corpus.corpus_utils import corpus_factory
from textrenderer.renderer import Renderer
from tenacity import retry

lock = mp.Lock()
counter = mp.Value('i', 0)
STOP_TOKEN = 'kill'

flags = parse_args()
cfg = load_config(flags.config_file)

fonts = font_utils.get_font_paths_from_list(flags.fonts_list)
bgs = utils.load_bgs(flags.bg_dir)

corpus = corpus_factory(flags.corpus_mode, flags.chars_file, flags.corpus_dir, flags.length)

renderer = Renderer(corpus, fonts, bgs, cfg,
                    height=flags.img_height,
                    width=flags.img_width,
                    clip_max_chars=flags.clip_max_chars,
                    debug=flags.debug,
                    gpu=flags.gpu,
                    strict=flags.strict)


def start_listen(q, fname):
    """ listens for messages on the q, writes to file. """

    f = open(fname, mode='a', encoding='utf-8')
    while 1:
        m = q.get()
        if m == STOP_TOKEN:
            break
        try:
            f.write(str(m) + '\n')
        except:
            traceback.print_exc()

        with lock:
            if counter.value % 1000 == 0:
                f.flush()
    f.close()


@retry
def gen_img_retry(renderer, img_index):
    try:
        return renderer.gen_img(img_index)
    except Exception as e:
        print("Retry gen_img: %s" % str(e))
        traceback.print_exc()
        raise Exception


def generate_img(img_index, q=None):
    global flags, lock, counter
    # Make sure different process has different random seed
    np.random.seed()

    im, word = gen_img_retry(renderer, img_index)

    base_name = '{:08d}'.format(img_index)

    if not flags.viz:
        fname = os.path.join(flags.save_dir, base_name + '.jpg')
        #im.dtype = 'uint8'
        #print (im)
        #print (cv2.t)
        #gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
        #ret, binary  = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imwrite(fname, im, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        label = "{} {}".format(base_name, word)

        if q is not None:
            q.put(label)

        with lock:
            counter.value += 1
            print_end = '\n' if counter.value == flags.num_img else '\r'
            if counter.value % 100 == 0 or counter.value == flags.num_img:
                print("{}/{} {:2d}%".format(counter.value,
                                            flags.num_img,
                                            int(counter.value / flags.num_img * 100)),
                      end=print_end)
    else:
        utils.viz_img(im)




if __name__ == "__main__":
    # It seems there are some problems when using opencv in multiprocessing fork way
    # https://github.com/opencv/opencv/issues/5150#issuecomment-161371095
    # https://github.com/pytorch/pytorch/issues/3492#issuecomment-382660636
    if utils.get_platform() == "OS X":
        mp.set_start_method('spawn', force=True)
    for i in range(0, 20):
        generate_img(i)
   
