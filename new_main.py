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
import signal
from contextlib import contextmanager
from hanging_threads import start_monitoring

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)



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


lock = mp.Lock()
#一旦是函数参数的问题，这个调用就停不下来了 - -!
@retry
def gen_img_retry(renderer, img_index, log_f, lock):
    try:
                #long_function_call()
        return renderer.gen_img(img_index, log_f, lock)
    except Exception as e:
        #print("Retry gen_img: %s" % str(e))
        traceback.print_exc()
        raise Exception


def generate_img(process_id, process_num, lock):
    # Make sure different process has different random seed
    np.random.seed()
    num_img_of_one_process = flags.num_img // process_num
    start_id = process_id * num_img_of_one_process
    end_id = (process_id + 1) * num_img_of_one_process
    print ("Process_id : ", process_id, " Process_num : ", process_num)
    print ("start_id : ", start_id, " end_id : ", end_id)
    counter_num = 0
    #存储label
    log_f = open(os.path.join(flags.save_dir, "log_" + str(process_id) + ".log"), 'w')
    try:
        label_fname = os.path.join(flags.save_dir, "label_" + str(process_id) + ".txt")
        f = open(label_fname, mode='w', encoding='utf-8')
        for img_index in range(start_id, end_id):
            im, word = gen_img_retry(renderer, img_index, log_f, lock)
            base_name = '{:08d}'.format(img_index)
            #print ("Generate Image : ", img_index)
            fname = os.path.join(flags.save_dir, base_name + '.jpg')
            cv2.imwrite(fname, im)
            label = "{} {}".format(base_name, word)
            f.write(label + '\n')
            f.flush()
            counter_num += 1
            if counter_num % 10000 == 0:
                print("{}/{} {:2d}%".format(counter_num * process_num, flags.num_img, int(counter_num * process_num / flags.num_img * 100)))
    except Exception as e:
        #log_f.write(e.message + "\n")
        #log_f.flush()
        traceback.print_exc()

def get_num_processes(flags):
    processes = flags.num_processes
    if processes is None:
        processes = max(os.cpu_count(), 2)
    return processes


if __name__ == "__main__":
    # It seems there are some problems when using opencv in multiprocessing fork way
    # https://github.com/opencv/opencv/issues/5150#issuecomment-161371095
    # https://github.com/pytorch/pytorch/issues/3492#issuecomment-382660636
    #为了解决cv2 多进程会出现阻塞的情况
    #mp.set_start_method('spawn')
    #一定提前运行一次
    #mp.set_start_method('spawn', force=True)
    generate_img(1, flags.num_img, lock)
    timer = Timer(Timer.SECOND)
    timer.start()
    process_num = get_num_processes(flags)
    monitoring_thread = start_monitoring()
    #process_num = 1
    for i in range(process_num):
        p = mp.Process(target=generate_img ,args=(i,process_num,lock,))
        p.start()
    timer.end("Finish generate data")

