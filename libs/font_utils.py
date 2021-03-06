import os
import pickle
import glob
from itertools import chain
import hashlib

from fontTools.ttLib import TTCollection, TTFont
from fontTools.unicode import Unicode

# from .utils import md5, load_chars


def load_chars(filepath):
    if not os.path.exists(filepath):
        print("Chars file not exists.")
        exit(1)

    ret = ''
    with open(filepath, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            ret += line.strip('\n')
    return ret

def md5(string):
    m = hashlib.md5()
    m.update(string.encode('utf-8'))
    return m.hexdigest()

def get_font_paths(fonts_dir):
    """
    Load font path recursively from a folder
    :param fonts_dir: folder contains ttf、otf or ttc format font
    :return: path of all fonts
    """
    print('Load fonts from %s' % os.path.abspath(fonts_dir))
    fonts = glob.glob(fonts_dir + '/**/*', recursive=True)
    fonts = list(filter(lambda x: os.path.isfile(x), fonts))
    print("Total fonts num: %d" % len(fonts))

    if len(fonts) == 0:
        print("Not found fonts in fonts_dir")
        exit(-1)
    return fonts


def get_font_paths_from_list(list_filename):
    with open(list_filename) as f:
        lines = f.readlines()
        fonts = [os.path.abspath(l.strip()) for l in lines]

        for font in fonts:
            if not os.path.exists(font):
                print("Font [%s] not exist." % font)
                exit(-1)

        print("Total fonts num: %d" % len(lines))
    font_dct = {}
    font_dct['chn'] = []
    font_dct['eng'] = []
    font_dct['jap'] = []
    font_dct['eng_strict'] = []
    font_dct['chn_strict'] = []
    font_dct['jap_checkbox'] = [os.path.abspath(line.strip()) for line in open('./data/fonts_list/jap_sp_symbol.txt','r').readlines()]
    font_dct['eng_checkbox'] = [os.path.abspath(line.strip()) for line in open('./data/fonts_list/eng_sp_symbol.txt','r').readlines()]

    for font_path in fonts:
        tmp_font_path = font_path.split('/')[-2]

        if 'chn' in tmp_font_path:

            if 'Traffic'.lower() not in font_path.lower() and '华文琥珀' not in font_path \
                and '华文新魏'not in font_path \
                and '华文行楷' not in font_path:
                font_dct['chn_strict'].append(font_path)

            font_dct['chn'].append(font_path)
            #font_dct['eng'].append(font_path)
        else:
            if 'jap' in tmp_font_path:
                font_dct['jap'].append(font_path)
                #font_dct['eng'].append(font_path)
            else:
                if 'walkway' not in font_path.lower() and 'raleway' not in font_path.lower() \
                        and 'courier' not in font_path.lower()\
                        and 'Sansation-LightItalic.ttf'.lower() not in font_path.lower()\
                        and 'CaviarDreams_BoldItalic.tt'.lower() not in font_path.lower()\
                        and 'Lato-SemiboldItalic.ttf'.lower() not in font_path.lower():
                    font_dct['eng_strict'].append(font_path)
                font_dct['eng'].append(font_path)
    print(font_dct)
    return font_dct


def load_font(font_path):
    """
    Read ttc, ttf, otf font file, return a TTFont object
    """

    # ttc is collection of ttf
    if font_path.endswith('ttc'):
        ttc = TTCollection(font_path)
        # assume all ttfs in ttc file have same supported chars
        return ttc.fonts[0]

    if font_path.endswith('ttf') or font_path.endswith('TTF') or font_path.endswith('otf'):
        ttf = TTFont(font_path, 0, allowVID=0,
                     ignoreDecompileErrors=True,
                     fontNumber=-1)

        return ttf


def check_font_chars(ttf, charset):
    """
    Get font supported chars and unsupported chars
    :param ttf: TTFont ojbect
    :param charset: chars
    :return: unsupported_chars, supported_chars
    """
    #chars = chain.from_iterable([y + (Unicode[y[0]],) for y in x.cmap.items()] for x in ttf["cmap"].tables)
    try:
        chars_int=set()
        for table in ttf['cmap'].tables:
            for k,v in table.cmap.items():
                chars_int.add(k)

        unsupported_chars = []
        supported_chars = []
        for c in charset:
            if ord(c) not in chars_int:
                unsupported_chars.append(c)
            else:
                supported_chars.append(c)

        ttf.close()
        return unsupported_chars, supported_chars
    except:
        return False


def get_fonts_chars(fonts, chars_file):
    """
    loads/saves font supported chars from cache file
    :param fonts: list of font path. e.g ['./data/fonts/msyh.ttc']
    :param chars_file: arg from parse_args
    :return: dict
        key -> font_path
        value -> font supported chars
    """
    out = {}

    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', '.caches'))
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    chars = load_chars(chars_file)
    chars = ''.join(chars)
    for language, font_list in fonts.items():
        for font_path in font_list:
    # for font_path in fonts:
            string = ''.join([font_path, chars])
            file_md5 = md5(string)

            cache_file_path = os.path.join(cache_dir, file_md5)

            if not os.path.exists(cache_file_path):
                try:
                    ttf = load_font(font_path)
                    _, supported_chars = check_font_chars(ttf, chars)
                    # if len(supported_chars) == 15:
                    #     print(font_path)
                    print('len(supported_chars)',len(supported_chars))
                    # print('Save font(%s) supported chars(%d) to cache' % (font_path, len(supported_chars)))

                    with open(cache_file_path, 'wb') as f:
                        pickle.dump(supported_chars, f, pickle.HIGHEST_PROTOCOL)
                except:
                    continue
            else:
                try:
                    with open(cache_file_path, 'rb') as f:
                        supported_chars = pickle.load(f)
                        # if len(supported_chars) == 2:
                            # print('supported_chars',supported_chars,cache_file_path)


                    # print('Load font(%s) supported chars(%d) from cache' % (font_path, len(supported_chars)))
                except:
                    print('这个字体不行' ,font_path)
                    continue

            out[font_path] = supported_chars

    return out


def get_unsupported_chars(fonts, chars_file):
    """
    Get fonts unsupported chars by loads/saves font supported chars from cache file
    :param fonts:
    :param chars_file:
    :return: dict
        key -> font_path
        value -> font unsupported chars
    """
    charset = load_chars(chars_file)
    charset = ''.join(charset)
    fonts_chars = get_fonts_chars(fonts, chars_file)
    fonts_unsupported_chars = {}
    for font_path, chars in fonts_chars.items():
        unsupported_chars = list(filter(lambda x: x not in chars, charset))
        fonts_unsupported_chars[font_path] = unsupported_chars
    return fonts_unsupported_chars


if __name__ == '__main__':
    font_paths = get_font_paths('/fengjing/data_script/OCR_textrender/data/fonts/jap')
    # char_file = '../data/chars/chn.txt'
    char_file = '../2.txt'
    chars = get_fonts_chars(font_paths, char_file)
    # print(chars)
    num = 1
    fonts_list = [line.strip() for line in open('../data/fonts_list/chn.txt','r').readlines()]
    eng_sp_symbol_txt = open('../data/fonts_list/jap_sp_symbol.txt','w')
    for char in chars.keys():
        if len(chars[char]) == 2:
            new_char = './'+char.split('OCR_textrender/')[1]
            if new_char in fonts_list:
                eng_sp_symbol_txt.writelines('./'+char.split('OCR_textrender/')[1]+'\n')
                print(char,chars[char])
                num+=1
    print(num)
