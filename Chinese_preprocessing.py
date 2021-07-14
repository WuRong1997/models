import thulac
import jieba
import jieba.posseg as pseg
from collections import Counter

thu = thulac.thulac()
word_counter = Counter() #统计词频
file_path = '***' #需要分词的文本文档
thu_char = ['v', 'a', 'n'] #词性选择
jieba_char = ['an', 'v', 'n', 't']

#thulac分词
with open(file_path, 'r') as f:
    lines = f.readlines()
    for elem in lines:
        data = elem.strip()
        result = thu.cut(data, text=True) #type:str
        words_char = result.split(' ')
        for elem in words_char: #知道_v
            if elem[-1] in thu_char:
                word_counter.update([elem[:-2]])

#jieba分词
with open(file_path, 'r') as f:
    lines = f.readlines()
    for elem in lines:
        data = elem.strip()
        word_char = pseg.cut(data)
        for elem in words_char:
            if elem.flag in jieba_char:
                word_counter.update([elem.word]])
