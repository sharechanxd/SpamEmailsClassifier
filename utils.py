import numpy as np
import os
import math
import re


class VocProc():
    def __init__(self, filename):
        self.filename = filename
        # self.labels = []
        # self.SMS_words = []

    def data_loader(self):
        """
        Load SMS text data from txt file
        :param self:
        :returns:
            SMS_words represent the txt data along with the labels.
        """
        labels = []
        SMS_words = []
        with open(self.filename, 'r',encoding = 'utf-8') as f:
            data = f.readlines()
            for line in data:
                # each row composed of two column
                d = line.strip().split('\t')
                if d[0] == 'ham':
                    labels.append(0)
                else:
                    labels.append(1)
                p = re.compile(r'[^a-zA-Z]|\d')
                words = p.split(d[1])
                words = [w.lower() for w in words if len(w) > 0]
                SMS_words.append(words)

        return SMS_words, labels

    def creatVocList(self):
        voclist = set()
        sms_words,_ = self.data_loader()
        for w in sms_words:
            voclist.update(set(w))
        return list(voclist)

    def creatVocVector(self):
        sms_words, _ = self.data_loader()
        vocabulary_list = self.creatVocList()
        vocab_marked_list = []
        for words in sms_words:
            vocab_marked = [0] * len(vocabulary_list)
            for word in words:
                if word in vocabulary_list:
                    vocab_marked[vocabulary_list.index(word)] += 1
            vocab_marked_list.append(vocab_marked)
        return vocab_marked_list








