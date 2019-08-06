import os
import io
import json
import gzip
import argparse
import collections
from time import time
from nltk.tokenize import TweetTokenizer
import statistics
import copy


class qclass():
    def __init__(self):
        super(qclass, self).__init__()

        self.tknzr = TweetTokenizer(preserve_case=False)
        self.color_tokens = []
        self.shape_tokens = []
        self.size_tokens = []
        self.texture_tokens = []
        self.action_tokens = []
        self.spatial_tokens =[]
        self.number_tokens = []
        self.object_tokens = []
        self.super_tokens = []

        self.attribute_tokens=[]

        #self.is_it_tokens = ['is', 'it']
        #self.a_tokens = ['the','a','an']

        word_annotation = '../data/word_annotation'

        with open(word_annotation) as f:
            lines = f.readlines()
            for line in lines:
        #         print(line)
                word1,word2 = line.split('\t')
        #         print(word1,word2)
                word1 = word1.strip().lower()
                word2 = word2.strip().lower()
                if word2 == 'color':
                    self.color_tokens.append(word1)
                    self.attribute_tokens.append(word1)
                elif word2 == 'shape':
                    self.shape_tokens.append(word1)
                    self.attribute_tokens.append(word1)
                elif word2 == 'size':
                    self.size_tokens.append(word1)
                    self.attribute_tokens.append(word1)
                elif word2 == 'texture':
                    self.texture_tokens.append(word1)
                    self.attribute_tokens.append(word1)
                elif word2 == 'action':
                    self.action_tokens.append(word1)
                    self.attribute_tokens.append(word1)
                elif word2 == 'spatial':
                    self.spatial_tokens.append(word1)
                    self.attribute_tokens.append(word1)
                elif word2 == 'number':
                    self.number_tokens.append(word1)
                    self.attribute_tokens.append(word1)
                elif word2 == 'object':
                    self.object_tokens.append(word1)
                elif word2 == 'super-category':
                    self.super_tokens.append(word1)
    def que_classify_attribute(self, que):
		#To Classify based on Attribute, Object and Super-category 
        que = que.lower()
        tokens = self.tknzr.tokenize(que)
        cat = '<NA>'
        if cat == '<NA>':
            for tok in tokens:
                if tok in self.attribute_tokens:
                    cat = '<attribute>'
                    break
            if cat == '<NA>':
                for tok in tokens:
                    if tok in self.object_tokens:
                        cat = '<object>'
                        break
                if cat == '<NA>':
                    for tok in tokens:
                        if tok in self.super_tokens:
                            cat = '<super-category>'
                            break
        return  cat

    def que_classify_multi(self, que):
		# Question Classification 
        que = que.lower()
        tokens = self.tknzr.tokenize(que)
        cat = '<NA>'
        if cat == '<NA>':
            tmp_cat=[]
            for tok in tokens:
                if tok in self.color_tokens:
                    tmp_cat.append('<color>')
                    break
            for tok in tokens:
                if tok in self.shape_tokens:
                    tmp_cat.append('<shape>')
                    break
            for tok in tokens:
                if tok in self.action_tokens:
                    tmp_cat.append('<action>')
                    break
            for tok in tokens:
                if tok in self.size_tokens:
                    tmp_cat.append('<size>')
                    break
            for tok in tokens:
                if tok in self.texture_tokens:
                    tmp_cat.append('<texture>')
                    break
            for tok in tokens:
                if tok in self.action_tokens:
                    tmp_cat.append('<action>')
                    break
            for tok in tokens:
                if tok in self.spatial_tokens or tok in self.number_tokens:
                    tmp_cat.append('<spatial>')
                    break

            if tmp_cat:
                cat = tmp_cat
            if cat == '<NA>':
                for tok in tokens:
                    if tok in self.object_tokens:
                        cat = '<object>'
                        break
                if cat == '<NA>':
                    for tok in tokens:
                        if tok in self.super_tokens:
                            cat = '<super-category>'
                            break


        return cat

