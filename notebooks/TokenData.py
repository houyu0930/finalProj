#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import nltk
# from deepmoji.HY_tokenizer import tokenize

train_in = '/home/houyu/learning/finalProj/notebooks/train.txt'
dev_in = '/home/houyu/learning/finalProj/notebooks/dev.txt'
train_out = '/home/houyu/learning/data/train_out.txt'
dev_out = '/home/houyu/learning/data/dev_out.txt'

def tokenize(text):
    RE_MENTION = '@[a-zA-Z0-9_]+'
    RE_EMOJI = re.compile('[''\U0001F300-\U0001F64F''\U0001F680-\U0001F6FF''\u2600-\u2B55]+', 
                    re.UNICODE)
    intab = '“”‘’；：《》，。！？【】（）％＃＠＆１２３４５６７８９０'
    outtab = '""\'\';:<>,.!?[]()%#@&1234567890'
    table= {f:t for f,t in zip(intab,outtab)}

    for i in table:
        if i in text:
            text = text.replace(i, table[i])

    # text = re.sub(RE_MENTION, "@user", text)
    text = re.sub(RE_MENTION, '', text)
    text = re.sub('\\s+', ' ', text)
    text = text.replace('\\n', ' ')

    new_text = '' 
    for char in text:
        if RE_EMOJI.search(char):
            new_text += (" " + char + " ")
        else:
            new_text += char

    tokens = nltk.word_tokenize(new_text)

    # Remove empty strings
    result = [t.lower() for t in tokens if t.strip() is not '']
    return result


def main():
    # i = 0
    with open(dev_out, 'w') as fdev:
        with open(dev_in, 'r') as fdev_in:
            for line in fdev_in.readlines():
                sp_line = line.split('\t')
                label = sp_line[1]
                raw_sentence = sp_line[0]
                sentence_list = tokenize(raw_sentence)
                fdev.write(str(sentence_list))
                fdev.write('\t')
                fdev.write(label)
                '''        
                i += 1
                if i == 10:
                    break
                '''
    
    with open(train_out, 'w') as ftrain:
        with open(train_in, 'r') as ftrain_in:
            for line in ftrain_in.readlines():
                sp_line = line.split('\t')
                label = sp_line[1]
                raw_sentence = sp_line[0]
                sentence_list = tokenize(raw_sentence)
                ftrain.write(str(sentence_list))
                ftrain.write('\t')
                ftrain.write(label)
    
                

if __name__ == '__main__':
    main()
