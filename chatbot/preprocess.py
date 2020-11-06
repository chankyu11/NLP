import os
import re
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from konlpy.tag import Okt

filters = "([~.,!?\"':;])(])"
pad = "<PAD>"   # 어떤 의미도 없는 패딩 토큰
std = "<SOS>"   # 시작 토큰을 의미
end = "<END>"   # 종료 토큰을 의미
unk = "<UNK>"   # 사전에 없는 단어를 의미

pad_index = 0
std_index = 1
end_index = 2
unk_index = 3

marker = [pad, std, end, unk]
change_filter = re.compile(filters)

max_sequence = 25

def load_data(path):
    data_df = pd.read_csv(path, header= 0 )
    question, answer = list(data_df['Q']), list(data_df['A'])

    return question, answer

def data_tokenizer(data):
    words = []
    for sentence in data:
        sentence = re.sub(change_filter, "", sentence)
        for word in sentence.split():
            words.append(word)
    return [word for word in words if word]

def prepro_like_morphlized(data):
    morph_analyzer = Okt()
    result_data = []
    for seq in tqdm(data):
        morphlized_seq = " ".join(morph_analyzer.morphs(seq.replace(" ", "")))
        result_data.append(morphlized_seq)
    
    return result_data

def load_vocabulary(path, vocab_path, tokenize_as_morph=False):
    vocabulary_list = []
    if not os.path.exists(vocab_path):
        if(os.path.exists(path)):
            data_df = pd.read_csv(path, encoding = 'utf-8')
            question, answer = list(data_df['Q']), list(data_df['A'])
            if tokenize_as_morph:  # 형태소에 따른 토크나이져 처리
                question = prepro_like_morphlized(question)
                answer = prepro_like_morphlized(answer)
            
            data = []

            data.extend(question)
            data.extend(answer)

            words =data_tokenizer(data)
            words = list(set(words))
            words[:0] = marker

        with open(vocab_path , 'w', encoding = 'utf-8') as vocabulary_file:
            for word in words:
                vocabulary_file.write(word + '\n')
    with open(vocab_path, 'r', encoding = 'utf-8') as vocabulary_file:
        for line in vocabulary_file:
            vocabulary_list.append(line.strip())

    word2idx, idx2word = make_vocabulary(vocabulary_list)

    return word2idx, idx2word, len(word2idx)

def make_vocabulary(vocabulary_list):
    word2idx = {word: idx for idx, word in enumerate(vocabulary_list)}

    idx2word = {idx: char for idx, char in enumerate(vocabulary_list)}

    return word2idx, idx2word

def enc_processing(value, dictionary, tokenize_as_morph=False):
    sequences_input_idx = []
    sequence_len = []

    if tokenize_as_morph:
        value = prepro_like_morphlized(value)

    for sequence in value:
        sequence = re.sub(change_filter, "", sequence)
        sequence_idx = []
        for word in sequence.split():
            if dictionary.get(word) is not None:
                sequence_idx.extend([dictionary[word]])

            else:
                sequence_idx.extend([dictionary[unk]])
        if  len(sequence_idx) > max_sequence:
            sequence_idx = sequence_idx[:max_sequence]

        sequence_len.append(len(sequence_idx))

        sequence_idx += (max_sequence - len(sequence_idx)) * [dictionary[pad]]

        sequences_input_idx.append(sequence_idx)

    return np.asarray(sequences_input_idx), sequence_len

def dec_output_processing(value, dictionary, tokenize_as_morph=False):
    sequences_output_index = []
    sequences_len = []

    if tokenize_as_morph:
        value = prepro_like_morphlized(value)

    for sequence in value:
        sequence = re.sub(change_filter, "", sequence)
        sequence_idx = []
        sequence_idx = [dictionary[std]] + [dictionary[word] if word in dictionary else dictionary[unk] for word in sequence.split()]

        if len(sequence_idx) > max_sequence:
            sequence_idx = sequence_idx[:max_sequence]
        sequences_len.append(len(sequence_idx))
        sequence_idx += (max_sequence - len(sequence_idx)) * [dictionary[pad]]
        sequences_output_index.append(sequence_idx)

    return np.asarray(sequences_output_index), sequences_len


def dec_target_processing(value, dictionary, tokenize_as_morph=False):
    sequence_target_index = []

    if tokenize_as_morph:
        value = prepro_like_morphlized(value)
        
    for sequence in value:
        sequence = re.sub(change_filter,"", sequence)
        sequence_idx = [dictionary[word] if word in dictionary else dictionary[unk] for word in sequence.split()]
        if len(sequence_idx) >= max_sequence:
            sequnce_idx = sequence_idx[:max_sequence - 1] + [dictionary[end]]
        
        else:
            sequence_idx += (max_sequence - len(sequence_idx)) * [dictionary[pad]]

            sequence_target_index.append(sequence_idx)
    return np.asarray(sequence_target_index)
