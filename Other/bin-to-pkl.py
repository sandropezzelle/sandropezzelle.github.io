'''     @author: thenghiapham
        Usage: python bin2text.py input_file output_file
        input_file: path to (word2vec) bin file
        output_file: path to pkl file'''

#This script will convert a .bin file from word2vec to a .pkl semantic space
#to be used as input for dissect toolkit

import sys
import struct
import numpy as np
from composes.semantic_space.space import Space
from composes.matrix.dense_matrix import DenseMatrix
from composes.utils import io_utils

input_file = "../../stuff/vectors.bin"
output_file = "../../vectors.pkl"

argv = sys.argv[1:]


def readWord(i_stream):
    word = ""
    c = i_stream.read(1)
    while (c != ' ' and c != '\n'):
        word = word + c
        c = i_stream.read(1)
    return word

with open(input_file, 'r') as i_stream:
    vocab_size = int(readWord(i_stream))
    vector_size = int(readWord(i_stream))
    print vocab_size, vector_size
    cooc_mat = np.zeros((vocab_size, vector_size))
    vocabs = []
    for i in range(vocab_size):
        vocabs.append(readWord(i_stream))
        for j in range(vector_size):
            cooc_mat[i,j] = struct.unpack("<f", i_stream.read(4))[0]
        i_stream.read(1)
    print cooc_mat[0,0], cooc_mat[0,1]
    print cooc_mat[1,0], cooc_mat[1,1]
    space = Space(DenseMatrix(cooc_mat), vocabs, [])
    io_utils.save(space, output_file)
