# WORD-EMBEDDING - USING WORD2VEC TO LEARN WORD EMBEDDINGS:

import re
import string

import tensorflow as tf

from code.model_using_vader import Vader_Model
from tensorflow.keras import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras import utils
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np
# We only want the dataset we don't want the sentiment analysis
simple_Model = Vader_Model()
# load headlines dataset
headlines = simple_Model.get_headline_values()
# construct a textLineDataset objec
#text_ds = tf.data.TextLineDataset(headlines).filter(lambda x: tf.cast(tf.strings.length(x), bool))
#np.savetxt('C:\\Users\\chuks\\PycharmProjects\\FYP\\dataset\\text.txt',headlines,fmt='%s', delimiter=',')
#print(headlines)
path_to_file = 'C:\\Users\\chuks\\PycharmProjects\\FYP\\dataset\\text'

raw_train_ds = preprocessing.text_dataset_from_directory(
    path_to_file,
    label_mode=None)
text_ds = tf.data.TextLineDataset("C:\\Users\\chuks\\PycharmProjects\\FYP\\dataset\\dataset.csv")

#.filter(lambda x: tf.cast(tf.strings.length(x), bool))
# lower case text and remove punctuation:
# def custom_standardization(input_data):
#     lowercase = tf.strings.lower(input_data)
#     return tf.strings.regex_replace(lowercase,
#                                     '[%s]' % re.escape(string.punctuation), '')
#
# # Define the vocabulary size and number of words in a sequence.
# vocab_size = 4096
# sequence_length = 10
#
# # Use the text vectorization layer to normalize, split, and map strings to
# # integers. Set output_sequence_length length to pad all samples to same length.
# vectorize_layer = TextVectorization(
#     standardize=custom_standardization,
#     max_tokens=vocab_size,
#     output_mode='int',
#     output_sequence_length=sequence_length)
#
print(text_ds.batch(1024))
#inverse_vocab = vectorize_layer.get_vocabulary()

#print(inverse_vocab[:20])
