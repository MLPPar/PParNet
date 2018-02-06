"""Defines a small unidirectional GRU encoder-decoder model without attention mechanism."""

import tensorflow as tf
import opennmt as onmt

def model():
  return onmt.models.SequenceToSequence(
      source_inputter=onmt.inputters.WordEmbedder(
          vocabulary_file_key="source_words_vocabulary",
          embedding_size=512),
      target_inputter=onmt.inputters.WordEmbedder(
          vocabulary_file_key="target_words_vocabulary",
          embedding_size=512),
      encoder=onmt.encoders.UnidirectionalRNNEncoder(
          num_layers=2,
          num_units=512,
          cell_class=tf.contrib.rnn.GRUCell,
          dropout=0.3,
          residual_connections=False),
      decoder=onmt.decoders.RNNDecoder(
          num_layers=2,
          num_units=512,
          bridge=onmt.layers.CopyBridge(),
          cell_class=tf.contrib.rnn.GRUCell,
          dropout=0.3,
          residual_connections=False))
