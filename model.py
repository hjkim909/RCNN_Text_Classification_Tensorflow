import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model


class RCNN(Model):
  def __init__(self, vocab_size, embedding_dim, hidden_size, hidden_size_linear, class_num, dropout, **kwargs):
    super(RCNN, self).__init__(**kwargs)
    self.embedding = keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
    self.lstm = keras.layers.LSTM(hidden_size, dropout= dropout, return_sequences=True)
    self.backward_lstm = keras.layers.LSTM(hidden_size, dropout=dropout, return_sequences=True, go_backwards= True)
    self.Bilstm = keras.layers.Bidirectional(self.lstm, backward_layer=self.backward_lstm)
    self.W = Dense(hidden_size_linear)
    self.fc = Dense(class_num, activation= 'softmax')

  def call(self, x):
    x_emb = self.embedding(x)
    output = self.Bilstm(x_emb)
    output = tf.concat([output, x_emb], 2)
    output = tf.transpose(keras.activations.tanh(self.W(output)), perm = [0,2,1])
    output = tf.keras.layers.MaxPool1D(output.shape[2])(output)
    output = tf.keras.layers.Flatten()(output)
    return self.fc(output)

