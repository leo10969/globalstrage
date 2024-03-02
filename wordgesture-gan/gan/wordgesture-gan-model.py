import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow_addons.layers import SpectralNormalization
import torch

# GPUを選択
GPU = 1

# Encoder
class VariationalEncoder(tf.keras.Model):
    def __init__(self):
        super(VariationalEncoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(192, activation=tf.keras.layers.LeakyReLU()),
            layers.Dense(96, activation=tf.keras.layers.LeakyReLU()),
            layers.Dense(48, activation=tf.keras.layers.LeakyReLU()),
            layers.Dense(32, activation=tf.keras.layers.LeakyReLU())
        ])
        self.mu = layers.Dense(32)
        self.log_var = layers.Dense(32)
    
    def call(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var


# Discriminator
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = tf.keras.Sequential([
            SpectralNormalization(192, activation=tf.keras.layers.LeakyReLU()),
            SpectralNormalization(96, activation=tf.keras.layers.LeakyReLU()),
            SpectralNormalization(48, activation=tf.keras.layers.LeakyReLU()),
            SpectralNormalization(24, activation=tf.keras.layers.LeakyReLU()),
            SpectralNormalization(1, activation=tf.keras.layers.LeakyReLU())
        ])
    
    def call(self, x):
        return self.discriminator(x)
    
    #disc-loss
    def disc_loss(real_output, fake_output):
        loss_D = -torch.mean(discriminator(real_output)) + torch.mean(discriminator(fake_output))
        return loss_D


# Generator
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.bilstm1 = layers.Bidirectional(layers.LSTM(32, return_sequences=True, activation='tanh'), input_shape=(35, 32))
        self.bilstm2 = layers.Bidirectional(layers.LSTM(32, return_sequences=True, activation='tanh'))
        self.bilstm3 = layers.Bidirectional(layers.LSTM(32, return_sequences=True, activation='tanh'))
        self.bilstm4 = layers.Bidirectional(layers.LSTM(32, activation='tanh'))
        self.dense = layers.Dense(3, activation='tanh')
    
    def call(self, x):
        x = self.bilstm1(x)
        x = self.bilstm2(x)
        x = self.bilstm3(x)
        x = self.bilstm4(x)
        return self.dense(x)


#------------------訓練部分--------------------
# def train_wordgesture-gan():
#     return

#------------------実行部分--------------------
BATCH_SIZE = 512
learning_rate = 0.0002
#パラメータ設定
lambda_feat = 1
lambda_rec = 5
lambda_lat = 0.5
lambda_KLD = 0.05

with tf.device('/gpu:{}'.format(GPU)):
    generator = Generator()
    discriminator = Discriminator()
    encoder = VariationalEncoder()



    