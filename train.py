import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation, Conv2DTranspose, Reshape, BatchNormalization, LeakyReLU, Dropout, Conv2D, Flatten
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm

class GenerativeAgent():
    def __init__(self):
        self.cross_entropy = BinaryCrossentropy(from_logits=True)
        self.generartor_optimizer = Adam(0.001)
        self.discriminator_optimizer = Adam(0.001)
        self.EPOCHS = 10
        self.NOISE_DIMENSION = 100
        self.BATCH_SIZE = 64
        self.CHECKPOINT_PATH = 'Generativemodel-crossentropy-noise100/checkpoint.ckpt'
        pass

    def create_generator(self):
        model = Sequential()

        model.add(Dense(7*7*7, input_shape=(100,), activation='relu'))
        model.add(Reshape((7,7,7)))

        model.add(Conv2DTranspose(64, 5, (2,2), padding='same', activation='relu'))
        model.add(Conv2DTranspose(32, 5, (2,2), padding='same', activation='relu'))
        model.add(Conv2DTranspose(1, 5, (1,1), padding='same', activation='sigmoid'))

        self.generator = model
        return self.generator

    def create_discriminator(self):
        model = Sequential()

        model.add(Conv2D(32, 5,(2,2), input_shape=(28,28,1), activation='relu'))
        model.add(Conv2D(64, 5,(2,2), activation='relu'))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        self.discriminator = model
        return self.discriminator
    
    def create_model(self):
        self.create_generator()
        self.create_discriminator()
        self.initialize_checkpoint()

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)
    
    def discriminator_loss(self, fake_output, real_output):
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        return real_loss + fake_loss

    def step(self, X):
        noise = tf.random.normal([self.BATCH_SIZE, self.NOISE_DIMENSION])
        self.generator(noise)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise)

            real_output = self.discriminator(X)
            fake_output = self.discriminator(generated_images)
            
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(fake_output, real_output)
        
        gen_gradient = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradient = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generartor_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(disc_gradient, self.discriminator.trainable_variables))
        return (gen_loss.numpy(), disc_loss.numpy())

    def train(self, images):
        for episode in tqdm(range(0, self.EPOCHS), ascii=True, unit='episode'):
            image_batch = []
            for index, (image) in enumerate(images):
                image_batch.append(image)
                if len(image_batch) >= self.BATCH_SIZE:
                    loss = self.step(np.array(image_batch).reshape(-1,28,28,1))
                    image_batch = []
                    if not index % 100: 
                        self.checkpoint.save(file_prefix=self.CHECKPOINT_PATH)
                        print(f"episode: {episode} index: {index} loss: {loss}")

    def initialize_checkpoint(self):
        self.checkpoint = tf.train.Checkpoint(generator=self.generator, discriminator=self.discriminator)

agent = GenerativeAgent()
agent.create_model()

(train_data,y), (_, _) = tf.keras.datasets.mnist.load_data()
X = train_data/255.

agent.train(X)
print('Training completed')