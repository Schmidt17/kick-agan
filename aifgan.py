import os
import numpy as np
import AudioLoader
import matplotlib.pyplot as plt
import tensorflow as tf
tf.random.set_seed(1000)

def discriminator(seq_len):
	""" Discriminator net with binary output: Real (1) or fake (0) """
	model = tf.keras.Sequential([
			tf.keras.layers.Conv1D(filters=16, kernel_size=5, 
				strides=2, padding='same', activation='linear', input_shape=(seq_len, 1)),
			tf.keras.layers.ReLU(),
			tf.keras.layers.Dropout(0.3),
			# shape=(None, seq_len*2, 16)

			tf.keras.layers.Conv1D(filters=32, kernel_size=5, 
				strides=2, padding='same', activation='linear'),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.ReLU(),
			# shape=(None, seq_len*4, 32)

			tf.keras.layers.Conv1D(filters=64, kernel_size=5, 
				strides=2, padding='same', activation='linear'),
			# tf.keras.layers.BatchNormalization(),
			tf.keras.layers.ReLU(),
			tf.keras.layers.Dropout(0.3),
			# shape=(None, seq_len*8, 64)

			tf.keras.layers.Flatten(),
			# shape=(None, seq_len*8*64)
			tf.keras.layers.Dense(16, activation="relu"),
			# shape=(None, 16)
			tf.keras.layers.Dense(1, activation="sigmoid")
			# shape=(None, 1)
		])
	return model

def generator(seq_len):
	""" Generator net which outputs an audio sample of length `seq_len` from a seed of length 100"""
	assert seq_len % 4 == 0, "Sequence length must be divisible by 4"
	dense_dim = (seq_len // 4)

	model = tf.keras.Sequential([
			tf.keras.layers.Dense(dense_dim * 64, use_bias=False, input_shape=(100,), activation='linear'),
			# tf.keras.layers.BatchNormalization(),
			tf.keras.layers.LeakyReLU(),
			# shape=(None, seq_len/4 * 64)

			tf.keras.layers.Reshape((1, dense_dim, 64)),
			# shape=(None, 1, seq_len/4, 64)

			tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(1, 8),  use_bias=False,
				strides=(1, 1), padding='same', activation='linear'),
			# tf.keras.layers.BatchNormalization(),
			tf.keras.layers.ReLU(),
			# shape=(None, 1, seq_len/4, 128)

			tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(1, 8),  use_bias=False,
				strides=(1, 2), padding='same', activation='linear'),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.LeakyReLU(),
			# shape=(None, 1, seq_len/2, 32)

			tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(1, 8),  use_bias=False,
				strides=(1, 2), padding='same', activation='relu'),
			# # tf.keras.layers.BatchNormalization(),
			# tf.keras.layers.LeakyReLU(),
			# shape=(None, 1, seq_len, 32)

			tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(1, 16),  use_bias=False,
				strides=(1, 1), padding='same', activation='tanh'),
			# shape=(None, 1, seq_len, 1)

			tf.keras.layers.Reshape((seq_len, 1))
			# shape=(None, seq_len, 1)
		])
	return model

# load all kick samples from Analog folder
downs = 60
aif_paths = [os.path.join("Analog", f) for f in os.listdir("Analog") if f.endswith(".aif")]
audio_data = []
for path in aif_paths:
	audio_data.append(AudioLoader.load_aif(path, downsample_by=downs))

max_len = max([arr.shape[0] for arr in audio_data])
if max_len % 4 != 0:  # make sure the desired length is a multiple of 4, to suit generator design
	max_len += 4 - (max_len % 4)
# pad all the files with their last value to the max_len
audio_data = np.array([np.pad(data, (0, max_len-data.shape[0]), mode='edge') for data in audio_data] * 8)
audio_data = audio_data[np.random.permutation(len(audio_data))]  # shuffle

ce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def disc_loss(real_output, fake_output):
	real_loss = ce(tf.ones_like(real_output), real_output)
	fake_loss = ce(tf.zeros_like(fake_output), fake_output)
	return real_loss + fake_loss

def generator_loss(fake_output):
	return ce(tf.ones_like(fake_output), fake_output)

lr = .5e-4  		# learning rate
num_epochs = 5000
batch_size = 8*11
noise_len = 100		# length of noise seed for generator
num_gens = 8*11
seed = tf.random.normal((num_gens, noise_len))

disc_opt = tf.keras.optimizers.Adam(lr)
gen_opt = tf.keras.optimizers.Adam(2.*lr)

train_samples = audio_data.reshape((audio_data.shape[0] // batch_size, batch_size, max_len, 1))

@tf.function
def train_step(train_audio):
	local_noise = tf.random.normal((batch_size, noise_len))

	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape1, tf.GradientTape() as disc_tape2:
		generated_samples = G(local_noise, training=True)

		real_output = D(train_audio, training=True)
		fake_output = D(generated_samples, training=True)

		gen_loss = generator_loss(fake_output)
		# d_loss = disc_loss(real_output, fake_output)
		d_loss_real = ce(tf.ones_like(real_output), real_output)
		d_loss_fake = ce(tf.zeros_like(fake_output), fake_output)

	gen_gradients = gen_tape.gradient(gen_loss, G.trainable_variables)
	disc_gradients1 = disc_tape1.gradient(d_loss_real, D.trainable_variables)
	disc_gradients2 = disc_tape2.gradient(d_loss_fake, D.trainable_variables)

	gen_opt.apply_gradients(zip(gen_gradients, G.trainable_variables))
	disc_opt.apply_gradients(zip(disc_gradients1, D.trainable_variables))
	disc_opt.apply_gradients(zip(disc_gradients2, D.trainable_variables))

D = discriminator(max_len)
G = generator(max_len)

def train(data, epochs):
	for epoch in range(epochs):
		print(f"Start epoch {epoch}")
		for sample_batch in data:
			train_step(sample_batch)

		# generate and plot a sample for progress monitoring (always the same seed for consistency)
		gen = G(seed, training=False)
		update_plot(gen[0, :, 0])

# Set up a plot of a generated sample that updates while training
fig, ax = plt.subplots(1, 1)
rdata = np.random.normal(size=max_len)  # some random data for initializing the plot
pls = ax.plot(audio_data[5], color='red')
pl = ax.plot(rdata, color='blue')[0]
plt.show(block=False)

def update_plot(data):
	pl.set_ydata(data)
	fig.canvas.draw()
	plt.pause(0.01)

train(train_samples, num_epochs)  # train the thing

G.save("generator.h5")

# save example as audio file (kinda hacky conversion from float[0., 1.] to signed int16)
gen = G(seed, training=False)
AudioLoader.save_aif(gen[0, :, 0].numpy() * 2**15, "test.aif", downs)

# show again so the window doesn't close and we can LOok At IT!
plt.show(block=True)