import aifc
import numpy as np

def load_aif(fpath, downsample_by=1):
	with aifc.open(fpath) as audio_file:
		N_frames = audio_file.getnframes()
		audio_frames = audio_file.readframes(N_frames)
	audio_arr = load24bitbufferas(audio_frames, np.float32)
	# separate the two channels (LRLRLRLRLR....)
	audio_arr = audio_arr.reshape((-1, 2))
	# and mix them together
	mono_mix = np.sum(audio_arr, axis=-1)
	# normalize to [0., 1.]
	# mono_mix -= np.min(mono_mix)
	mono_mix /= np.max(np.abs(mono_mix))
	mono_mix = mono_mix[::downsample_by]  # downsampling

	return mono_mix

def save_aif(data, fpath, ds=1):
	with aifc.open(fpath, mode="wb") as f:
		f.setnchannels(1)
		f.setsampwidth(2)
		f.setframerate(96000 // ds)
		m = np.min(data)
		data -= m
		data = data.astype('>i2')
		data += m.astype('>i2')
		f.writeframes(data.tostring())


def load24bitbufferas(buff, dtype=np.float32, endianess='big'):
	assert endianess in ['big', 'small'], f"Unrecognized endianess specifier '{endianess}'"

	# load the 24-bit buffer string into a 32-bit numpy array:
	# load bytewise (in chunks of int8); group 3 bytes (24 bit) together as one frame
	raw_audio_bytes = np.frombuffer(buff, 'b').reshape((-1, 3))
	if endianess == 'small':
		raw_audio_bytes = raw_audio_bytes[:, ::-1]  # reverse byte order
	audio_arr = np.zeros((raw_audio_bytes.shape[0], 4), dtype='b')  # template array for 32-bit groups
	audio_arr[:, :-1] = raw_audio_bytes  # leave the last byte (least significant byte - file is big-endian) at zero (to correctly copy the sign bit)
	audio_arr = audio_arr.flatten().view('>i4') >> 8  # flatten the bytes out and return a big-endian signed 32-bit view of the array, shift the bits back to right significance, leaving sign correct, since python bitshift does not shift the sign bit

	return audio_arr.astype(dtype)  # cast the array to the desired type
