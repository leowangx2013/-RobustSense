import numpy as np
from os.path import exists
import os
import scipy.io as scio
from meta_loader import load_meta
from scipy import signal

PATH_A = "/home/tianshi/data/ACIDS/ACIDSData_public_testset-mat/Acoustics"
PATH_S = "/home/tianshi/data/ACIDS/ACIDSData_public_testset-mat/Seismic"

TEST_FILE = "/home/tianshi/GAN_Vehicle/acid_dataset_utils/test_file_list.txt"

# FILES_train = ['Gv1a1002.mat', 'Gv1a2002.mat', 'Gv1a1012.mat', 'Gv1a2012.mat', 'Gv1b1020.mat', 'Gv1b2020.mat', 
# 	'Gv1b1164.mat', 'Gv1b2164.mat', 'Gv1b2022.mat', 'Gv1c1020.mat', 'Gv1c2020.mat', 'Gv1c1030.mat', 'Gv1c2030.mat',
# 	'Gv1c1084.mat', 'Gv1c2022.mat', 'Gv1d1132.mat', 'Gv2a1078.mat', 'Gv2a1080.mat', 'Gv2b1008.mat', 'Gv2b2008.mat',
# 	'Gv2c1008.mat', 'Gv2c1010.mat', 'Gv2c2008.mat', 'Gv2c2010.mat', 'Gv3c1034.mat', 'Gv3c1036.mat', 'Gv4c1040.mat', 
# 	'Gv4c1042.mat', 'Gv4c1044.mat', 'Gv4c1096.mat', 'Gv4d1036.mat', 'Gv4d2036.mat', 'Gv5a1046.mat', 'Gv5a2046.mat',
# 	'Gv5a1048.mat', 'Gv5a2048.mat', 'Gv5a1108.mat', 'Gv5b2046.mat', 'Gv5c1046.mat', 'Gv5c2046.mat', 'Gv5c1102.mat',
# 	'Gv5d1120.mat', 'Gv6c1050.mat', 'Gv6c2050.mat', 'Gv6c1052.mat', 'Gv6d1014.mat', 'Gv6d2014.mat', 'Gv6d1016.mat',
# 	'Gv6d2016.mat', 'Gv6d1018.mat', 'Gv6d2018.mat', 'Gv6d1086.mat', 'Gv6d2086.mat', 'Gv7a1068.mat', 'Gv7a1130.mat',
# 	'Gv8a1056.mat', 'Gv8b1058.mat', 'Gv8c1062.mat', 'Gv8c2062.mat', 'Gv8c1118.mat', 'Gv8c1210.mat', 'Gv8c1212.mat',
# 	'Gv8d1108.mat', 'Gv9a1060.mat', 'Gv9c1070.mat', 'Gv9c2070.mat', 'Gv9c1126.mat', 'Gv9d1126.mat']

# FILES_val = ['Gv1a1136.mat', 'Gv1a2014.mat', 'Gv1b1166.mat', 'Gv1b2166.mat', 'Gv1c1086.mat', 'Gv1c1202.mat', 
# 	'Gv1d1134.mat', 'Gv2a1052.mat', 'Gv2a2052.mat', 'Gv2b1010.mat', 'Gv2b2010.mat', 'Gv2c1080.mat', 'Gv3c1090.mat',
# 	'Gv4c1098.mat', 'Gv4c1100.mat', 'Gv4d1038.mat', 'Gv4d2038.mat', 'Gv5a1148.mat', 'Gv5a2148.mat', 'Gv5b2048.mat',
# 	'Gv5c1048.mat', 'Gv5c2048.mat', 'Gv6c2052.mat', 'Gv6d1088.mat', 'Gv6d2088.mat', 'Gv7c1056.mat', 'Gv8a2056.mat',
# 	'Gv8b2058.mat', 'Gv8c1214.mat', 'Gv8c1216.mat', 'Gv8d1110.mat', 'Gv9a1122.mat', 'Gv9c1128.mat']

# FILES_test = ['Gv1a1004.mat', 'Gv1a2134.mat', 'Gv1b2168.mat', 'Gv1b2170.mat', 'Gv1c1032.mat', 'Gv1c1204.mat', 
# 	'Gv1c2032.mat', 'Gv1d1136.mat', 'Gv2a1140.mat', 'Gv2a2140.mat', 'Gv2b1198.mat', 'Gv2b2198.mat', 'Gv2c1082.mat',
# 	'Gv3c1092.mat', 'Gv4c1206.mat', 'Gv4c1208.mat', 'Gv4d1040.mat', 'Gv4d2040.mat', 'Gv5a1150.mat', 'Gv5a2150.mat',
# 	'Gv5c1104.mat', 'Gv5d1122.mat', 'Gv6c1106.mat', 'Gv6d1090.mat', 'Gv6d2090.mat', 'Gv7c1058.mat', 'Gv8a2060.mat',
# 	'Gv8b2060.mat', 'Gv8c1218.mat', 'Gv8c1220.mat', 'Gv8d1112.mat', 'Gv9c1072.mat', 'Gv9c2072.mat', 'Gv9d1128.mat']

FILES = ['Gv1a1002.mat', 'Gv1a2002.mat', 'Gv1a1012.mat', 'Gv1a2012.mat', 'Gv1b1020.mat', 'Gv1b2020.mat', 
	'Gv1b1164.mat', 'Gv1b2164.mat', 'Gv1b2022.mat', 'Gv1c1020.mat', 'Gv1c2020.mat', 'Gv1c1030.mat', 'Gv1c2030.mat',
	'Gv1c1084.mat', 'Gv1c2022.mat', 'Gv1d1132.mat', 'Gv2a1078.mat', 'Gv2a1080.mat', 'Gv2b1008.mat', 'Gv2b2008.mat',
	'Gv2c1008.mat', 'Gv2c1010.mat', 'Gv2c2008.mat', 'Gv2c2010.mat', 'Gv3c1034.mat', 'Gv3c1036.mat', 'Gv4c1040.mat', 
	'Gv4c1042.mat', 'Gv4c1044.mat', 'Gv4c1096.mat', 'Gv4d1036.mat', 'Gv4d2036.mat', 'Gv5a1046.mat', 'Gv5a2046.mat',
	'Gv5a1048.mat', 'Gv5a2048.mat', 'Gv5a1108.mat', 'Gv5b2046.mat', 'Gv5c1046.mat', 'Gv5c2046.mat', 'Gv5c1102.mat',
	'Gv5d1120.mat', 'Gv6c1050.mat', 'Gv6c2050.mat', 'Gv6c1052.mat', 'Gv6d1014.mat', 'Gv6d2014.mat', 'Gv6d1016.mat',
	'Gv6d2016.mat', 'Gv6d1018.mat', 'Gv6d2018.mat', 'Gv6d1086.mat', 'Gv6d2086.mat', 'Gv7a1068.mat', 'Gv7a1130.mat',
	'Gv8a1056.mat', 'Gv8b1058.mat', 'Gv8c1062.mat', 'Gv8c2062.mat', 'Gv8c1118.mat', 'Gv8c1210.mat', 'Gv8c1212.mat',
	'Gv8d1108.mat', 'Gv9a1060.mat', 'Gv9c1070.mat', 'Gv9c2070.mat', 'Gv9c1126.mat', 'Gv9d1126.mat', 'Gv1a1136.mat', 
	'Gv1a2014.mat', 'Gv1b1166.mat', 'Gv1b2166.mat', 'Gv1c1086.mat', 'Gv1c1202.mat', 
	'Gv1d1134.mat', 'Gv2a1052.mat', 'Gv2a2052.mat', 'Gv2b1010.mat', 'Gv2b2010.mat', 'Gv2c1080.mat', 'Gv3c1090.mat',
	'Gv4c1098.mat', 'Gv4c1100.mat', 'Gv4d1038.mat', 'Gv4d2038.mat', 'Gv5a1148.mat', 'Gv5a2148.mat', 'Gv5b2048.mat',
	'Gv5c1048.mat', 'Gv5c2048.mat', 'Gv6c2052.mat', 'Gv6d1088.mat', 'Gv6d2088.mat', 'Gv7c1056.mat', 'Gv8a2056.mat',
	'Gv8b2058.mat', 'Gv8c1214.mat', 'Gv8c1216.mat', 'Gv8d1110.mat', 'Gv9a1122.mat', 'Gv9c1128.mat',
	'Gv1a1004.mat', 'Gv1a2134.mat', 'Gv1b2168.mat', 'Gv1b2170.mat', 'Gv1c1032.mat', 'Gv1c1204.mat', 
	'Gv1c2032.mat', 'Gv1d1136.mat', 'Gv2a1140.mat', 'Gv2a2140.mat', 'Gv2b1198.mat', 'Gv2b2198.mat', 'Gv2c1082.mat',
	'Gv3c1092.mat', 'Gv4c1206.mat', 'Gv4c1208.mat', 'Gv4d1040.mat', 'Gv4d2040.mat', 'Gv5a1150.mat', 'Gv5a2150.mat',
	'Gv5c1104.mat', 'Gv5d1122.mat', 'Gv6c1106.mat', 'Gv6d1090.mat', 'Gv6d2090.mat', 'Gv7c1058.mat', 'Gv8a2060.mat',
	'Gv8b2060.mat', 'Gv8c1218.mat', 'Gv8c1220.mat', 'Gv8d1112.mat', 'Gv9c1072.mat', 'Gv9c2072.mat', 'Gv9d1128.mat']

def gene_data(FILES_SET, sample_len, mode="fft"):
	overlap = sample_len // 2

	meta_info = load_meta()
	# print("meta_info: ", meta_info.keys())
	# x_dict = {}
	# for i in range(1,10):
	# 	x_dict[i] = []
	X = []
	Y = []
	file_sample_count = np.zeros(len(FILES_SET))
	file_labels = []
	for n, filename in enumerate(FILES_SET):
		label = int(filename[2])

		if filename not in meta_info:
			continue

		y = meta_info[filename]

		file_labels.append(label)
		if not exists(os.path.join(PATH_A, filename)):
			print("A File not exist: "+filename)
			continue 

		s_filename = filename[0:3]+'s'+filename[3:]
		if not exists(os.path.join(PATH_S, s_filename)):
			print("S File not exist: "+s_filename)
			continue

		data = scio.loadmat(os.path.join(PATH_A, filename))
		x_a = data['Output_data']
		x_a = x_a[:,:-20]  # Delete outliers

		data = scio.loadmat(os.path.join(PATH_S, s_filename))
		x_s = data['Output_data']
		x_s = x_s[:,:-20]  # Delete outliers
		# print x_s.shape
		if x_a.shape[1]!=x_s.shape[1]:
			print("shape not equal",x_a.shape,x_s.shape)

		if filename == "Gv3c1090.mat" or filename == "Gv1a1136.mat":
			print(f"{filename}: {x_a.shape} {x_a[0][:10]}")

		x = np.concatenate((x_a,x_s),axis=0)

		# Only take the middle 50% data
		# x = x[:, x.shape[1]//4:x.shape[1]*3//4]
		# print("before norm, x: ", np.mean(x, axis=1))
		# x = (x - np.mean(x, axis=1, keepdims=True) ) / np.std(x, axis=1, keepdims=True)

		i = 0
		while i+sample_len <= x.shape[1]:
			file_sample_count[n] += 1
			
			if mode == "fft":
				# fft_signal = np.abs(np.fft.fft(x[:,i:i+SAMPLE_LEN], axis=-1))[:, :SAMPLE_LEN//2]
				fft_signal = np.fft.fft(x[:,i:i+sample_len], axis=-1)[:, :sample_len//2]
				fft_signal = np.concatenate([fft_signal.real, fft_signal.imag], axis=0) 
				fft_signal = np.concatenate([[fft_signal[0]], [fft_signal[5]], [fft_signal[3]], [fft_signal[8]]], axis=0) # Keep the real and imag parts of sensor 0 and 3 
				X.append(fft_signal)
				Y.append(y)
			elif mode == "stft":
				f, t, Zxx = signal.stft(x[:,i:i+sample_len], 1024, nperseg=128, noverlap=64)
				# f, t, Zxx = signal.stft(x[:,i:i+sample_len], 1024, nperseg=511)

				stft_signal = np.abs(Zxx)
				# print("Zxx.shape: ", Zxx.shape)
				# exit()
				# print("f len: ", len(f))
				# print("t len: ", len(t))
				# stft_signal = np.concatenate([Zxx.real, Zxx.imag], axis=0)
				
				# stft_signal = np.transpose(stft_signal, (0, 2, 1))

				# Only take 1 axis from each sensor
				# stft_signal = np.concatenate([[stft_signal[0]], [stft_signal[5]], [stft_signal[3]], [stft_signal[8]]], axis=0) # Keep the real and imag parts of sensor 0 and 3
				# stft_signal = np.concatenate([[stft_signal[0]], [stft_signal[0]], [stft_signal[0]], [stft_signal[0]]], axis=0) # Keep the real and imag parts of sensor 0 and 3
				# stft_signal = np.concatenate([[Zxx[0].real], [Zxx[0].real], [Zxx[0].real], [Zxx[0].real]], axis=0) # Keep the real and imag parts of sensor 0 and 3
				stft_signal = np.concatenate([[np.abs(Zxx[0])], [np.abs(Zxx[3])]], axis=0) # Keep the real and imag parts of sensor 0 and 3

				# print("stft_signal: ", stft_signal.shape)
				# print("before norm, stft_signal: ", np.mean(stft_signal, axis=(1,2)))
				# stft_signal = (stft_signal - np.mean(stft_signal, axis=(1,2), keepdims=True)) / np.std(stft_signal, axis=(1,2), keepdims=True)
				# print("stft_signal shape: ", stft_signal.shape, ", after norm, stft_signal: ", np.mean(stft_signal, axis=(1,2)))
				# exit()
				# print("stft_signal: ", stft_signal.shape)
				# exit()
				X.append(stft_signal)
				Y.append(y)
			# item = x[:,i:i+SAMPLE_LEN]
			# x_dict[label].append(item)
			i = i + sample_len - overlap
	X = np.array(X)
	# if mode == "fft":
		# X = (X - np.min(X)) / (np.max(X) - np.min(X))
	return np.array(X), np.array(Y), file_sample_count, file_labels

def load_data(sample_len=1024, mode="fft"):
	test_files = []
	with open(TEST_FILE, "r") as f:
		for line in f.readlines():
			test_files.append(line.strip())
	
	train_files = []
	for fn in FILES:
		if fn not in test_files:
			train_files.append(fn)
	train_X, train_Y, train_sample_count, train_labels = gene_data(train_files, sample_len, mode=mode)
	test_X, test_Y, test_sample_count, test_labels = gene_data(test_files, sample_len, mode=mode)

	return train_X, train_Y, test_X, test_Y, train_sample_count, test_sample_count, train_labels, test_labels
	
def load_generated_data(data_path):
	Xs = []
	Ys = []
	for fn in glob.glob(os.path.join(data_path, "*.pt")):
		data = torch.load(fn)
		audio_data = data['data']['shake']['audio']
		seismic_data = data['data']['shake']['seismic']

		Xs.append(np.concatenate([audio_data, seismic_data], axis=0))

		label = data['label']['vehicle_type']
		Ys.append(label)
		
	return np.array(Xs), np.array(Ys)

		
