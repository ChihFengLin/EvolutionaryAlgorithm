

import sys
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import scipy.signal as signal
import pywt
import pyeeg

def return_filtered_epoch(epoch_data):
    
    xs = epoch_data[:fft_size]
    hann = signal.hann(fft_size, sym=0)
    xs = np.array(xs, np.float)
    hann = np.array(hann, np.float)

    xs_hann = xs * hann
    xf = np.fft.rfft(xs)/fft_size
    power_xf = np.abs(xf)
    xfp = 20*np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    freqs = np.linspace(0, sampling_rate/2, fft_size/2 + 1)

    xout = filter60hz(xs_hann)
    xout = xout[50:fft_size+50]
    bpf = signal.remez(2000, (0.0, 30.0, 30.1, 64.0), (1.0, 0.01), Hz=sampling_rate, type='bandpass')
    w, h = signal.freqz(bpf, sampling_rate)

    xout = signal.lfilter(bpf, 1, xout)
    xf_filtered = np.fft.rfft(xout)/fft_size
    power_xf_filtered = np.abs(xf_filtered)
    xfp_filtered = 20*np.log10(np.clip(np.abs(xf_filtered), 1e-20, 1e100))

    return power_xf, power_xf_filtered, freqs, xfp


# Design Filter - band pass filter (0.5 ~ 30 Hz)
def filter60hz(A):
    
    filter=[0.0056, 0.0190, 0.0113, -0.0106, 0.0029, 0.0041,
            -0.0082, 0.0089, -0.0062, 0.0006, 0.0066, -0.0129,
            0.0157, -0.0127, 0.0035, 0.0102, -0.0244, 0.0336,
            -0.0323, 0.0168, 0.0136, -0.0555, 0.1020, -0.1446,
            0.1743, 0.8150, 0.1743, -0.1446, 0.1020, -0.0555,
            0.0136, 0.0168, -0.0323, 0.0336, -0.0244, 0.0102,
            0.0035, -0.0127, 0.0157, -0.0129, 0.0066, 0.0006,
            -0.0062, 0.0089, -0.0082, 0.0041, 0.0029, -0.0106,
            0.0113, 0.0190, 0.0056]
        #convolution
    P = len(A)
    Q = len(filter)
    N = P + Q - 1
    c = []
    for k in range(N):
        t = 0
        lower = max(0, k-(Q-1))
        upper = min(P-1, k)
        for i in range(lower, upper+1):
            t = t + A[i] * filter[k-i]
        c.append(t)
    return c

def return_dominant_freq(freq_list, filtered_power):
	# Return Dominant Frequency
	max_power = np.amax(filtered_power)
	point = np.where(filtered_power==max_power)
	dominant_f = freq_list[point]
	#print dominant_f[0]
	return dominant_f


def return_freq_point_low(lower_f, freqs):
	for i in range(len(freqs)):
		if freqs[i] >= lower_f:
			return i

def return_freq_point_high(higher_f, freqs):
	for i in range(len(freqs)):
		if freqs[i] >= higher_f:
			return i-1

def return_power_ratio(freq_list, filtered_power):
	
	power_sum = np.sum(filtered_power)
	
	# Calculate 0.5~3 Hz power ratio
	low_bound = return_freq_point_low(0.5, freq_list)
	high_bound = return_freq_point_high(3, freq_list)
	delta_band_power = np.sum(filtered_power[low_bound:high_bound])
	delta_ratio = delta_band_power/ power_sum

	# Calculate 3~8 Hz power ratio
	low_bound = return_freq_point_low(3, freq_list)
	high_bound = return_freq_point_high(8, freq_list)
	theta_band_power = np.sum(filtered_power[low_bound:high_bound])
	theta_ratio = theta_band_power/ power_sum

	# Calculate 8~12 Hz power ratio
	low_bound = return_freq_point_low(8, freq_list)
	high_bound = return_freq_point_high(12, freq_list)
	alpha_band_power = np.sum(filtered_power[low_bound:high_bound])
	alpha_ratio = alpha_band_power/ power_sum

	# Calculate 12~16 Hz power ratio
	low_bound = return_freq_point_low(12, freq_list)
	high_bound = return_freq_point_high(16, freq_list)
	sigma_band_power = np.sum(filtered_power[low_bound:high_bound])
	sigma_ratio = sigma_band_power/ power_sum

	# Calculate 16~30 Hz power ratio
	low_bound = return_freq_point_low(16, freq_list)
	high_bound = return_freq_point_high(30, freq_list)
	beta_band_power = np.sum(filtered_power[low_bound:high_bound])
	beta_ratio = beta_band_power/ power_sum

	return delta_ratio, theta_ratio, alpha_ratio, sigma_ratio, beta_ratio


def calculate_mean_power(series):
	series_length = len(series)
	result = 0
	for i in series:
		result += (np.abs(i))**2
	return result/float(series_length)

def return_DWT_feature(epoch_data):
	# Discrete Wavelet Transform Coeifficients
	(cA5, cD5, cD4, cD3, cD2, cD1) = pywt.wavedec(epoch_data, 'db4', level=5)
	
	A5_mean = np.mean(cA5)
	D5_mean = np.mean(cD5)
	D4_mean = np.mean(cD4)
	D3_mean = np.mean(cD3)
	A5_std = np.std(cA5)
	D5_std = np.std(cD5)
	D4_std = np.std(cD4)
	D3_std = np.std(cD3)
	A5_pm = calculate_mean_power(cA5)
	D5_pm = calculate_mean_power(cD5)
	D4_pm = calculate_mean_power(cD4)
	D3_pm = calculate_mean_power(cD3)
	A5_ratio_mean = np.mean(np.abs(cA5)) / float(np.mean(np.abs(cD5)))
	D5_ratio_mean = np.mean(np.abs(cD5)) / float(np.mean(np.abs(cD4)))
	D4_ratio_mean = np.mean(np.abs(cD4)) / float(np.mean(np.abs(cD3)))
	D3_ratio_mean = np.mean(np.abs(cD3)) / float(np.mean(np.abs(cD2)))
	
	return A5_mean, D5_mean, D4_mean, D3_mean, A5_std, D5_std, D4_std, D3_std, A5_pm, D5_pm, D4_pm, D3_pm, \
			A5_ratio_mean, D5_ratio_mean, D4_ratio_mean, D3_ratio_mean

# Statistical Parameters
def getfmax(a):
    return max(a)

def getfmin(a):
    return min(a)

def getfmean(a):
    af=np.array(a, np.float)
    return np.mean(af)
    
def getfstd(a):
    af=np.array(a, np.float)
    return np.std(af)

def getfvar(a):
    af=np.array(a, np.float)
    return np.var(af)

def getfskew(a):
    af=np.array(a, np.float)
    return np.var(af)*3/2/(np.std(af)**3)
    
def getfkur(a):
    af=np.array(a, np.float)
    return np.var(af)*2/(np.std(af)**4)
    
def getfmd(a):
    af=np.array(a, np.float)
    return np.median(af)
    
def getzcnum(a):
    af=np.array(a, np.float)
    sign= np.sign(af)
    sign[sign==0] = -1 #replace zeros with -1
    zc = np.where(np.diff(sign))[0]
    return len(zc)




# Main


fd = open('/Users/chih-fenglin/Dropbox/CMU Course/Mobile Hardware for Software Engineers/Final Project/lin_day3/DATA7.txt', 'r')

curr_position = fd.tell()
line = fd.readline()


while True:

	fd.seek(curr_position)

	sampling_rate = 128
	fft_size = sampling_rate * 30
	t = np.arange(0, 30.0, 1.0/sampling_rate)
	count = 0
	time_series = []
	vibration_list = []
	flag = True
	flag1 = False
	stage_flag = False
	test_count = 0
	#for line in fd.readlines():
	while line:
#		print test_count
#		test_count += 1
		if count == fft_size:
			curr_position = fd.tell()
			break
		line = line.strip()
		line = line.split(',')
		#sleep_stage = "Unknown"

		if (line[0].find('Unix Timestamp') != -1):
			line = fd.readline()
			if flag1:
				line = line.strip()
				line = line.split(',')
#				flag = True
#				if (line[0].find('Vibration') != -1):
				line = line[0].split(':')
#				if flag:
				vibration_list.append(line[1])
#				flag = False
				line = fd.readline()
				flag1 = True
				continue
			continue
		if (line[0].find('Signal Quality') != -1):
			line = fd.readline()
			if flag1:
				line = line.strip()
				line = line.split(',')
#				flag = True
#				if (line[0].find('Vibration') != -1):
				line = line[0].split(':')
#				if flag:
				vibration_list.append(line[1])
				flag = False
				flag1 = False
				line = fd.readline()
				continue
			continue
		if (line[0].find('error') != -1):
			line = fd.readline()
			continue
		if (line[0].find('Vibration') != -1):
			line = line[0].split(':')
			if flag:
				vibration_list.append(line[1])
			flag = False
			line = fd.readline()
			continue
		if (line[0].find('Awake') != -1 or line[0].find('REM') != -1 \
				or line[0].find('Light') != -1 or line[0].find('Deep') != -1 \
				or line[0].find('Undefined') != -1):
			#sleep_stage = "Unknown"
			if stage_flag:
				sleep_stage = line[0]
			line = fd.readline()
			continue

		time_series.append(float(line[0]))
		count += 1
		flag = True
		flag1 = True
		stage_flag = True
		line = fd.readline()

	if not line:
		sys.exit('STOP')
#	print vibration_list
	print len(vibration_list)
	if len(vibration_list) > 90:
		vibration_list = vibration_list[:90]
	while len(vibration_list) != 90:
		vibration_list.append(0)
	#print vibration_list
	#vibration_array = np.zeros(fft_size)
	#for i in range(len(vibration_list)):
	#	index = (vibration_list[i]-1) * sampling_rate
	#	vibration_array[index:index+sampling_rate] = 1

	(power_xf, power_xf_filtered, freqs, xfp) = return_filtered_epoch(time_series)

	dominant_f = return_dominant_freq(freqs, power_xf_filtered)
	(delta_ratio, theta_ratio, alpha_ratio, sigma_ratio, beta_ratio) = return_power_ratio(freqs, power_xf_filtered)
	(A5_mean, D5_mean, D4_mean, D3_mean, A5_std, D5_std, D4_std, D3_std, A5_pm, D5_pm, D4_pm, D3_pm, \
			A5_ratio_mean, D5_ratio_mean, D4_ratio_mean, D3_ratio_mean) = return_DWT_feature(time_series)

	hurst_index = pyeeg.hurst(time_series)
	pfd_index = pyeeg.pfd(time_series)
	sp_entropy = pyeeg.spectral_entropy(time_series, [0.5, 3, 8, 12, 16, 30], sampling_rate, Power_Ratio = None)
	hj_activity, hj_mobility, hj_complexity = pyeeg.hjorth(time_series)

	fmax=getfmax(time_series)
	fmin=getfmin(time_series)
	fmean=getfmean(time_series)
	fstd=getfstd(time_series)
	fvar=getfvar(time_series)
	fskew=getfskew(time_series)
	fkur=getfkur(time_series)
	fmd=getfmd(time_series)
	zcnum=getzcnum(time_series)

	print [fmax, fmin, fmean, fstd, fvar, fskew, fkur, fmd, zcnum, dominant_f[0], delta_ratio, \
			theta_ratio, alpha_ratio, sigma_ratio, beta_ratio, A5_mean, D5_mean, D4_mean, D3_mean, \
			A5_std, D5_std, D4_std, D3_std, A5_pm, D5_pm, D4_pm, D3_pm, A5_ratio_mean, D5_ratio_mean, \
			D4_ratio_mean, D3_ratio_mean, hurst_index, pfd_index, sp_entropy, hj_activity, hj_mobility, \
			hj_complexity]
	#print sleep_stage

	time = [i for i in range(90)]


	#fd.close()
	pl.figure(figsize=(8, 4))
	pl.subplot(411)
	pl.title("EEG Data")
	pl.xlabel("Time(s)")
	pl.plot(t[:fft_size], time_series)
	pl.subplot(412)
	pl.title("Vibration Signal")
	pl.xlabel("Time(s)")
	pl.plot(time, vibration_list)
	#pl.ylim(20, -20)
	pl.subplots_adjust(hspace = 0.8)
	pl.subplot(413)
	pl.title("EEG Data in Frequency Domain - w/o filter")
	pl.xlabel("Frequency(Hz)")
	pl.ylabel("dB")
	pl.plot(freqs, xfp)  #xfp
	pl.subplots_adjust(hspace = 0.8)
	pl.subplot(414)
	pl.title("EEG Data in Frequency Domain - w/ filter")
	pl.xlabel("Frequency(Hz)")
	pl.ylabel("Power")
	pl.plot(freqs, power_xf_filtered)
	pl.subplots_adjust(hspace = 0.8)
	pl.show()

