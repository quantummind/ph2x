import tempfile
from astropy.io.votable import parse_single_table

import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import re
import scipy.signal
from astropy.stats import LombScargle

#
# PART I
#

def gaussian(A, B, L, t):
	"""
	Gaussian function
	Args:
		A: normalization coefficient of Gaussian function
		B: spread of the Gaussian function
		L: interval for fourier transform
		t: independent variable
	Returns:
		Gaussian function at t
	"""
	return A*np.exp(-B*(t-L/2)**2)

def analyticGFFT(A, B, L, N, k):
	"""
	Discrete Fourier transform coefficients h_k for Gaussian function
	Args:
		A: normalization coefficient of Gaussian function
		B: spread of the Gaussian function
		L: interval for fourier transform
		N: number of intervals in fourier transform
		k: coefficient index
	Returns:
		h_k for Gaussian function
	"""
	return N*A*np.sqrt(np.pi)/np.sqrt(B*L**2) * np.exp(k*np.pi*(1j-k*np.pi/(B*L**2)))

def analyticCosine(A, C, L, m, phi, N, k):
	"""
	Discrete Fourier transform coefficients h_k for cosine function
	Args:
		A: amplitude of the cosine
		C: y-axis shift of the cosine
		L: interval for fourier transform
		m: sets frequency of cosine
		phi: phase shift of cosine
		N: number of intervals in fourier transform
		k: coefficient index
	Returns:
		h_k for Gaussian function
	"""
	return N*A/(2*N)*np.exp(-1j*phi) * (\
		np.exp(-1j*m)*(-np.exp(1j*m) + np.exp(2j*k*np.pi)) / (-1+np.exp(-1j*(m-2*k*np.pi)/N)) + \
		np.exp(2j*phi)*(-1+np.exp(1j*(m+2*k*np.pi))) / (-1+np.exp(1j*(m+2*k*np.pi)/N)) \
	)
		

def cosine(A, C, L, m, phi, t):
	"""
	Cosine function
	Args:
		A: amplitude of the cosine
		C: y-axis shift of the cosine
		L: interval for fourier transform
		m: sets frequency of cosine
		phi: phase shift of cosine
		t: independent variable
	Returns:
		Cosine function at t
	"""
	return C + A*np.cos(m*t/L + phi)

def calcError(a1, a2, i1, i2):
	"""
	Percentage error between magnitudes of 2 arrays
	Args:
		a1: numpy array
		a2: numpy array
		i1: start index to measure error
		i2: final index to measure error
	Returns:
		numpy array of percentage error
	"""
	a = np.absolute(a1[i1:i2]) - np.absolute(a2[i1:i2])
	b = np.absolute(a1[i1:i2])
	return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

def transform(showPlots=False):
	"""
	Does FFT on Gaussian function and cosine function
	Args:
		showPlots: show plots comparing analytical and numpy FFT coefficients
	"""
	L = 100
	N = 100
	tVals = np.linspace(0, L, num=N)
	kVals = np.linspace(0, tVals.size-1, num=tVals.size)
	
	gVals = gaussian(100, 1, L, tVals)
	gFFTA = analyticGFFT(100, 1, L, N, kVals)
	cVals = cosine(2, 1, L, 60, 1, tVals)
	cFFTA = analyticCosine(2, 1, L, 60, 1, N, kVals)
	
	gFFT = np.fft.fft(gVals)
	cFFT = np.fft.fft(cVals)
	
	if showPlots:
		plt.figure()
		plt.ylabel('FFT')
		plt.xlabel('k (index)')
		plt.title('Cosine')
		plt.plot(kVals, np.absolute(cFFT), kVals, np.absolute(cFFTA))
		
		plt.figure()
		plt.ylabel('FFT')
		plt.xlabel('k (index)')
		plt.title('Gaussian')
		plt.plot(kVals, np.absolute(gFFT), kVals, np.absolute(gFFTA))
		plt.show()
	
	print('Largest percentage error between numpy FFT and analytic FFT of Gaussian function:', \
		np.ndarray.max(calcError(gFFT, gFFTA, 0, gFFT.size//2)))
	print('Largest percentage error between numpy FFT and analytic FFT of cosine function:', \
		np.ndarray.max(calcError(cFFT, cFFTA, 1, cFFT.size)))		# trim out 0 point because of numpy spike at 0
	
	gIFFT = np.fft.ifft(gFFT)
	cIFFT = np.fft.ifft(cFFT)
	
	print('Largest percentage error between Gaussian function and inverse FFT of FFT of Gaussian function:', \
		np.ndarray.max(calcError(gVals, gIFFT, 0, gIFFT.size)))
	print('Largest percentage error between cosine function and inverse FFT of FFT of cosine function:', \
		np.ndarray.max(calcError(cVals, cIFFT, 0, cIFFT.size)))


#
# PART II
#


def transformData(filename, timestep, showPlots=False):
	"""
	Fourier transforms time series data
	Args:
		filename: file with time series data
		timestep: time between data points in time series (s)
		showPlots: show plot of signal vs t, plot of FFT vs freq
	Returns:
		tuple containing (time values, time series signal, frequencies after FFT, data with FFT applied)
	"""
	data = np.loadtxt(filename)
	t = np.linspace(timestep, data.size*timestep, num=data.size)
	freqs = np.fft.fftfreq(data.size, d=timestep)
	dataFFT = np.fft.fft(data)
	
	if showPlots:
		plt.figure()
		plt.ylabel('Signal')
		plt.xlabel('t (s)')
		plt.title('Raw Data')
		plt.plot(t, data)
	
		plt.figure()
		plt.ylabel('Magnitude of Signal FFT')
		plt.xlabel('Frequency (Hz)')
		plt.title('FFT')
		plt.plot(freqs, np.absolute(dataFFT))
		plt.show()
	
	return (t, data, freqs, dataFFT)

def findSignalFreq(transformedData):
	"""
	Finds frequency of signal using FFT
	Args:
		transformedData: tuple of (frequencies, data with FFT applied)
	Returns:
		signal frequency in Hz
	"""
	
	return transformedData[2][np.argmax(np.absolute(transformedData[3]))]

def findEnvelope(transformedData, showPlots=False):
	"""
	Finds delta t of Gaussian envelope by minimizing mean-squared-error of Fourier
		coefficients for Gaussian envelope and actual data
	Args:
		transformedData: tuple of (frequencies, data with FFT applied)
		showPlots: show FFT vs freq for different envelopes with different delta t values
	Returns:
		delta t (width of Gaussian envelope) in seconds
	"""
	deltaTs = np.linspace(1, 5, num=10)
	t = transformedData[0]
	timestep = t[1] - t[0]
	freqs = transformedData[2]
	fftMag = np.absolute(transformedData[3])
	offset = np.argmax(np.absolute(transformedData[3]))
	width = 20		# parameter for pretty graphs, defines number of points around peak in data
	gFFTs = []
	freqs = freqs[offset-width:offset+width]
	fftMag = fftMag[offset-width:offset+width] / np.amax(fftMag)
	lowestError = -1
	best_dt = -1
	for dt in deltaTs:
		gFFT = np.fft.fft(np.exp(-(t)**2/dt**2))
		gFFT = np.roll(gFFT, offset) / np.amax(gFFT)
		gFFT = np.absolute(gFFT[offset-width:offset+width])
		gFFTs.append(gFFT)
		mse = ((gFFT - fftMag)**2).mean()
		if lowestError == -1 or mse < lowestError:
			lowestError = mse
			best_dt = dt
	
	if showPlots:
		plt.figure()
		plt.ylabel('FFT')
		plt.xlabel('Frequency (Hz)')
		plt.title('FFT')
		
		plt.plot(freqs, fftMag)
		for i in range(len(gFFTs)):
			plt.plot(freqs, gFFTs[i], label=str(deltaTs[i]) + ' s')
		plt.legend()
		plt.show()
	
	return best_dt

#
# PART III
#

def fftLS(t, data, maxFreq, showFreq=-1, showPlots=False):
	"""
	Fourier transforms time series data with Lomb-Scargle
	Args:
		t: time data
		data: signal corresponding to time data
		maxFreq: highest frequency to check for FFT
		showFreq: draw a vertical line at a certain frequency
		showPlots: show plot of signal vs t, plot of FFT vs freq
	Returns:
		tuple containing (time values, time series signal, frequencies after FFT, data with FFT applied)
	"""
	freqs = np.linspace(0.0001, maxFreq, num=data.size)
	dataFFT = LombScargle(t, data).power(freqs)
	
	if showPlots:
		plt.figure()
		plt.ylabel('Signal')
		plt.xlabel('t')
		plt.title('Raw Data')
		plt.scatter(t, data)
	
		plt.figure()
		plt.ylabel('Magnitude of Signal FFT')
		plt.xlabel('Frequency (units 1/[t])')
		plt.title('FFT')
		plt.plot(freqs, np.absolute(dataFFT))
		if showFreq != -1:
			plt.axvline(x=showFreq, color='r', linestyle='--')
		plt.show()
	
	return (t, data, freqs, dataFFT)

def transformDataLS(filename, timestep, maxFreq, showPlots=False):
	"""
	Fourier transforms time series data with Lomb-Scargle
	Args:
		filename: file with time series data
		timestep: time between data points in time series
		maxFreq: highest frequency to check for FFT
		showPlots: show plot of signal vs t, plot of FFT vs freq
	Returns:
		tuple containing (time values, time series signal, frequencies after FFT, data with FFT applied)
	"""
	data = np.loadtxt(filename)
	t = np.linspace(timestep, data.size*timestep, num=data.size)
	return fftLS(t, data, maxFreq, showPlots=showPlots)

def extractVotData(vot):
	"""
	Gets data from vottable and puts into numpy array
	Args:
		vot: raw data from vottable
	Returns:
		tuple of 2 numpy arrays (time, time series data)
	"""
	data = []
	with tempfile.TemporaryFile() as tmp:
		tmp.write(vot)
		votable = parse_single_table(tmp).array
		times = [item[0] for item in votable['ObsTime']]
		mags = [item[0] for item in votable['Mag']]
		mags = [tVal for _,tVal in sorted(zip(times, mags))]
		times = sorted(times)
		data = (np.array(times), np.array(mags))
        # TA comment: Recommend leaving the data in a numpy array for ease of further manipulation
	return data

def analyzeHerX1():
	"""
	Identifies orbital period of Her X-1 binary star system
	"""

	vot = urllib.request.urlopen('http://nesssi.cacr.caltech.edu/DataRelease/upload/result_web_fileHTlmTa.vot')
	votSource = vot.read()
	rawData = extractVotData(votSource)

	transformedData = fftLS(rawData[0], rawData[1], 5, showFreq=1/1.7)
	freq = findSignalFreq(transformedData)
	print('Her X-1 orbital period (days):', 1/freq)

if __name__ == '__main__':
	print('README')
	print('To see any of the plots, call showPlots=True to a given function.')
	print()
	
	print('PART I')
	transform()
	print()
	
	print('PART II')
	transformedData = transformData('arecibo1.txt', 0.001)
	signalFreq = findSignalFreq(transformedData)
	dt = findEnvelope(transformedData)
	print('Signal frequency:', signalFreq, 'Hz')
	print('Envelope width:', dt, 's')
	print()
	
	print('PART III')
	transformedDataLS = transformDataLS('arecibo1.txt', 0.001, 500)
	signalFreq = findSignalFreq(transformedDataLS)
	dt = findEnvelope(transformedDataLS)
	print('Signal frequency (Lomb-Scargle):', signalFreq, 'Hz')
	print('Envelope width (Lomb-Scargle):', dt, 's')
	analyzeHerX1()
	# Note: the analyzeHerX1 fails to produce a period of 1.70 days, because the resolution of the data is too coarse