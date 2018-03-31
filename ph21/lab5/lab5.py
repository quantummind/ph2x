import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# for comparison
from sklearn.decomposition import PCA as spca
from sklearn.preprocessing import StandardScaler

def pca(data, num_dimensions):
	"""
	Performs principal component analysis
	Args:
		data: m x n 2d numpy array, where m is the number of dimensions and
			n is the number of samples
		num_dimensions: number of principal components to keep when transforming newdata
	Returns:
		tuple of rescaled data using top num_dimensions principal components,
			all eigenvalues, and all eigenvectors
	"""
	mean = np.average(data, axis=1)
	stdev = np.std(data, axis=1)
	for i in range(len(data)):
		data[i] -= mean[i]
		data[i] /= stdev[i]
	cov = np.cov(data)
	eigvals, eigvecs = np.linalg.eigh(cov)
	
	idx = np.argsort(eigvals)[::-1]		# sort eigenvalues descending
	eigvals = eigvals[idx]
	eigvecs = eigvecs[:, idx]
	print('Percent of variance accounted for', sum(eigvals[:num_dimensions])/sum(eigvals))
	newData = np.transpose(eigvecs[:, :num_dimensions]).dot(data)
	
	return (newData, eigvals, eigvecs)


def linearData():
	"""
	Generates randomized 2d data with linear dependence
	Returns:
		numpy array of [[x1, x2, x3, ...], [y1, y2, y3, ...]]
	"""
	length = 50
	x = np.linspace(1, 10, num=length) + np.random.rand(length)
	y = x + np.random.rand(length)
	return np.array([x, y])

def plotPCA2d(x, y, title):
	"""
	Plots PCA scatter plots
	Args:
		x: x series
		y: y series
		title: title to display
	"""
	plt.figure()
	plt.scatter(x, y)
	plt.xlabel('Principal Component 1')
	plt.ylabel('Principal Component 2')
	plt.title(title)


def ballData():
	"""
	Generates data recorded by 3 moving Galilean observers for single ball moving on x axis
	"""
	observers = [-5, 1, 0, 2]
	times = np.linspace(0, 10)
	data = np.empty((len(observers), times.size))
	for i in range(times.size):
		for j in range(len(observers)):
			data[j][i] = np.sin(times[i]) - observers[j]*times[i]
	return data

if __name__ == '__main__':
	#
	# LINEAR DATA
	#
	print('Linear Data:')
	d = linearData()
	print('Eigenvalues', pca(d, 2)[1])
	print('Principal components\n', pca(d, 2)[2])
	print()
	
	#
	# GALILEAN OBSERVER DATA
	#
	print('Ball Data:')
	d = ballData()
	print('Eigenvalues', pca(d, 2)[1])
	print('Principal components\n', pca(d, 2)[2])
	print()
	
	#
	# IRIS DATA
	#
	
	print('Iris Data:')
	iris = np.transpose(datasets.load_iris().data)
	irisPca = pca(iris, 2)		# reduce dimensionality of iris data from 4 features to 2 features
	print('Principal components of iris data\n', irisPca[2])
	plotPCA2d(irisPca[0][0], irisPca[0][1], 'Iris dataset - custom implementation')
	
	# compare to sklearn PCA
	sk = np.transpose(spca(n_components=2).fit_transform(StandardScaler().fit_transform(np.transpose(iris))))
	plotPCA2d(sk[0], sk[1], 'Iris dataset - SKLearn')
	
	print("""Note that the SKLearn PCA is vertically flipped compared to the custom implenetation, 
which implies that Principal Component 1 has opposite sign. This does not affect the 
purpose of PCA, and thus the two graphs are effectively equivalent.""")
	plt.show()