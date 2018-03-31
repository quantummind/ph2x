# Edge detection can prove to be highly reliant on the particular image and the
# purpose of the edge detection. Setting appropriate thresholds and other
# parameters are thus problem-specific. After computing derivatives, other
# procedures must be used to aid with thinning the edges and refining them.
# For the CAPTCHA problem in particular, edge detection would be effective at
# isolating text, yet the main difficulty lies in reading distorted text. However,
# the necessity of isolating the text makes edge detection useful for OCR. Other
# applications, such as in the field of self-driving vehicles, could include
# recognizing stop signs and demarcations on the road.


from PIL import Image
from PIL import ImageFilter
import numpy as np
import math


def preprocess(img, blurRadius):
	"""
	Blurs image to remove pixel noise
	Args:
		img: Image object that is unprocessed
		blurRadius: pixel radius to blur
	Returns:
		blurred Image object
	"""
	return img.filter(ImageFilter.GaussianBlur(radius=blurRadius))

def evaluateKernel(kernel, pixelData, dim, x, y):
	"""
	Evalutes kernel at given position on image
	Args:
		kernel: matrix to be used as weightings (must be square)
		pixelData: Image data obtained from im.load()
		dim: image size (tuple)
		x: x-coord to evaluate kernel at
		y: y-coord to evaluate kernel at
	Returns:
		kernel evaluated on image, normalized by the kernel size
	"""
	width = (len(kernel)-1) // 2
	val = 0
	for iInd in range(0, len(kernel)):
		i = x - width + iInd
		if i < 0:
			i = 0
		elif i >= dim[0]:
			i = dim[0] - 1
		
		for jInd in range(0, len(kernel)):
			j = y - width + jInd
			if j < 0:
				j = 0
			elif j >= dim[1]:
				j = dim[1] - 1
			val += kernel[iInd][jInd] * pixelData[i, j]
	
	return val / kernel.size

def generateSobelRow(startNum, length):
	"""
	Generates row of Sobel kernel
	Args:
		startNum: first number in row of kernel
		length: length of row
	Returns:
		array of kernel row
	"""
	row = [0] * length
	for i in range((length-1)//2):
		row[i] = startNum - i
		row[length-i - 1] = -(startNum - i)
	return row

def generateSobelKernel(size):
	"""
	Generates Sobel kernel of given size
	Args:
		size: side length of kernel
	Returns:
		numpy array of kernel
	"""
	kernel = [0] * size
	kernel[size//2] = generateSobelRow(size - 1, size)
	for i in range(size//2):
		row = generateSobelRow(size - i - 2, size)
		kernel[size//2 - i - 1] = row
		kernel[size//2 + i + 1] = row
	
	return np.array(kernel)

def getPixel(data, x, y):
	"""
	Utility method to get pixel at edge boundaries from array
	Args:
		data: 2d array
		x: x-coord
		y: y-coord
	Returns:
		data[x][y] or closest boundary value if out of bounds
	"""
	if x < 0:
		x = 0
	elif x >= len(data):
		x = len(data) - 1
	
	if y < 0:
		y = 0
	elif y >= len(data[0]):
		y = len(data[0]) - 1
	
	return data[x][y]

def thinEdges(g, dir):
	"""
	Thin edges with simple non-maximum suppression (divides into 3x3 kernel)
	Args:
		g: 2d list of image gradient magnitudes
		dir: direction of gradients (angle)
	Returns:
		maximum sections of gradients
	"""
	gThin = []
	for x in range(len(g)):
		gThin.append([0]*len(g[0]))
		for y in range(len(g[0])):
			roundedDir = dir[x][y] // (math.pi/8)
			if roundedDir == 3 or roundedDir == 4:
				if g[x][y] == max(getPixel(g, x-1, y), g[x][y], getPixel(g, x+1, y)):
					gThin[x][y] = g[x][y]
			elif roundedDir == 5 or roundedDir == 6:
				if g[x][y] == max(getPixel(g, x-1, y+1), g[x][y], getPixel(g, x+1, y-1)):
					gThin[x][y] = g[x][y]
			elif roundedDir == 0 or roundedDir == 7:
				if g[x][y] == max(getPixel(g, x, y-1), g[x][y], getPixel(g, x, y+1)):
					gThin[x][y] = g[x][y]
			elif roundedDir == 1 or roundedDir == 2:
				if g[x][y] == max(getPixel(g, x-1, y-1), g[x][y], getPixel(g, x+1, y+1)):
					gThin[x][y] = g[x][y]
	
	return gThin

def sobel(img, kernelSize):
	"""
	Evalutes Sobel edge detection on image, performs non-maximum suppression for edge thinning
	Args:
		img: Image to edge detect
	Returns:
		Image object that shows edges
	"""
	bw = img.convert('L')
	kernelX = generateSobelKernel(kernelSize)
	kernelY = np.transpose(kernelX)
	pixelData = bw.load()
	dim = img.size
	edges = Image.new('L', dim)
	edgeData = edges.load()
	g = []
	dir = []
	maxG = 0
	for x in range(dim[0]):
		g.append([0]*dim[1])
		dir.append([0]*dim[1])
		for y in range(dim[1]):
			gx = evaluateKernel(kernelX, pixelData, dim, x, y)
			gy = evaluateKernel(kernelY, pixelData, dim, x, y)
			g[x][y] = math.sqrt(gx**2 + gy**2)
			if (g[x][y] > maxG):
				maxG = g[x][y]
			dir[x][y] = math.atan2(gy, gx)
	
	thinImg = thinEdges(g, dir)
	
	for x in range(dim[0]):
		for y in range(dim[1]):
			edgeData[x, y] = int(thinImg[x][y] * 255/maxG)
	
	return edges

if __name__ == '__main__':
	filename = 'img2.jpg'
	im = Image.open(filename)
	
	# test a variety of blur kernel widths and Sobel kernel widths
	for i in range(3):
		blurSize = 2*i
		blurredIm = preprocess(im, blurSize)
		for j in range(3):
			sobelSize = 4*j+3
			edgeIm = sobel(blurredIm, sobelSize)
			edgeIm.save('gaussian' + str(blurSize) + 'px_' + 'sobel' + str(sobelSize) + 'px_' + filename)
			print('Image', i*3 + j + 1, 'out of 9 completed')