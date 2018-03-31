import math
import random
import numpy as np
import matplotlib.pyplot as plt


#
# PART I
#

def tossCoin(H):
	"""
	Flips a coin
	Args:
		H: probability of coin landing on heads
	Returns:
		1 of heads, 0 if tails
	"""
	return int(random.uniform(0, 1) < H)

def testBinary(n, H):
	"""
	Does binary problem n times and gives number of times it came up 1
	Args:
		n: number of trials to execute
		H: probability of event resulting in 1
	Returns:
		number of times event came up 1 after n trials
	"""
	headCount = 0
	for i in range(n):
		headCount += tossCoin(H)
	return headCount

def probD(H, n, h):
	"""
	Uses binomial distribution to find probability of data given H and I
	Args:
		H: probability of coin landing on heads
		n: number of total flips
		h: number of flips that came up heads
	Returns:
		Non-normalized prob(D|H,I)
	"""
	return math.factorial(n) // math.factorial(n-h) / math.factorial(h) * H**h * (1-H)**(n-h)

def uniform(H):
	"""
	Uniform prior
	Args:
		H: probability of coin landing on heads
	Returns:
		1 (a constant)
	"""
	return 1

def gaussian(H, center):
	"""
	Gaussian prior
	Args:
		H: probability of coin landing on heads
		center: center coord of distribution
	Returns:
		Gaussian distribution centered around center with standard deviation 0.1
	"""
	stdev = 0.1
	return 1 / math.sqrt(2*math.pi*stdev**2) * math.exp(-(H-center)**2 / (2*stdev**2))

def gaussian20(H):
	"""
	Gaussian prior centered around 20%
	Args:
		H: probability of coin landing on heads
		center: 
	Returns:
		Gaussian distribution centered around 20% with standard deviation 0.1
	"""
	return gaussian(H, 0.2)

def gaussian60(H):
	"""
	Gaussian prior centered around 60%
	Args:
		H: probability of coin landing on heads
		center: 
	Returns:
		Gaussian distribution centered around 60% with standard deviation 0.1
	"""
	return gaussian(H, 0.6)


def probH(H, n, h, prior):
	"""
	Non-normalized Bayesian probability prob(H|D, I)
	Args:
		H: probability of coin landing on heads
		n: number of total flips
		h: number of flips that came up heads
		prior: function of H that gives prior probability
	Returns:
		Non-normalized prob(H|D,I)
	"""
	return probD(H, n, h) * prior(H)

def evaluatePrior(hVals, nVals, flipData, prior):
	"""
	Updates Bayesian probability distribution of coin bias using given prior
	Args:
		hVals: h values to iterate over to form distribution
		nVals: different number of flips per distribution
		flipData: number of heads flipped per value of n
		prior: prior function of H
	Returns:
		2d numpy array of Bayesian distributions, where each distribution has a
			different n value and varies over h
	"""
	dh = hVals[1] - hVals[0]
	dists = np.empty([nVals.size, hVals.size])
	for i in range(nVals.size):
		n = int(nVals[i])
		hResult = flipData[i]
		for j in range(hVals.size):
			dists[i][j] = probH(hVals[j], n, hResult, prior)
		dists[i] /= np.trapz(dists[i], dx=dh)
	return dists

def priorComparison(setH):
	"""
	Plots 3 figures of prob(H|D,I) vs. H given constant H=setH. Each figure
	corresponds to a different prior distribution.
	Args:
		setH: true H bias of coin
	"""
	nVals = np.geomspace(1, 2**10, num = 11)
	hVals = np.linspace(0, 1, num=1001)
	
	
	flipData = np.empty(nVals.size)
	for i in range(nVals.size):
		n = int(nVals[i])
		flipData[i] = testBinary(n, setH)
	
	priors = {'Uniform Prior' : uniform, 'Gaussian, stdev=0.1, mean=0.2' : gaussian20, 'Gaussian, stdev=0.1, mean=0.6' : gaussian60}
	for name, fnc in priors.items():
		plt.figure()
		plt.ylabel('prob(H|D,I)')
		plt.xlabel('H (probability of heads)')
		plt.title(name + ', H=' + str(setH))
		dists = evaluatePrior(hVals, nVals, flipData, fnc)
		for i in range(len(dists)):
			plt.plot(hVals, dists[i], color=str(((len(dists) - i - 1)/len(dists))**0.5))

def hComparison(setN):
	"""
	Plots figure of prob(H|D,I) vs. H given constant n=setN
	Args:
		setN: number of total flips per test value of H
	"""
	trueH = np.linspace(0.1, 0.9, num=9)
	hVals = np.linspace(0, 1, num=1001)
	dh = hVals[1] - hVals[0]
	
	n = setN
	plt.figure()
	plt.ylabel('prob(H|D,I)')
	plt.xlabel('H (probability of heads)')
	plt.title('Changing true H for n=' + str(setN))
	for i in range(trueH.size):
		hResult = testBinary(n, trueH[i])
		dist = np.zeros(hVals.size)
		for j in range(hVals.size):
			dist[j] = probH(hVals[j], n, hResult, uniform)
		dist /= np.trapz(dist, dx=dh)
		plt.plot(hVals, dist, color=str((trueH.size - i - 1)/trueH.size))


#
# PART II
#

def probFlash(x, alpha, beta):
	"""
	Gives the probability of a flash at position x given alpha and beta
	(Cauchy/Lorentzian distribution)
	Args:
		x: position on beach where flash could be detected
		alpha: x-coord of lighthouse
		beta: y-coord of lighthous
	Returns:
		probability of a flash at position x
	"""
	return beta / (math.pi * (beta**2 + (x-alpha)**2))


def inferAlpha(alpha, beta):
	"""
	Finds alpha given fixed beta by simulating lighthouse flashes and recording
	them on shore. The plots on the figure show different values of n.
	Args:
		alpha: true alpha
		beta: true beta
	"""
	alphaGrid = np.linspace(-4*alpha, 4*alpha, num=1000)
	xVals = np.linspace(2*(alpha - 4*beta), 2*(alpha + 4*beta), num=1000)
	xProbs = probFlash(xVals, alpha, beta)
	xProbs /= sum(xProbs)
	nVals = np.geomspace(1, 2**8, num=9)
	
	plt.figure()
	plt.ylabel('prob(alpha|beta,flashes)')
	plt.xlabel('alpha')
	plt.title('Changing n for alpha=' + str(alpha))
	for i in range(nVals.size):
		n = int(nVals[i])
		flashes = np.random.choice(xVals, n, p=xProbs)
		alphaDist = np.empty(alphaGrid.size)
		for j in range(alphaGrid.size):
			probA = math.log(1 / alphaGrid.size)		# uniform prior
			for flash in flashes:
				probA += math.log(probFlash(flash, alphaGrid[j], beta))
			alphaDist[j] = probA
		alphaDist -= np.average(alphaDist)
		alphaDist = np.exp(alphaDist)
		alphaDist /= sum(alphaDist)
		print('Average flash position:', np.average(flashes), '; Most likely alpha:', alphaGrid[np.argmax(alphaDist)])
		plt.plot(alphaGrid, alphaDist, color=str((nVals.size - i - 1)/nVals.size))
	
	# The average flash position does not indicate the most probable value of alpha
	# because averaging does not include a prior probability. Additionally, the Cauchy
	# distribution has fat tails, so the sample mean will be unreliable.

def inferAlphaBeta(alpha, beta, n):
	"""
	Finds alpha and beta by simulating lighthouse flashes and recording them on shore
	Args:
		alpha: true alpha
		beta: true beta
		n: number of flashes recorded
	"""
	alphaGrid = np.linspace(-4*alpha, 4*alpha, num=100)
	betaGrid = np.linspace(0.01, 2*beta, num=100)
	xVals = np.linspace(2*(alpha - 4*beta), 2*(alpha + 4*beta), num=1000)
	xProbs = probFlash(xVals, alpha, beta)
	xProbs /= sum(xProbs)
	flashes = np.random.choice(xVals, n, p=xProbs)
	
	plt.figure()
	plt.xlabel('prob(alpha|flashes)')
	plt.ylabel('prob(beta|flashes)')
	plt.title('Determing alpha and beta for alpha=' + str(alpha) + ' and beta=' + str(beta))
	
	dist = np.full((betaGrid.size, alphaGrid.size), math.log(1/(alphaGrid.size*betaGrid.size)))
	for i in range(betaGrid.size):
		for j in range(alphaGrid.size):
			for flash in flashes:
				dist[i][j] += math.log(probFlash(flash, alphaGrid[j], betaGrid[i]))
	dist -= np.average(dist)
	dist = np.exp(dist)
	plt.contour(alphaGrid, betaGrid, dist)

if __name__ == '__main__':
	# PART I
	hComparison(100)
	priorComparison(0.65)
	
	# PART II
	inferAlpha(1, 1)
	inferAlphaBeta(1, 1, 2**8)

	plt.show()