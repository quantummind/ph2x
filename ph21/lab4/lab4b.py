import math
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pymc3 as pm


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

def testCoin(n, H):
	"""
	Flips coin n times and gives number of times it came up heads
	Args:
		n: number of trials to execute
		H: probability of flip showing heads
	Returns:
		number of times heads came up in n trials
	"""
	headCount = 0
	for i in range(n):
		headCount += tossCoin(H)
	return headCount

def priorComparison(setH):
	"""
	Plots 3 figures of probability density vs. H given constant H=setH. Each figure
	corresponds to a different prior distribution.
	Args:
		setH: true H bias of coin
	"""
	nVals = np.geomspace(1, 2**10, num = 11)
	niter = 1000
	
	flipData = np.empty(nVals.size)
	for i in range(nVals.size):
		n = int(nVals[i])
		flipData[i] = testCoin(n, setH)
	
	priors = ['Uniform Prior', 'Gaussian, stdev=0.1, mean=0.2', 'Gaussian, stdev=0.1, mean=0.6']
	for n in range(len(priors)):
		plt.figure()
		plt.ylabel('Probability density')
		plt.xlabel('H (probability of heads)')
		plt.title(priors[n] + ', H=' + str(setH))
		for i in range(nVals.size):
			with pm.Model() as model:
				prior = None
				if n == 0:
					prior = pm.Uniform('prior', lower=0, upper=1)
				elif n == 1:
					prior = pm.Normal('prior', mu=0.2, sd=0.1)
				elif n == 2:
					prior = pm.Normal('prior', mu=0.6, sd=0.1)
				likelihood = pm.Binomial('likelihood', n=int(nVals[i]), p=prior, observed=flipData[i])
				start = pm.find_MAP()
				step = pm.Metropolis()
				trace = pm.sample(niter, step, start, random_seed=123)
				plt.hist(trace['prior'], max(int(nVals[i])//50, 1), histtype='step', density=True, color=str(((nVals.size - i - 1)/nVals.size)**0.5))

def hComparison(setN):
	"""
	Plots figure of probability density vs. H given constant n=setN
	Args:
		setN: number of total flips per test value of H
	"""
	trueH = np.linspace(0.2, 0.8, num=4)
	nchains = np.geomspace(2, 32, num=5)
	
	plt.figure()
	plt.ylabel('Probability density')
	plt.xlabel('H (probability of heads)')
	plt.title('Changing true H for product=2^15 (darker means more short chains)')
	
	product = 2**15
	for i in range(trueH.size):
		for j in range(len(nchains)):
			with pm.Model() as model:
				numHeads = testCoin(setN, trueH[i])
				prior = pm.Uniform('prior', lower=0, upper=1)
				likelihood = pm.Binomial('likelihood', n=setN, p=prior, observed=numHeads)
				start = pm.find_MAP()
				step = pm.Metropolis()
				trace = pm.sample(product//nchains[j], step, start, chains=int(nchains[j]), random_seed=123)
				pct = (nchains.size - j - 1)/nchains.size
				plt.hist(trace['prior'], setN//50, histtype='step', density=True, color=str(pct))


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

def simulateLighthouse(alpha, beta, n):
	"""
	Finds alpha and beta by simulating lighthouse flashes and recording them on shore
	Args:
		alpha: true alpha
		beta: true beta
		n: number of flashes recorded
	"""
	xVals = np.linspace(2*(alpha - 4*beta), 2*(alpha + 4*beta), num=1000)
	xProbs = probFlash(xVals, alpha, beta)
	xProbs /= sum(xProbs)
	flashes = np.random.choice(xVals, n, p=xProbs)
	
	title = 'Lighthouse: determing alpha and beta for alpha=' + str(alpha) + ' and beta=' + str(beta)
	inferAlphaBeta(alpha, beta, flashes, title)

def simulateLighthouseInterloper(alphaLighthouse, betaLighthouse, alphaInterloper, betaInterloper, n):
	"""
	Finds alpha and beta by simulating lighthouse and interloper flashes and recording them on shore
	Args:
		alpha: true alpha
		beta: true beta
		n: number of flashes recorded
	"""
	xVals = np.linspace(2*(alphaLighthouse - 4*betaLighthouse), 2*(alphaLighthouse + 4*betaLighthouse), num=1000)
	lighthouseProbs = probFlash(xVals, alphaLighthouse, betaLighthouse)
	interloperProbs = probFlash(xVals, alphaInterloper, betaInterloper)
	xProbs = lighthouseProbs + interloperProbs
	xProbs /= sum(xProbs)
	flashes = np.random.choice(xVals, n, p=xProbs)
	
	title = 'Lighthouse w/ interloper: determing alpha and beta for alpha=' + str(alphaLighthouse) + ' and beta=' + str(betaLighthouse)
	inferAlphaBeta(alphaLighthouse, betaLighthouse, flashes, title)

def inferAlphaBeta(alpha, beta, flashes, title):
	"""
	Finds alpha and beta by simulating lighthouse flashes and recording them on shore
	Args:
		flashes: positions on shore where flashes were recorded
	"""
	
	product = 2**15
	nchains = np.array([16])
# 	nchains = np.linspace(2, 32, num=2)
	for i in range(nchains.size):
		with pm.Model() as model:
			alphaPrior = pm.Uniform('alpha', lower=-4*alpha, upper=4*alpha)
			betaPrior = pm.Uniform('beta', lower=0, upper=4*beta)
			likelihood = pm.Cauchy('likelihood', alpha=alphaPrior, beta=betaPrior, observed=flashes)
			start = pm.find_MAP()
			step = pm.NUTS()
			trace = pm.sample(product//nchains[i], step, start, chains=int(nchains[i]), random_seed=123)
			plt.figure()
			plt.xlabel('Alpha')
			plt.ylabel('Beta')
			plt.title(title)
			sn.kdeplot(trace['alpha'], trace['beta'])

if __name__ == '__main__':
	# PART I
	hComparison(1000)
	priorComparison(0.65)
	
	# PART II
	simulateLighthouse(1, 1, 2**8)
	simulateLighthouseInterloper(1, 1, -2, 0.5, 2**8)

	plt.show()