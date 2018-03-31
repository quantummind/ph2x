import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

def extendedTrapezoid(func, a, b, N):
	h_N = (float(b)-a)/(N-1)
	xVals = np.linspace(a, b, N)
	yVals = func(xVals)
	return h_N*(yVals[0]/2 + sum(yVals[1:-1]) + yVals[-1]/2)

def extendedSimpsons(func, a, b, N):
	h_N = (float(b)-a)/(N-1)
	xValsInteger = np.linspace(a, b, N)
	yValsInteger = func(xValsInteger)
	xValsHalf = np.linspace(a + h_N/2, b - h_N/2, N-1)
	yValsHalf = func(xValsHalf)
	return h_N*(yValsInteger[0]/6 + sum(yValsInteger[1:-1])/3 + 2*sum(yValsHalf)/3 + yValsInteger[-1]/6)

def eX(npArray):
	return np.exp(npArray)

def sinX(npArray):
	return np.sin(npArray)
	
def plotError():
	nValues = 2**np.arange(1, 30)
	correct = np.e - 1
	
	trapezoidErrors = []
	simpsonsErrors = []
	
	for n in np.nditer(nValues):
		trapezoidErrors.append(np.abs(extendedTrapezoid(eX, 0, 1, n) - correct))
		simpsonsErrors.append(np.abs(extendedSimpsons(eX, 0, 1, n) - correct))
	
	tPlot, = plt.loglog(nValues, trapezoidErrors, label='Trapezoidal Error')
	sPlot, = plt.loglog(nValues, simpsonsErrors, label='Simpson\'s Error')
	plt.legend(handles=[tPlot, sPlot])
	plt.ylabel('log Error')
	plt.xlabel('log N')
	plt.title('Trapezoidal and Simpson\'s Errors')
	plt.show()

def evaluateToAccuracy(func, a, b, errorCap):
	n0 = 4
	k = 0
	error = 2*errorCap
	while error > errorCap:
		simps2k = extendedSimpsons(func, a, b, 2**k * n0)
		error = abs((simps2k - extendedSimpsons(func, a, b, 2**(k+1) * n0)) / simps2k)
		k += 1
	print 'Total number of iterations: ' + str(k)
	print 'Relative error: ' + str(error)
	return extendedSimpsons(func, a, b, 2**(k-1) * n0)

def testAccuracy():
	I_simp = evaluateToAccuracy(eX, 0, 1, 1E-13)
# 	print 'Q6 absolute error for e^x: ' + str(np.abs(I_simp - (np.e-1)))

	I_simp = evaluateToAccuracy(sinX, 0, np.pi, 1E-13)
# 	print 'Q6 absolute error for sin(x): ' + str(np.abs(I_simp - 2))

def compareIntegrationMethods():
	print 'Integrating e^x from 0 to 1 with quad: ' + repr(integrate.quad(eX, 0, 1)[0])
	print 'Integrating e^x from 0 to 1 with romberg: ' + repr(integrate.romberg(eX, 0, 1))
	print 'Integrating e^x from 0 to 1 with trapezoid: ' + repr(extendedTrapezoid(eX, 0, 1, 2**20))
	print 'Integrating e^x from 0 to 1 with simpsons: ' + repr(extendedSimpsons(eX, 0, 1, 2**20))
	print 'Analytical integration of e^x from 0 to 1: ' + repr(np.e-1)

# q2
print(extendedTrapezoid(eX, 0, 1, 100000))

# q3
print(extendedSimpsons(eX, 0, 1, 100000))

# q4
plotError()

# q6
testAccuracy()

# q7
compareIntegrationMethods()