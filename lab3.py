import numpy as np
import matplotlib.pyplot as plt


def stepEulerExplicit(x_i, v_i, h):
	return [x_i + h*v_i, v_i - h*x_i]

def stepEulerSpringImplicit(x_i, v_i, h):
# 	x_i+1 = (x_i + h v_i) / (1+h^2)
# 	v_i+1 = (v_i - h x_i) / (1 - h^2)
	x_i1 = (x_i + h*v_i) / (1+h**2)
	return [x_i1, v_i - h*x_i1]

def stepEulerSpringSymplectic(x_i, v_i, h):
	# x_i+1 = x_i + h v_i
	# v_i+1 = v_u - h(x_i + hv_i+1)
	x_i1 = x_i + h*v_i
	return [x_i1, v_i - h*x_i1]

initX = 0
initV = 10

def eulerSpringSpecific(h, plot, step):
	N = int(50/h)

	tVals = np.linspace(0, int(N*h), N)
	xVals = np.zeros(N)
	vVals = np.zeros(N)
	xVals[0] = initX
	vVals[0] = initV
	for i in range(N-1):
		stepResult = step(xVals[i], vVals[i], h)
		xVals[i+1] = stepResult[0]
		vVals[i+1] = stepResult[1]
	
	if plot:
		plt.ylabel('X(t)')
		plt.xlabel('t')
		plt.title('Position')
		plt.plot(tVals, xVals)
		plt.show()
	
		plt.ylabel('V(t)')
		plt.xlabel('t')
		plt.title('Velocity')
		plt.plot(tVals, vVals)
		plt.show()
	
	return [tVals, xVals, vVals]

def eulerSpringExplicit(h, plot):
	return eulerSpringSpecific(h, plot, stepEulerExplicit)


def eulerSpringImplicit(h, plot):
	return eulerSpringSpecific(h, plot, stepEulerSpringImplicit)

def eulerSpringSymplectic(h, plot):
	return eulerSpringSpecific(h, plot, stepEulerSpringSymplectic)


def eulerSpringError(txv):
	tVals = txv[0]
	xVals = txv[1]
	vVals = txv[2]
	
	xError = initX * np.cos(tVals) + initV * np.sin(tVals) - xVals
	vError = -initX * np.sin(tVals) + initV * np.cos(tVals) - vVals
	
	plt.ylabel('X(t) Error')
	plt.xlabel('t')
	plt.title('Position Error')
	plt.plot(tVals, xError)
	plt.show()
	
	plt.ylabel('V(t) Error')
	plt.xlabel('t')
	plt.title('Velocity Error')
	plt.plot(tVals, vError)
	plt.show()

def eulerSpringMaxError(txv):
	tVals = txv[0]
	xVals = txv[1]
	vVals = txv[2]
	return np.amax(initX * np.cos(tVals) + initV * np.sin(tVals) - xVals)

def hError(eulerSpring):
	N = 5
	h0 = 0.001
	hVals = h0 / 2**np.arange(0, N)
	errors = np.zeros(len(hVals))
	for i in range(len(hVals)):
		errors[i] = eulerSpringMaxError(eulerSpring(hVals[i], False))
	plt.ylabel('X(t) Max Error')
	plt.xlabel('h')
	plt.title('Error vs. h')
	plt.plot(hVals, errors)
	plt.show()

def springEnergy(eulerSpring):
	txv = eulerSpring(0.0001, False)
	eVals = txv[1]**2 + txv[2]**2
	
	plt.ylabel('E(t)')
	plt.xlabel('t')
	plt.title('Energy')
	plt.plot(txv[0], eVals)
	plt.show()

def plotPhaseSpace(txv):
	plt.ylabel('V(t)')
	plt.xlabel('X(t)')
	plt.title('Phase Space')
	plt.plot(txv[1], txv[2])
	plt.show()

def symplecticPhaseLag():
	h = 0.03
	N = int(5000/h)
	end = int(50/h)

	tVals = np.linspace(0, int(N*h), N)
	xVals = np.zeros(N)
	vVals = np.zeros(N)
	xVals[0] = initX
	vVals[0] = initV
	for i in range(N-1):
		stepResult = stepEulerSpringSymplectic(xVals[i], vVals[i], h)
		xVals[i+1] = stepResult[0]
		vVals[i+1] = stepResult[1]
	exactX = initX * np.cos(tVals) + initV * np.sin(tVals)
	exactV = -initX * np.sin(tVals) + initV * np.cos(tVals)
	
	xPlotEuler, = plt.plot(tVals[-end:], xVals[-end:], label='X(t) Eulerian')
	xPlotAnalytic, = plt.plot(tVals[-end:], exactX[-end:], label='X(t) Analytic')
	plt.legend(handles=[xPlotEuler, xPlotAnalytic])
	plt.ylabel('Eulerian, Analytic')
	plt.xlabel('t')
	plt.title('X(t) Eulerian and Analytic Solutions')
	plt.show()

	vPlotEuler, = plt.plot(tVals[-end:], vVals[-end:], label='V(t) Eulerian')
	vPlotAnalytic, = plt.plot(tVals[-end:], exactV[-end:], label='V(t) Analytic')
	plt.legend(handles=[vPlotEuler, vPlotAnalytic])
	plt.ylabel('Eulerian, Analytic')
	plt.xlabel('t')
	plt.title('V(t) Eulerian and Analytic Solutions')
	plt.show()

# 1.1
txv = eulerSpringExplicit(0.001, True)

# 1.2
eulerSpringError(txv)

# 1.3
hError(eulerSpringExplicit)

# 1.4
springEnergy(eulerSpringExplicit)

# 1.5
txv = eulerSpringImplicit(0.001, True)
eulerSpringError(txv)
hError(eulerSpringImplicit)
springEnergy(eulerSpringImplicit)

# 2.1
plotPhaseSpace(eulerSpringExplicit(0.01, False))
plotPhaseSpace(eulerSpringImplicit(0.01, False))

# 2.2
plotPhaseSpace(eulerSpringSymplectic(0.01, False))

# 2.3
springEnergy(eulerSpringSymplectic)
txv = eulerSpringSymplectic(0.01, True)
eulerSpringError(txv)

# 2.4
symplecticPhaseLag()