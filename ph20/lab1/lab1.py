import sys
import numpy as np
import matplotlib.pyplot as plt

tVals = []
xVals = []
yVals = []
zVals = []


def X(Ax, fx, t):
	return Ax * np.cos(2*np.pi * fx * t)
def Y(Ay, fy, t, phi):
	return Ay * np.sin(2*np.pi * fy * t + phi)
def Z(Ax, Ay, fx, ty, t, phi):
	return X(Ax, fx, t) + Y(Ay, fy, t)
def Z_precompute(xVal, yVal):
	return xVal + yVal
	
def writeToFile(args):
	file = open('./lab1-output.txt', 'w')
	file.write('System params: ')
	file.write('f_X=' + str(a[1]) + ', f_Y=' + str(a[2]) + ', A_X=' + str(a[3]) + ', A_Y=' + str(a[4]) + ', phi=' + str(a[5]) + ', dt=' + str(a[6]) + ', N=' + str(a[7]))
	file.write('\n\n')
	file.write('t\tX\tY\tZ\n')
	for i in range(len(xVals)):
		file.write(str(tVals[i]) + '\t' + str(xVals[i]) + '\t' + str(yVals[i]) + '\t' + str(zVals[i]) + '\n')
	file.close()

def step(Ax, Ay, fx, fy, t, phi):
	tVals.append(t)
	xVals.append(X(Ax, fx, t))
	yVals.append(Y(Ay, fy, t, phi))
	zVals.append(Z_precompute(xVals[-1], yVals[-1]))

def run(fx, fy, Ax, Ay, phi, dt, N):
	for i in range(N+1):
		step(Ax, Ay, fx, fy, i*dt, phi)

def plotXYZ():
	file = open('./lab1-output.txt', 'w')
	file.write('t\tX\tY\tZ\n')
	for i in range(len(xVals)):
		file.write(str(tVals[i]) + '\t' + str(xVals[i]) + '\t' + str(yVals[i]) + '\t' + str(zVals[i]) + '\n')
	file.close()
	
	xPlot, = plt.plot(tVals, xVals, label='X(t)')
	yPlot, = plt.plot(tVals, yVals, label='Y(t)')
	zPlot, = plt.plot(tVals, zVals, label='Z(t)')
	plt.legend(handles=[xPlot, yPlot, zPlot])
	plt.ylabel('X(t), Y(t), Z(t)')
	plt.xlabel('t')
	plt.title('X, Y, Z')
	plt.show()

def plotLissajous():
	plt.ylabel('X(t)')
	plt.xlabel('Y(t)')
	plt.title('Lissajous')
	plt.plot(yVals, xVals)
	plt.show()
	
def plotBeats():
	plt.ylabel('Z(t)')
	plt.xlabel('t')
	plt.title('Beats')
	plt.plot(tVals, zVals)
	plt.show()
	
if len(sys.argv) == 8:
	a = sys.argv
	run(float(a[1]), float(a[2]), float(a[3]), float(a[4]), float(a[5]), float(a[6]), int(a[7]))
	writeToFile(sys.argv)
	
	# q2
# 	plotXYZ()
	
	# q3
# 	plotLissajous()
	
	# q4
	plotBeats()