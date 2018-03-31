import sys
import numpy as np
import matplotlib.pyplot as plt

tVals = np.array([])
xVals = np.array([])
yVals = np.array([])
zVals = np.array([])


def run(fx, fy, Ax, Ay, phi, dt, N):
	global tVals, xVals, yVals, zVals
	tVals = np.linspace(0, int(N*dt), N)
	xVals = Ax * np.cos(2*np.pi * fx * tVals)
	yVals = Ax * np.sin(2*np.pi * fy * tVals + phi)
	zVals = xVals + yVals

def writeToFile(args):
	file = open('./lab1-np-output.txt', 'w')
	file.write('System params: ')
	file.write('f_X=' + str(a[1]) + ', f_Y=' + str(a[2]) + ', A_X=' + str(a[3]) + ', A_Y=' + str(a[4]) + ', phi=' + str(a[5]) + ', dt=' + str(a[6]) + ', N=' + str(a[7]))
	file.write('\n\n')
	file.write('t\tX\tY\tZ\n')
	for i in range(len(xVals)):
		file.write(str(tVals[i]) + '\t' + str(xVals[i]) + '\t' + str(yVals[i]) + '\t' + str(zVals[i]) + '\n')
	file.close()
	
def plotXYZ():
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
	
# take args in order: f_X, f_Y, A_X, A_Y, phi, delta t, N
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