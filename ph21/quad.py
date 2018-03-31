import numpy as np

# E, ignoring kq (just returns sgn(q)/r^2)
def calcE(r, q):
	return np.sign(q)/r**2

def calcQuadE(l, r):
	r_q1 = r - l/np.sqrt(2)
	r_q2 = r + l/np.sqrt(2)
	r_q3 = np.sqrt(r**2 + l**2/2)

	totalE = calcE(r_q1, 1) + calcE(r_q2, 1) + 2*calcE(r_q3, -1)
	return totalE

l = 1
for i in range(100):
	r = (i+1)*10
	ratio = calcQuadE(l, r) * r**4/l**2
	print(ratio)