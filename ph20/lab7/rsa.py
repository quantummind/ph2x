from fractions import gcd
import random
import math

def isPrime(n):
	return all(n % i for i in range(3, int(math.sqrt(n))+1, 2))

def isCoprime(a, b):
	return gcd(a, b) == 1

def lcm(a, b):
	return int(a*b / gcd(a, b))

# from Wikibooks, extended Euclidean algorithm
def egcd(a, b):
	if a == 0:
		return(b, 0, 1)
	else:
		g, x, y = egcd(b % a, a)
		return (g, y - (b // a) * x, x)
def modInv(b, n):
	g, x, _ = egcd(b, n)
	if g == 1:
		return x % n


primes = [2*i+1 for i in range(128, 256) if isPrime(2*i+1)]

# returns [public key, private key], where public key = (n, e) and private key = (n, d)
def generateKeys():
	p = random.choice(primes)
	q = p
	while q == p:
		q = random.choice(primes)
	
	n = p*q
	lambda_n = lcm(p-1, q-1)
	
	e = 2**4
	while not isCoprime(lambda_n, e):
		e += 1
	
	d = modInv(e, lambda_n)
	
	return [(n, e), (n, d)]


def encryptMessage(msg, publicKey):
	return msg**publicKey[1] % publicKey[0]

def decryptMessage(cipher, privateKey):
	return cipher**privateKey[1] % privateKey[0]

# generate Alice's public and private keys
aliceKeys = generateKeys()
print('Alice\'s keys: ', aliceKeys)

# message that Bob wants to send Alice
mBob = 137

# cipher that Bob transmits, given Alice's public key
c = encryptMessage(mBob, aliceKeys[0])

# Alice decrypts
mAlice = decryptMessage(c, aliceKeys[1])

print('Original message ({}) -> cipher ({}) -> decrypted message({})'.format(mBob, c, mAlice))

# Eve tries collision attack on the cipher c
possibleMessages = []
for i in range(1000000):
	if encryptMessage(i, aliceKeys[0]) == c:
		possibleMessages.append(i)

print('Eve\'s possible messages: ', possibleMessages)