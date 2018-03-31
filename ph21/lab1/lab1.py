import numpy as np
import tempfile
import urllib.request
import re
import matplotlib.pyplot as plt
from astropy.io.votable import parse_single_table


#
# PART I
#

# gets data from line with JSON format that contains keyword
def extractData(variable, html):
	# first find line with variable name, then put JSON object into a string
	varIndex = html.find(variable)
	line = html[html.find('=', varIndex)+1 : html.find('\n', varIndex)].strip()
	
	# extract data entry from JSON object, assuming data is the last entry in the object
	dataStr = line[line.find('[', line.find('data:')) : line.find('}')]
	
	# place into list
	data = []
	entryPattern = re.compile('\[\s+\d[0-9,.\s]+\]')
	valPattern = re.compile('[0-9.]+')
	
	outer = entryPattern.finditer(dataStr)
	for match in outer:
		entry = []
		inner = valPattern.finditer(match.group(0))
		for val in inner:
			entry.append(float(val.group(0)))
		data.append(entry)
	
	return np.array(data)

# plots from a 2d list using the first two elements of each inner list
def plotData(data):
	x = [item[0] for item in data]
	y = [item[1] for item in data]
	plt.xlabel('Phase')
	plt.ylabel('V mag')
	plt.title('Light Curves')
	plt.plot(x, y, 'bo', ms=3)
	plt.show()
	
ascii = urllib.request.urlopen('http://nesssi.cacr.caltech.edu/cgi-bin/getcssconedb_id_phase.cgi?ID=1135075045477&PLOT=plot')
source = ascii.read().decode('ascii')
asciiData = extractData('var dataSet0', source)
plotData(asciiData)


#
# PART II
#

def extractVotData(vot):
	data = []
	with tempfile.TemporaryFile() as tmp:
		tmp.write(votSource)
		votable = parse_single_table(tmp).array
		times = [item[0] for item in votable['ObsTime']]
		mags = [item[0] for item in votable['Mag']]
		data = [times, mags]
	return data

def plotVotData(x, y):
	plt.xlabel('Obs Time')
	plt.ylabel('V mag')
	plt.title('Light Curves')
	plt.plot(x, y, 'bo', ms=3)
	plt.show()

vot = urllib.request.urlopen('http://nesssi.cacr.caltech.edu/DataRelease/upload/result_web_fileHTlmTa.vot')
votSource = vot.read()
votData = extractVotData(votSource)
plotVotData(votData[0], votData[1])