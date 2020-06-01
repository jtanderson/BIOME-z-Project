import xml.etree.ElementTree as ET

tree = ET.parse('pubmedProj.xml')
root = tree.getroot()

numKeyword = 0
thislist = []
for neighbor in root.iter('keyword'):
	numKeyword = numKeyword + 1;

for neighbor in root.iter('keyword'): 
	thislist.append(neighbor.text);

titles[]
numPapers = 0
for x in root.iter('title'):
	numPapers = numPapers + 1;
	titles.append(x.text);

paperTags = [0] * numPapers
for i in range(numPapers):
	paperTags[i] = [0] * numKeyword

for a in root.iter('title'):
	for
