import json
from termSemanticType import *

class Similarity:

    def __init__(self):
    	self.termSemanticTypeObj = TermSemanticType()

    def readData(self, url):
    	data = []
    	with open(url) as json_file:
    		data = json.load(json_file)

    	return data

    def calculateSimilarity(self, url):

    	result = self.termSemanticTypeObj.getTermSemanticType('diabetes')
    	print(result)
    	exit()
    	data = self.readData(url)
    	print('Total Attribute: ', len(data))
    	for attr in data:
    		print('attr')
    		print(attr)
    		print(attr['schemaName'])
    		exit()


simObj = Similarity()
simObj.calculateSimilarity('Data/data.json') 

		