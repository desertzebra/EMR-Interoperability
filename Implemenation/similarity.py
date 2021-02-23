class Similarity:

	def readData(self, url):
		
		data = []
		file = open(url, "r")

		for row in file:
			data.append(row)

		return data
  		


simObj = Similarity()
data = simObj.readData('Data/attempt4.txt') 
print(len(data))
		