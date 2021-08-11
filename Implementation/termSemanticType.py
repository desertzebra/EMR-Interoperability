
import requests
import json
import lxml.html as lh
from lxml.html import fromstring

profileKey = "ec98614f-6197-4a85-bf9e-1b20bef64670"
baseURL = "https://utslogin.nlm.nih.gov"

class TermSemanticType:

	def __init__(self):

		self.tgt = self.gettgt()

	#  *********** TGT Ticket ********************
	def gettgt(self):

		params = {'apikey': profileKey}
		response = requests.post(baseURL + "/cas/v1/tickets", data = {'apikey': profileKey})
		response = fromstring(response.text)
		tgt = response.xpath('//form/@action')[0]

		return tgt

	#  ********** Single use Service Ticket ********************
	def getst(self):

		params = {'service': 'http://umlsks.nlm.nih.gov'}
		h = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain", "User-Agent":"python"}
		r = requests.post(self.tgt,data=params,headers=h)
		st = r.text

		return st

	#  **************** Search Term ***********************
	def searchTerm(self, term):

		singleUseTicket = self.getst()

		r = requests.get('https://uts-ws.nlm.nih.gov/rest/search/current?string='+term+'&ticket='+singleUseTicket)
		items  = json.loads(r.text)
		jsonData = items['result']['results']

		return jsonData

	#  ***************** Get Term Semantic Type **************
	def getTermSemanticType(self, term):

		response = 'NONE'
		try :
			termId = self.searchTerm(term)[0]['ui']
		except NameError:
			termId = 'NONE'

		if not termId == "NONE":
			singleUseTicket = self.getst()
			r = requests.get('https://uts-ws.nlm.nih.gov/rest/content/2015AA/CUI/'+termId+'?ticket='+singleUseTicket)
			
			if(r.status_code == 200):
				items  = json.loads(r.text)
				try:
					response = items
					# response = items['result']['semanticTypes'][0]['name']
				except NameError:
					response = 'NONE'
					# response = 'NONE'

		return response


# termSemanticTypeObj = TermSemanticType()

# print('Semantic Type')
# tokens = ("Discussed risks, goals, alternatives, advance directives, and the necessity of other members of the healthcare team participating in the procedure with the patient and his mother.").split()


# for token in tokens:
# 	print('token: ' + token)
	
# 	print(termSemanticTypeObj.getTermSemanticType(token))
# # print(termSemanticTypeObj.getTermSemanticType(("Discussed goals, risks, alternatives, advanced directives, and the necessity of other members of the surgical team participating in the procedure with the patient.").split())) #abcedeede
