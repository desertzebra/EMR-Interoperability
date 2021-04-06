import ujson as json
import copy
import pickle
from sentence_transformers import SentenceTransformer, util
from joblib import Parallel, delayed
import multiprocessing


class Similarity:

    def __init__(self):
        self.embeddingVectors = {}
        self.models = ['bert-base-nli-mean-tokens', 'bert-base-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens']

    def readData(self, url):

        with open(url) as json_file:
            data = json.load(json_file)

        # data = []
        # with open(url, "rb") as fp:
        #     data = pickle.load(fp)

        return data

    def getSimpleAttributes(self, attributes):

        print('in getSimpleAttributes function')

        data = []
        for attr in attributes:
            data.append(attr['schemaName'] + ' ' + attr['tableName'] + ' ' + attr['name'])

        return data

    def calcualteBaseLineSimilarity(self, attributes, modelName):

        print('len(attributes)')
        print(len(attributes))

        attributeSimilarities = []

       # model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
        model = SentenceTransformer(modelName)

        for i, attr in enumerate(attributes):

            print(modelName, ' line i: ', i)

            formatedAttr = copy.deepcopy(attr)

            embedding1 = model.encode(attr['nodeLeft']['name'], convert_to_tensor=True)
            embedding2 = model.encode(attr['nodeRight']['name'], convert_to_tensor=True)

            similarity = util.pytorch_cos_sim(embedding1, embedding2)
            similarity = "{:.4f}".format(similarity[0][0])

            formatedAttr['relationshipList'].append({'type': '', 'method': modelName, 'comments': None, 'confidence': similarity})

            attributeSimilarities.append(formatedAttr)


        return attributeSimilarities

    def isDataTypeConvertable(self, type1, type2):

        result = False

        if((type1 == 'Long' and type2 == 'Double') or (type1 == 'Double' and type2 == 'Long')):
            result = True
        elif((type1 == 'Date' and type2 == 'DateTime') or (type1 == 'DateTime' and type2 == 'Date')):
            result = True

        return  result

    def getSyntacticSimilarity(self, attr):

        similarity = 0

        if( (attr['nodeLeft']['dataType'] == attr['nodeRight']['dataType']) or (self.isDataTypeConvertable(attr['nodeLeft']['dataType'], attr['nodeRight']['dataType']))):
            similarity = 1

        return similarity

    def getSemanticSimilarity(self, attrPair, model):

        # formatedattrPair = copy.deepcopy(attrPair)
        nodeLeftIdentifier = "_".join([attrPair['nodeLeft']['schemaName'], attrPair['nodeLeft']['tableName'],
                                       attrPair['nodeLeft']['name'], attrPair['nodeLeft']['schemaVersion']])
        nodeRightIdentifier = "_".join([attrPair['nodeRight']['schemaName'], attrPair['nodeRight']['tableName'],
                                        attrPair['nodeRight']['name'], attrPair['nodeRight']['schemaVersion']])

        self.embeddingVectors[nodeLeftIdentifier] = self.createSentenceForAA(attrPair['nodeLeft'])
        self.embeddingVectors[nodeRightIdentifier] = self.createSentenceForAA(attrPair['nodeRight'])

        nodeLeftContext = self.embeddingVectors[nodeLeftIdentifier]
        nodeRightContext = self.embeddingVectors[nodeRightIdentifier]

        embedding1 = model.encode(nodeLeftContext, convert_to_tensor=True)
        embedding2 = model.encode(nodeRightContext, convert_to_tensor=True)

        similarity = util.pytorch_cos_sim(embedding1, embedding2)
        similarity = "{:.4f}".format(similarity[0][0])

        return similarity

    def calculateAmplifiedSimilarity(self, attributes, modelName):

        attributeSimilarities = []

        print('len(attributes)')
        print(len(attributes))

        model = SentenceTransformer(modelName)

        for index, attrPair in enumerate(attributes):

            formatedAttr = copy.deepcopy(attrPair)

            syntacticSimilarity = self.getSyntacticSimilarity(attrPair)
            semanticSimilarity = self.getSemanticSimilarity(attrPair, model)

            # print("syntacticSimilarity: ", syntacticSimilarity, ', semanticSimilarity: ', semanticSimilarity)
            print('Semantic index: ', index)

            similarity = (0.5 * float(syntacticSimilarity)) + (0.5 * float(semanticSimilarity))

            formatedAttr['relationshipList'].append({'type': '', 'method': modelName+'_SYN_AND_SEM_MATCH', 'comments': None, 'confidence': similarity})
            attributeSimilarities.append(formatedAttr)

        return attributeSimilarities

    def createSentenceForAA(self, node):
        amplifiedWordList = []
        result = ''
        for word in node['suffixArray']:
            wordWithItsConcept = word
            for concept in node['conceptArray']:
                if word == concept["token"]:
                    wordWithItsConcept = " ".join([wordWithItsConcept, ",", concept["name"]])
            amplifiedWordList.append(wordWithItsConcept)

        result = " ".join([node['name'], "<", ";".join(amplifiedWordList), ">"])


        return result

    def processSchemaMapNode(self, attrPair, model):
        # formatedattrPair = copy.deepcopy(attrPair)
        nodeLeftIdentifier = "_".join([attrPair['nodeLeft']['schemaName'], attrPair['nodeLeft']['tableName'],
                                       attrPair['nodeLeft']['name'], attrPair['nodeLeft']['schemaVersion']])
        nodeRightIdentifier = "_".join([attrPair['nodeRight']['schemaName'], attrPair['nodeRight']['tableName'],
                                        attrPair['nodeRight']['name'], attrPair['nodeRight']['schemaVersion']])

        self.embeddingVectors[nodeLeftIdentifier] = self.createSentenceForAA(attrPair['nodeLeft'])
        self.embeddingVectors[nodeRightIdentifier] = self.createSentenceForAA(attrPair['nodeRight'])

        # if nodeLeftIdentifier not in self.embeddingVectors:
        #     self.embeddingVectors[nodeLeftIdentifier] = self.createSentenceForAA(attrPair['nodeLeft'])
        #
        # if nodeRightIdentifier not in self.embeddingVectors:
        #     self.embeddingVectors[nodeRightIdentifier] = self.createSentenceForAA(attrPair['nodeRight'])

        nodeLeftContext = self.embeddingVectors[nodeLeftIdentifier]
        nodeRightContext = self.embeddingVectors[nodeRightIdentifier]

        embedding1 = model.encode(nodeLeftContext, convert_to_tensor=True)
        embedding2 = model.encode(nodeRightContext, convert_to_tensor=True)

        similarity = util.pytorch_cos_sim(embedding1, embedding2)
        similarity = "{:.4f}".format(similarity[0][0])

        i = 0
        while i < len(attrPair['relationshipList']):
            if attrPair['relationshipList'][i]['method'] == 'SYN_AND_SEM_MATCH':
                attrPair['relationshipList'][i]['confidence'] = similarity
                break

        return attrPair

    def calculateSimilarity(self, attributes):

        for model in self.models:

            print('model: ', model)

            attributes = self.calcualteBaseLineSimilarity(attributes, model)
            attributes = self.calculateAmplifiedSimilarity(attributes, model)

        return attributes







num_cores = multiprocessing.cpu_count() - 1
print("num_cores:", num_cores)

simObj = Similarity()
data = simObj.readData('Data/schema_processed_1617174847206.json')


print('data')
print(len(data))

#data = simObj.calcualteBaseLineSimilarity(data)


similarity = simObj.calculateSimilarity(data)




#synAndSemSimilarity = simObj.calculateAmplifiedSimilarity(data)

# print('synAndSemSimilarity')
# print(synAndSemSimilarity)

# print('synAndSemSimilarity')
# print(synAndSemSimilarity)

with open("Data/AmplifiedSimilarity-V0.1.txt", "wb") as fp:   #Pickling
    pickle.dump(similarity, fp)

