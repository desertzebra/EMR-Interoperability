import ujson as json
import copy
import pickle
from sentence_transformers import SentenceTransformer, util
from joblib import Parallel, delayed
import multiprocessing
import itertools


class Similarity:

    def __init__(self):
        self.embeddingVectors = {}
        self.models = ['bert-large-nli-stsb-mean-tokens', 'bert-base-nli-mean-tokens', 'bert-base-nli-stsb-mean-tokens']
        # self.models = ['bert-base-nli-stsb-mean-tokens']

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
        model = SentenceTransformer(modelName, device='cuda')

        for i, attr in enumerate(attributes):
            print(modelName, ' line i: ', i)

            formatedAttr = copy.deepcopy(attr)

            embedding1 = model.encode(attr['nodeLeft']['name'], convert_to_tensor=True)
            embedding2 = model.encode(attr['nodeRight']['name'], convert_to_tensor=True)

            similarity = util.pytorch_cos_sim(embedding1, embedding2)
            similarity = "{:.4f}".format(similarity[0][0])

            formatedAttr['relationshipList'].append(
                {'type': '', 'method': modelName, 'comments': None, 'confidence': similarity})

            attributeSimilarities.append(formatedAttr)

        return attributeSimilarities

    def isDataTypeConvertable(self, type1, type2):

        result = False

        if ((type1 == 'Long' and type2 == 'Double') or (type1 == 'Double' and type2 == 'Long')):
            result = True
        elif ((type1 == 'Date' and type2 == 'DateTime') or (type1 == 'DateTime' and type2 == 'Date')):
            result = True

        return result

    # @staticmethod
    # def createSentenceForAA(node):
    #     amplified_sentences = []
    #     word_concept_map = {}
    #     # for word in node['suffixArray']:
    #     #     word_concept_map["word"] = []
    #     amplified_sentences.append(" ".join(node['suffixArray']))
    #     for concept in node['conceptArray']:
    #         if concept['token'] not in word_concept_map:
    #             word_concept_map[concept['token']] = []
    #         word_concept_map[concept['token']].append(concept['name'])
    #     conceptset = word_concept_map.values()
    #     product_concepts = list(itertools.product(*conceptset))
    #     for concept_product_tuple in product_concepts:
    #         amplified_sentences.append(" ".join(concept_product_tuple))
    #
    #     return amplified_sentences
    @staticmethod
    def createSentenceForAA(node):
        amplified_sentences = []
        # word_concept_map = {}
        # for word in node['suffixArray']:
        #     word_concept_map["word"] = []
        amplified_sentences.extend(node['suffixArray'])
        for concept in node['conceptArray']:
            amplified_sentences.append(concept['name'])

        return amplified_sentences


    def getSyntacticSimilarity(self, attr):

        similarity = 0

        if ((attr['nodeLeft']['dataType'] == attr['nodeRight']['dataType']) or (
        self.isDataTypeConvertable(attr['nodeLeft']['dataType'], attr['nodeRight']['dataType']))):
            similarity = 1

        return similarity

    def getSemanticSimilarity(self, attrPair, model):

        # formatedattrPair = copy.deepcopy(attrPair)
        nodeLeftIdentifier = "_".join([attrPair['nodeLeft']['schemaName'], attrPair['nodeLeft']['tableName'],
                                       attrPair['nodeLeft']['name'], attrPair['nodeLeft']['schemaVersion']])
        nodeRightIdentifier = "_".join([attrPair['nodeRight']['schemaName'], attrPair['nodeRight']['tableName'],
                                        attrPair['nodeRight']['name'], attrPair['nodeRight']['schemaVersion']])

        self.embeddingVectors[nodeLeftIdentifier] = Similarity.createSentenceForAA(attrPair['nodeLeft'])
        self.embeddingVectors[nodeRightIdentifier] = Similarity.createSentenceForAA(attrPair['nodeRight'])

        # print("sentences in left:",len(self.embeddingVectors[nodeLeftIdentifier]))
        # print(self.embeddingVectors[nodeLeftIdentifier])
        # print("sentences in right:",len(self.embeddingVectors[nodeRightIdentifier]))
        # print(self.embeddingVectors[nodeRightIdentifier])
        # memoization to save time on calculating the embedding vector
        sentence_encoding_map = {}
        max_left_term_similarity = 0.0
        for term_left in self.embeddingVectors[nodeLeftIdentifier]:
            if term_left not in sentence_encoding_map:
                embedding1 = model.encode(term_left, convert_to_tensor=True)
                sentence_encoding_map[term_left] = embedding1
            else:
                embedding1 = sentence_encoding_map[term_left]
            max_right_term_similarity = 0.0
            # this similarity is between 0 and 1
            for term_right in self.embeddingVectors[nodeRightIdentifier]:
                if term_right not in sentence_encoding_map:
                    embedding2 = model.encode(term_right, convert_to_tensor=True)
                    sentence_encoding_map[term_right] = embedding2
                else:
                    embedding2 = sentence_encoding_map[term_right]

                term_similarity = util.pytorch_cos_sim(embedding1, embedding2)

                if float(term_similarity[0][0]) > max_right_term_similarity:
                    max_right_term_similarity = float("{:.4f}".format(term_similarity[0][0]))
            max_left_term_similarity += max_right_term_similarity

        # calculate the average similarity of the disjoint union of two sets
        avg_similarity = max_left_term_similarity / len(self.embeddingVectors[nodeLeftIdentifier])
        # print("avg_similarity:",avg_similarity)
        # nodeLeftContext = self.embeddingVectors[nodeLeftIdentifier]
        # nodeRightContext = self.embeddingVectors[nodeRightIdentifier]

        # embedding1 = model.encode(nodeLeftContext, convert_to_tensor=True)
        # embedding2 = model.encode(nodeRightContext, convert_to_tensor=True)
        #
        # similarity = util.pytorch_cos_sim(embedding1, embedding2)
        # similarity = "{:.4f}".format(similarity[0][0])

        # avg_similarity =
        return avg_similarity

    def calculateAmplifiedSimilarity(self, attributes, modelName):

        attributeSimilarities = []

        print('len(attributes)')
        print(len(attributes))

        model = SentenceTransformer(modelName)

        for index, attrPair in enumerate(attributes):
            formatedAttr = copy.deepcopy(attrPair)

            syntacticSimilarity = self.getSyntacticSimilarity(attrPair)
            semanticSimilarity = self.getSemanticSimilarity(attrPair, model)
            # similarity = self.getSemanticSimilarity(attrPair, model)

            print("syntacticSimilarity: ", syntacticSimilarity, ', semanticSimilarity: ', semanticSimilarity)
            # print('Semantic index: ', index)

            similarity = (0.5 * float(syntacticSimilarity)) + (0.5 * float(semanticSimilarity))

            formatedAttr['relationshipList'].append(
                {'type': '', 'method': modelName + '_SYN_AND_SEM_MATCH', 'comments': None, 'confidence': similarity})
            attributeSimilarities.append(formatedAttr)

        return attributeSimilarities

    # function optimized to run on gpu
    # @jit(target="cuda")

    # def processSchemaMapNode(self, attrPair, model):
    #     # formatedattrPair = copy.deepcopy(attrPair)
    #     nodeLeftIdentifier = "_".join([attrPair['nodeLeft']['schemaName'], attrPair['nodeLeft']['tableName'],
    #                                    attrPair['nodeLeft']['name'], attrPair['nodeLeft']['schemaVersion']])
    #     nodeRightIdentifier = "_".join([attrPair['nodeRight']['schemaName'], attrPair['nodeRight']['tableName'],
    #                                     attrPair['nodeRight']['name'], attrPair['nodeRight']['schemaVersion']])
    #
    #     self.embeddingVectors[nodeLeftIdentifier] = self.createSentenceForAA(attrPair['nodeLeft'])
    #     self.embeddingVectors[nodeRightIdentifier] = self.createSentenceForAA(attrPair['nodeRight'])
    #
    #     # if nodeLeftIdentifier not in self.embeddingVectors:
    #     #     self.embeddingVectors[nodeLeftIdentifier] = self.createSentenceForAA(attrPair['nodeLeft'])
    #     #
    #     # if nodeRightIdentifier not in self.embeddingVectors:
    #     #     self.embeddingVectors[nodeRightIdentifier] = self.createSentenceForAA(attrPair['nodeRight'])
    #     max_similarity = 0.0
    #     max_sentence_left = ""
    #     max_sentence_right = ""
    #     for sentence_left in self.embeddingVectors[nodeLeftIdentifier]:
    #         for sentence_right in self.embeddingVectors[nodeRightIdentifier]:
    #             embedding1 = model.encode(sentence_left, convert_to_tensor=True)
    #             embedding2 = model.encode(sentence_right, convert_to_tensor=True)
    #             similarity = util.pytorch_cos_sim(embedding1, embedding2)
    #             similarity = "{:.4f}".format(similarity[0][0])
    #             if float(similarity) > max_similarity:
    #                 max_similarity = float(similarity)
    #                 max_sentence_left = sentence_left
    #                 max_sentence_right = max_sentence_right
    #
    #     i = 0
    #     while i < len(attrPair['relationshipList']):
    #         if attrPair['relationshipList'][i]['method'] == 'SYN_AND_SEM_MATCH':
    #             attrPair['relationshipList'][i]['sentence_left'] = max_sentence_left
    #             attrPair['relationshipList'][i]['sentence_right'] = max_sentence_right
    #             attrPair['relationshipList'][i]['confidence'] = max_similarity
    #             break
    #
    #     return attrPair

    def calculateSimilarity(self, attributes):

        for model in self.models:
            print('model: ', model)
            attributes = self.calculateAmplifiedSimilarity(attributes, model)
            attributes = self.calcualteBaseLineSimilarity(attributes, model)

        return attributes


num_cores = multiprocessing.cpu_count() - 1
print("num_cores:", num_cores)

simObj = Similarity()
data = simObj.readData('Data/schema_processed_1617174847206.json')

print('data')
print(len(data))

# data = simObj.calcualteBaseLineSimilarity(data)


similarity = simObj.calculateSimilarity(data)

# synAndSemSimilarity = simObj.calculateAmplifiedSimilarity(data)

# print('synAndSemSimilarity')
# print(synAndSemSimilarity)

# print('synAndSemSimilarity')
# print(synAndSemSimilarity)

with open("Data/AmplifiedSimilarity-V0.2.txt", "wb") as fp:  # Pickling
    pickle.dump(similarity, fp)
