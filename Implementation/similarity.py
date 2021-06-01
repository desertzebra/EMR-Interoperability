import ujson as json
import copy
import pickle
from sentence_transformers import SentenceTransformer, util
from joblib import Parallel, delayed
import multiprocessing
import itertools
import torch
from numba import njit, cuda, prange
from numba import jit
import numpy as np
from fastdist import fastdist
from numpy import dot
from numpy.linalg import norm


# Function to print the settings
def print_settings():
    print('Using device:', device)
    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print_settings()


@jit(nopython=True, parallel=True)
def cosine_similarity_numba(u: np.ndarray, v: np.ndarray):
    assert (u.shape[0] == v.shape[0])
    uv = 0
    uu = 0
    vv = 0
    for i in prange(u.shape[0]):
        uv += u[i] * v[i]
        uu += u[i] * u[i]
        vv += v[i] * v[i]
    cos_theta = 1
    if uu != 0 and vv != 0:
        cos_theta = uv / np.sqrt(uu * vv)
    return cos_theta


def cosine_similarity_numpy(u: np.ndarray, v: np.ndarray):
    cos_sim = dot(u, v) / (norm(u) * norm(v))
    return cos_sim


class Similarity:

    def __init__(self):
        self.embeddingVectors = {}
        self.models = ['bert-base-nli-stsb-mean-tokens',
                       'bert-large-nli-stsb-mean-tokens',
                       'roberta-base-nli-stsb-mean-tokens',
                       'roberta-large-nli-stsb-mean-tokens',
                       'distilbert-base-nli-stsb-mean-tokens',
                       'bert-base-nli-mean-tokens',
                       'bert-large-nli-mean-tokens',
                       'roberta-base-nli-mean-tokens',
                       'roberta-large-nli-mean-tokens',
                       'distilbert-base-nli-mean-tokens'
                       ]
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
        model = SentenceTransformer(modelName, device=device)

        for index, attr in enumerate(attributes):
            if index != 0 and index % 1000 == 0:
                print("index", index)
                print_settings()
            # print(modelName, ' line i: ', i)
            formatedAttr = copy.deepcopy(attr)

            # embedding1 = model.encode(attr['nodeLeft']['name'], convert_to_tensor=True)
            embedding1 = model.encode(attr['nodeLeft']['name'])
            # embedding2 = model.encode(attr['nodeRight']['name'], convert_to_tensor=True)
            embedding2 = model.encode(attr['nodeRight']['name'])

            # similarity = util.pytorch_cos_sim(embedding1, embedding2)
            similarity = cosine_similarity_numpy(embedding1, embedding2)

            formatedAttr['relationshipList'].append(
                {'type': '', 'method': modelName, 'comments': None, 'confidence': "{:.4f}".format(similarity)})

            attributeSimilarities.append(formatedAttr)

        return attributeSimilarities

    def isDataTypeConvertable(self, type1, type2):

        result = False

        if ((type1 == 'Long' and type2 == 'Double') or (type1 == 'Double' and type2 == 'Long')):
            result = True
        elif ((type1 == 'Date' and type2 == 'DateTime') or (type1 == 'DateTime' and type2 == 'Date')):
            result = True

        return result

    # Does not work because it produces very large list of sentences
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
    #     print(amplified_sentences)
    #     exit()
    #     return amplified_sentences
    # Only suffixes and concepts go into the list and no sentence
    # @staticmethod
    # def createSentenceForAA(node):
    #     amplified_sentences = []
    #     # word_concept_map = {}
    #     # for word in node['suffixArray']:
    #     #     word_concept_map["word"] = []
    #     amplified_sentences.extend(node['suffixArray'])
    #     print(amplified_sentences)
    #     print(node['conceptArray'])
    #     for concept in node['conceptArray']:
    #         amplified_sentences.append(concept['name'])
    #     print(amplified_sentences)
    #     exit()
    #     return amplified_sentences
    # 1 full sentence
    @staticmethod
    def createSentenceForAA(node):
        word_concept_map = {}
        for concept in node['conceptArray']:
            if concept['token'] not in word_concept_map:
                word_concept_map[concept['token']] = []
            word_concept_map[concept['token']].append(concept['name'])
        # print(word_concept_map)
        sentence = ""
        for suffix in word_concept_map:
            sentence += suffix + " " #+ " ".join(word_concept_map[suffix]) + ";"

        return [sentence]

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
        max_similarity = 0.0
        avg_similarity = 0.0
        for term_left in self.embeddingVectors[nodeLeftIdentifier]:
            sentence_left = ""
            if isinstance(term_left, list):
                sentence_left = " ".join(term_left)
            else:
                sentence_left = term_left

            if sentence_left not in sentence_encoding_map:
                # embedding1 = model.encode(term_left, convert_to_tensor=True)
                embedding1 = model.encode(sentence_left)
                sentence_encoding_map[sentence_left] = embedding1
            else:
                embedding1 = sentence_encoding_map[sentence_left]
            # this similarity is between 0 and 1
            for term_right in self.embeddingVectors[nodeRightIdentifier]:
                sentence_right = ""
                if isinstance(term_right, list):
                    sentence_right = " ".join(term_right)
                else:
                    sentence_right = term_right

                if sentence_left == sentence_right:
                    # The two sentences are equal
                    return float(1.0)

                if sentence_right not in sentence_encoding_map:
                    # embedding2 = model.encode(term_right, convert_to_tensor=True)
                    embedding2 = model.encode(sentence_right)
                    sentence_encoding_map[sentence_right] = embedding2
                else:
                    embedding2 = sentence_encoding_map[sentence_right]

                # term_similarity = util.pytorch_cos_sim(embedding1, embedding2)
                # term_similarity1 = cosine_similarity_numba(embedding1, embedding2)
                term_similarity = dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
                # term_similarity = cosine_similarity_numpy(embedding1, embedding2)

                # if float(term_similarity[0][0]) == float(1.0):

                if float(term_similarity) == float(1.0):
                    # The two embeddings are equal but not the sentence, highly unlikely!
                    print("Same embeddings")
                    print(sentence_left)
                    print(sentence_right)
                    return float(1.0)

                if max_similarity == float(0.0) and term_similarity != float(0.0):
                    max_similarity = term_similarity

                if float(term_similarity) > max_similarity:
                    max_similarity = float(term_similarity)

        # calculate the average similarity of the disjoint union of two sets
        # final_similarity = max_left_term_similarity / len(self.embeddingVectors[nodeLeftIdentifier])
        final_similarity = max_similarity
        # print("avg_similarity:",avg_similarity)
        # nodeLeftContext = self.embeddingVectors[nodeLeftIdentifier]
        # nodeRightContext = self.embeddingVectors[nodeRightIdentifier]

        # embedding1 = model.encode(nodeLeftContext, convert_to_tensor=True)
        # embedding2 = model.encode(nodeRightContext, convert_to_tensor=True)
        #
        # similarity = util.pytorch_cos_sim(embedding1, embedding2)
        # similarity = "{:.4f}".format(similarity[0][0])
        # if final_similarity > float(0.0):
        #     print("final_similarity:", final_similarity)
        return final_similarity

    def calculateAmplifiedSimilarity(self, attributes, modelName):

        attributeSimilarities = []

        print('len(attributes)')
        print(len(attributes))

        model = SentenceTransformer(modelName, device=device)

        for index, attrPair in enumerate(attributes):
            if index != 0 and index % 1000 == 0:
                print("index", index)
                print_settings()
            formatedAttr = copy.deepcopy(attrPair)

            syntacticSimilarity = self.getSyntacticSimilarity(attrPair)
            semanticSimilarity = self.getSemanticSimilarity(attrPair, model)
            # similarity = self.getSemanticSimilarity(attrPair, model)

            # print("syntacticSimilarity: ", syntacticSimilarity, ', semanticSimilarity: ', semanticSimilarity)
            # print('Semantic index: ', index)

            # similarity = (0.5 * float(syntacticSimilarity)) + (0.5 * float(semanticSimilarity))

            formatedAttr['relationshipList'].append(
                {'type': '', 'method': modelName + '_SYN_MATCH', 'comments': None, 'confidence': syntacticSimilarity})
            formatedAttr['relationshipList'].append(
                {'type': '', 'method': modelName + '_SEM_MATCH', 'comments': None, 'confidence': semanticSimilarity})
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

            with open("Data/AmplifiedSimilarity-V0.6" + model + ".txt", "wb") as fp:  # Pickling
                pickle.dump(attributes, fp)

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

with open("Data/AmplifiedSimilarity-V0.6.txt", "wb") as fp:  # Pickling
    pickle.dump(similarity, fp)
