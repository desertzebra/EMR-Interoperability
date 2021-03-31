import pickle
import csv


class Results:

    BASE_INDEX = 2
    FUZZY_WUZZY_INDEX = 1
    SYN_AND_SEM_SIM_INDEX = 0

    def readData(self, url):

        data = []

        with open(url, "rb") as fp:
            data = pickle.load(fp)

        return data

    def generateCSV(self, attributes):

        data = []
        data.append(['NodeLeftSchemaName', 'NodeLeftTableName', 'NodeLeftName', 'NodeLeftDataType', 'NodeRightSchemaName', 'NodeRightTableName', 'NodeRightName', 'NodeRightDataType', 'BERT_BASE', 'FUZZY_MATCH', 'SYN_AND_SEM_MATCH'])

        for attrIndex, attr in enumerate(attributes):
            #print("count:", attrIndex)
            data.append(
                [attr['nodeLeft']['schemaName'], attr['nodeLeft']['tableName'], attr['nodeLeft']['name'], attr['nodeLeft']['dataType'], attr['nodeRight']['schemaName'], attr['nodeRight']['tableName'], attr['nodeRight']['name'], attr['nodeRight']['dataType'], attr['relationshipList'][2]['confidence'], attr['relationshipList'][1]['confidence'], attr['relationshipList'][0]['confidence']])

        with open('Results/resultsv0.4.csv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerows(data)

    def printBaseHeatMap(self, attributes, confidenceIndex):
        print('you are here in print Base Heat Map')

        leftNodes = []
        rightNodes = []
        matrix = {}

        for attrIndex, attr in enumerate(attributes):

            print('attrIndex: ', attrIndex)
            #print(attr)
            leftKey = attr['nodeLeft']['schemaName']+"_"+attr['nodeLeft']['tableName']+"_"+attr['nodeLeft']['name']
            rightKey = attr['nodeRight']['schemaName'] + "_" + attr['nodeRight']['tableName'] + "_" + attr['nodeRight'][
                'name']

            # print('leftKey: ', leftKey)
            # print('rightKey: ', rightKey)
            #
            # print('matrix.keys')
            # print(matrix.keys())

            #print("count:", attrIndex)
            if(leftKey not in matrix.keys()):
                matrix[leftKey] = {}
                leftNodes.append(leftKey)
            #if (rightKey not in matrix.keys()):

            if (rightKey not in rightNodes):
                rightNodes.append(rightKey)

            if(confidenceIndex == 1):
                matrix[leftKey][rightKey] = str(float(attr['relationshipList'][confidenceIndex]['confidence'])/ 100)
            else:
                matrix[leftKey][rightKey] = attr['relationshipList'][confidenceIndex]['confidence']


        print("Building the "+str(confidenceIndex)+" table")

        leftNodes.sort()
        rightNodes.sort()

        rowSize = len(leftNodes)
        colSize = len(rightNodes)
        i,j = 0,0
        table = " ,"

        table += ', '.join(rightNodes)
        table += '\r\n'

        while(i<rowSize):
            table += leftNodes[i] + " ,"
            j=0
            while(j<colSize):
                if rightNodes[j] not in matrix[leftNodes[i]].keys():
                    table += "-, "
                    j += 1
                    continue

                #print("leftNodes[i]=", leftNodes[i], ", rightNodes[j]=", rightNodes[j])
                #print("matrix[leftNodes[i]][rightNodes[j]]=", matrix[leftNodes[i]][rightNodes[j]])
                if(leftNodes[i] == rightNodes[j]):
                    table += str(1) + ", "
                else:
                    table += str(matrix[leftNodes[i]][rightNodes[j]]) + ", "
                j += 1
            table += '\r\n'
            i += 1

        print("Done. Now saving it:")
        with open("Data/"+str(confidenceIndex)+"-table-V0.4.csv", "wb") as fp:  # Pickling
            pickle.dump(table, fp)
        #print(table)


        #print(matrix)
        #exit()






resultObj = Results()
# data = simObj.readData('Data/dataV03.json')
data = resultObj.readData('Data/AmplifiedSimilarity-bert-base-nli-mean-tokensV0.2.txt')

print('data')
print(len(data))

#resultObj.generateCSV(data)
resultObj.printBaseHeatMap(data,resultObj.BASE_INDEX)
resultObj.printBaseHeatMap(data,resultObj.FUZZY_WUZZY_INDEX)
resultObj.printBaseHeatMap(data,resultObj.SYN_AND_SEM_SIM_INDEX)
