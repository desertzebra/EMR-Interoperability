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
        data.append(
            ['NodeLeftSchemaName', 'NodeLeftTableName', 'NodeLeftName', 'NodeLeftDataType', 'NodeRightSchemaName',
             'NodeRightTableName', 'NodeRightName', 'NodeRightDataType', 'BERT_BASE', 'FUZZY_MATCH',
             'SYN_AND_SEM_MATCH'])
        for attrIndex, attr in enumerate(attributes):
            # print("count:", attrIndex)
            data.append(
                [attr['nodeLeft']['schemaName'], attr['nodeLeft']['tableName'], attr['nodeLeft']['name'],
                 attr['nodeLeft']['dataType'], attr['nodeRight']['schemaName'], attr['nodeRight']['tableName'],
                 attr['nodeRight']['name'], attr['nodeRight']['dataType'], attr['relationshipList'][2]['confidence'],
                 attr['relationshipList'][1]['confidence'], attr['relationshipList'][0]['confidence']])

        with open(
                '/content/drive/MyDrive/papers/EMR-Interoperability/Implementation/Data/ComputedResults/resultsv0.4.csv',
                'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerows(data)

    def printBaseHeatMap(self, attributes, confidenceIndex):
        rowHeaderNodes = []
        columnHeaderNodes = []
        matrix = {}

        for attrIndex, attr in enumerate(attributes):

            # print('attrIndex: ', attrIndex)
            # print(attr)
            leftKey = attr['nodeLeft']['schemaName'] + "_" + attr['nodeLeft']['tableName'] + "_" + attr['nodeLeft'][
                'name']
            rightKey = attr['nodeRight']['schemaName'] + "_" + attr['nodeRight']['tableName'] + "_" + attr['nodeRight'][
                'name']

            # print('leftKey: ', leftKey)
            # print('rightKey: ', rightKey)
            #
            # print('matrix.keys')
            # print(matrix.keys())

            # print("count:", attrIndex)

            # This ensures the rightKey goes into the column head
            if rightKey not in columnHeaderNodes:
                columnHeaderNodes.append(rightKey)
            # Same for left key
            if leftKey not in columnHeaderNodes:
                columnHeaderNodes.append(leftKey)

            if leftKey not in matrix.keys():
                matrix[leftKey] = {}
                rowHeaderNodes.append(leftKey)
            # if (rightKey not in matrix.keys()):

            # Now to look for transposed results
            if rightKey not in matrix.keys():
                matrix[rightKey] = {}
                rowHeaderNodes.append(rightKey)

            if (confidenceIndex == 1):
                matrix[leftKey][rightKey] = str(float(attr['relationshipList'][confidenceIndex]['confidence']) / 100)
                matrix[rightKey][leftKey] = str(float(attr['relationshipList'][confidenceIndex]['confidence']) / 100)
            else:
                matrix[leftKey][rightKey] = attr['relationshipList'][confidenceIndex]['confidence']
                matrix[rightKey][leftKey] = attr['relationshipList'][confidenceIndex]['confidence']

        print("Building the " + str(confidenceIndex) + " table")

        rowHeaderNodes.sort()
        columnHeaderNodes.sort()
        rowSize = len(rowHeaderNodes)
        colSize = len(columnHeaderNodes)
        i, j = 0, 0
        table = "__,"

        table += ', '.join(columnHeaderNodes)
        table += '\r\n'

        while (i < rowSize):
            table += rowHeaderNodes[i] + " ,"
            j = 0
            while (j < colSize):
                # print("rowHeaderNodes[i]=", rowHeaderNodes[i], ", columnHeaderNodes[j]=", columnHeaderNodes[j])
                # print("matrix[rowHeaderNodes[i]][columnHeaderNodes[j]]=", matrix[rowHeaderNodes[i]][columnHeaderNodes[j]])
                leftSchema = (rowHeaderNodes[i].split('_')[0])
                leftSchema = leftSchema.strip()
                rightSchema = (columnHeaderNodes[j].split('_')[0])
                rightSchema = rightSchema.strip()

                if leftSchema == rightSchema:
                    table += "-" + ","
                elif columnHeaderNodes[j] not in matrix[rowHeaderNodes[i]].keys():
                    print("rowHeaderNodes[i]", rowHeaderNodes[i], "columnHeaderNodes[j]", columnHeaderNodes[j])
                    table += "-, "
                else:
                    # print("leftSchema=", leftSchema, ", rightSchema=", rightSchema, end=';')
                    table += str(matrix[rowHeaderNodes[i]][columnHeaderNodes[j]]) + ", "
                j += 1
            # remove last ,
            table = table[:-1] + '\r\n'
            i += 1
        print()
        print("Done. Now saving it:")
        with open("Data/" + str(confidenceIndex) + "-table-V0.4.csv", "wb") as fp:  # Pickling
            pickle.dump(table, fp)
        # print(table)

        # print(matrix)
        # exit()


resultObj = Results()
# data = simObj.readData('Data/dataV03.json')
data = resultObj.readData('Data/AmplifiedSimilarity-bert-base-nli-mean-tokensV0.2.txt')

print('data')
print(len(data))

#resultObj.generateCSV(data)
resultObj.printBaseHeatMap(data,resultObj.BASE_INDEX)
resultObj.printBaseHeatMap(data,resultObj.FUZZY_WUZZY_INDEX)
resultObj.printBaseHeatMap(data,resultObj.SYN_AND_SEM_SIM_INDEX)
