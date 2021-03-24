import csv
from scipy.stats import pearsonr

class Correlation:

    def readCSV(self, url):
        data= []

        with open(url) as csvFile:

            csvReader = csv.reader(csvFile, delimiter=',')

            i = 0

            for row in csvReader:
                rowData = []
                if i == 0:
                    i += 1
                    continue
                for attr in row :
                    if attr == '-' or attr == '0':
                        rowData.append(0)
                    elif attr == '<' or attr == '>':
                        rowData.append(0.5)
                    elif attr == '1':
                        rowData.append(1)

                data.append(rowData)

        return data

    def getCorrelation(self, list1, list2):

        print('correlation')
        corr, _ = pearsonr(list1, list2)
        corr = round(corr, 2)

        return corr

corrObj = Correlation()

annotator1Data = corrObj.readCSV('Data/Annotated/Annotator1.csv')
print('annotator1 data loaded: ', len(annotator1Data))


annotator2Data = corrObj.readCSV('Data/Annotated/Annotator2.csv')
print('annotator2 data loaded: ', len(annotator2Data))


annotator3Data = corrObj.readCSV('Data/Annotated/Annotator3.csv')
print('annotator3 data loaded: ', len(annotator3Data))

annotator4Data = corrObj.readCSV('Data/Annotated/Annotator4.csv')
print('annotator4 data loaded: ', len(annotator4Data))


#List1 = [1, 2, 3]
#List2 = [1, 2, 3]

corr1vs4 = corrObj.getCorrelation(annotator1Data, annotator4Data)
print('corr between annotator1 and annotator4: ', corr1vs4)