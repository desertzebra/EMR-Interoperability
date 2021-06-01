import csv
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score
import numpy as np

class Correlation:

    def readCSV(self, url):
        data= []

        with open(url) as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',')
            rowIterator = 0
            for row in csvReader:
                rowIterator += 1
                if rowIterator==1:
                    continue
                colIterator = 0
                rowData = []
                for attr in row:
                    attr = attr.strip()
                    colIterator += 1
                    if colIterator == 1:
                        continue

                    if attr == '-' or attr == '~':
                        rowData.append("-1.0")
                    elif attr == '' and len(rowData) < 260:
                        rowData.append("-1.0")
                    elif attr == '0':
                        rowData.append("0.0")
                    elif attr == '<':
                        rowData.append("0.5")
                    elif attr == '1':
                        rowData.append("1.0")
                    elif  attr == '>':
                        rowData.append("1.5")
                    elif attr == '':
                        print("rowIterator:",rowIterator,",colIterator:",colIterator, ",attr:",attr)
                        #colIterator = colIterator-1
                        continue
                    # else:
                    #     print("colIterator:",colIterator, ",attr:",attr)
                if 0 < len(rowData) < 259:
                    print("Row Iterator:", rowIterator)
                    print("Column size:", colIterator)
                    print("Columns in the original data row:", len(row))
                    print("problematic row:", row)
                    print("Columns in the rowData:", len(rowData))
                    print("rowData:", rowData)

                if len(rowData)>0:
                    data.append(rowData)
        print("data:", data)
        return data

    def getCorrelation(self, list1, list2):
        correlationBetweenLists = []
        for list1_attr, list2_attr in zip (list1,list2):
            (correlation, p_value) = pearsonr(list1_attr, list2_attr)
            formattedCorrelationValue = str(round(correlation, 2))
            correlationBetweenLists.append(formattedCorrelationValue)

            # correlation = round(correlation, 2)
            # print('%7.4f %7.4f' % (correlation, p_value))
        return correlationBetweenLists

    def getKappaCorrelation(self, list1, list2):
        kappaCorrelationBetweenLists = []
        for list1_attr, list2_attr in zip (list1,list2):
            correlation = cohen_kappa_score(list1_attr, list2_attr)
            formattedCorrelationValue = str(round(correlation, 2))
            kappaCorrelationBetweenLists.append(formattedCorrelationValue)

            # correlation = round(correlation, 2)
            # print('%7.4f %7.4f' % (correlation, p_value))
        return kappaCorrelationBetweenLists

corrObj = Correlation()

# print("*****************************************Annotator 1*****************************************")
annotator1Data = corrObj.readCSV('Data/Annotated/Annotator1.csv')
print('annotator1 data loaded: ', len(annotator1Data))
# print("**********************************************************************************")
# print("*****************************************Annotator 2*****************************************")
annotator2Data = corrObj.readCSV('Data/Annotated/Annotator2.csv')
print('annotator2 data loaded: ', len(annotator2Data))
# print("**********************************************************************************")
# print("*****************************************Annotator 3*****************************************")
annotator3Data = corrObj.readCSV('Data/Annotated/Annotator3.csv')
print('annotator3 data loaded: ', len(annotator3Data))
# print("**********************************************************************************")
# print("*****************************************Annotator 4*****************************************")
annotator4Data = corrObj.readCSV('Data/Annotated/Annotator4.csv')
print('annotator4 data loaded: ', len(annotator4Data))
print("**********************************************************************************")


# allCorrelations = ""
# #List1 = [1, 2, 3]
# #List2 = [1, 2, 3]
# #
# # print(annotator1Data)
# # print(annotator4Data)
# # print('Pearson Correlation between annotator1 and annotator2: ')
# allCorrelations += "annotator1 vs annotator2,"+",".join(corrObj.getCorrelation(annotator1Data, annotator2Data))+"\r\n"
# # print("**********************************************************************************")
#
# # print('Pearson Correlation between annotator1 and annotator3: ')
# allCorrelations += "annotator1 vs annotator3,"+",".join(corrObj.getCorrelation(annotator1Data, annotator3Data))+"\r\n"
# # print("**********************************************************************************")
#
# # print('Pearson Correlation between annotator1 and annotator4: ')
# allCorrelations += "annotator1 vs annotator4,"+ ",".join(corrObj.getCorrelation(annotator1Data, annotator4Data))+"\r\n"
# # print("**********************************************************************************")
#
# # print('Pearson Correlation between annotator2 and annotator3: ')
# allCorrelations += "annotator2 vs annotator3,"+ ",".join(corrObj.getCorrelation(annotator2Data, annotator3Data))+"\r\n"
# # print("**********************************************************************************")
#
# # print('Pearson Correlation between annotator2 and annotator4: ')
# allCorrelations += "annotator2 vs annotator4,"+ ",".join(corrObj.getCorrelation(annotator2Data, annotator4Data))+"\r\n"
# # print("**********************************************************************************")
#
# # print('Pearson Correlation between annotator3 and annotator4: ')
# allCorrelations += "annotator3 vs annotator4,"+ ",".join(corrObj.getCorrelation(annotator3Data, annotator4Data))+"\r\n"
# print("**********************************************************************************")
# print(allCorrelations)
# print("**********************************************************************************")

allKappaCorrelations = "\r\n"

print('Cohen kappa score  between annotator1 and annotator2: ')
allKappaCorrelations += "annotator1 vs annotator2,"+",".join(corrObj.getKappaCorrelation(annotator1Data, annotator2Data))+"\r\n"
print("**********************************************************************************")


print('Cohen kappa score  between annotator1 and annotator3: ')
allKappaCorrelations += "annotator1 vs annotator3,"+",".join(corrObj.getKappaCorrelation(annotator1Data, annotator3Data))+"\r\n"
print("**********************************************************************************")

print('Cohen kappa score  between annotator1 and annotator4: ')
allKappaCorrelations += "annotator1 vs annotator4,"+ ",".join(corrObj.getKappaCorrelation(annotator1Data, annotator4Data))+"\r\n"
print("**********************************************************************************")

print('Cohen kappa score  between annotator2 and annotator3: ')
allKappaCorrelations += "annotator2 vs annotator3,"+ ",".join(corrObj.getKappaCorrelation(annotator2Data, annotator3Data))+"\r\n"
print("**********************************************************************************")

print('Cohen kappa score  between annotator2 and annotator4: ')
allKappaCorrelations += "annotator2 vs annotator4,"+ ",".join(corrObj.getKappaCorrelation(annotator2Data, annotator4Data))+"\r\n"
print("**********************************************************************************")

print('Cohen kappa score  between annotator3 and annotator4: ')
allKappaCorrelations += "annotator3 vs annotator4,"+ ",".join(corrObj.getKappaCorrelation(annotator3Data, annotator4Data))+"\r\n"
print("**********************************************************************************")
print(allKappaCorrelations)
#
#
# print('corr between annotator1 and annotator4: ')
# corrObj.getCorrelation(annotator1Data, annotator4Data)
# print("**********************************************************************************")
# print('corr between annotator2 and annotator3: ')
# corrObj.getCorrelation(annotator2Data, annotator3Data)
# print("**********************************************************************************")
# print('corr between annotator1 and annotator4: ')
# corrObj.getCorrelation(annotator2Data, annotator4Data)
# print("**********************************************************************************")
# print('corr between annotator3 and annotator4: ')
# corrObj.getCorrelation(annotator3Data, annotator4Data)
# print("**********************************************************************************")