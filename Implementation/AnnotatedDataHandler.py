import csv
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report, multilabel_confusion_matrix, confusion_matrix
import numpy as np
#import matplotlib.pyplot as plt
import re
import pandas as pd
from statistics import mode, StatisticsError


class AnnotatedDataHandler:
    annotator1Data = []
    annotator2Data = []
    annotator3Data = []
    annotator4Data = []
    method1Data = []
    method2Data = []
    method3Data = []

    logLevel = ["DEBUG", "INFO"]  # ["TRACE", "DEBUG", "INFO"]       # Leave out only those levels, which should fire

    def __init__(self):
        pass

    def initDataStructures(self):
        self.annotator1Data = []
        self.annotator2Data = []
        self.annotator3Data = []
        self.annotator4Data = []
        self.method1Data = []
        self.method2Data = []
        self.method3Data = []

    def log(self, msg, log_at="INFO"):
        if log_at in self.logLevel:
            if isinstance(msg, list):
                print(' '.join(str(v) for v in msg))
            else:
                print(msg)

    def readCSV(self, url, isComputed=False, thresholds={}):
        data = []
        with open(url) as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',')
            for row in csvReader:
                cellData = []
                for attr in row:
                    if not isComputed:
                        attr = self.convertAnnotatedAttrValue(attr)
                    else:
                        attr = self.convertComputedAttrValue(attr, thresholds)
                    cellData.append(attr)
                if len(cellData) > 0:
                    data.append(cellData)
        return data

    def readCSVWithoutHeaders(self, url, isComputed=False, thresholds={}):
        data = []
        with open(url) as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',')
            rowIterator = 0
            for row in csvReader:
                rowIterator += 1
                if rowIterator == 1:
                    continue
                colIterator = 0
                cellData = []
                for attr in row:
                    colIterator += 1
                    if colIterator == 1:
                        continue

                    # Convert the attribute values, according to thresholds
                    if not isComputed:
                        attr = self.convertAnnotatedAttrValue(attr)
                    else:
                        attr = self.convertComputedAttrValue(attr, thresholds)

                    if attr == '':
                        self.log(["rowIterator:", rowIterator, ",colIterator:", colIterator, ",attr:", attr], "DEBUG")
                        # colIterator = colIterator-1
                        continue
                    else:
                        cellData.append(attr)
                    # else:
                    #     log(["colIterator:",colIterator, ",attr:",attr], "DEBUG")
                if 0 < len(cellData) < 259:
                    self.log(["Row Iterator:", rowIterator], "DEBUG")
                    self.log(["Column size:", colIterator], "DEBUG")
                    self.log(["Columns in the original data row:", len(row)], "DEBUG")
                    self.log(["problematic row:", row], "DEBUG")
                    self.log(["Columns in the cellData:", len(cellData)], "DEBUG")
                    self.log(["cellData:", cellData], "DEBUG")

                if len(cellData) > 0:
                    data.append(cellData)
        return data

    def getPearsonCorrelationScore(self, list1, list2):
        correlationBetweenLists = []
        for list1_attr, list2_attr in zip(list1, list2):
            (correlation_value, p_value) = pearsonr(list(float(v) for v in list1_attr),
                                                    list(float(v) for v in list2_attr))
            formatted_correlation_value = str(round(correlation_value, 2))
            correlationBetweenLists.append(formatted_correlation_value)
            # AnnotatedDataHandler = round(AnnotatedDataHandler, 2)
            # print('%7.4f %7.4f' % (AnnotatedDataHandler, p_value))
        return correlationBetweenLists

    def getKappaCorrelationScore(self, list1, list2):
        kappaCorrelationBetweenLists = []
        for list1_attr, list2_attr in zip(list1, list2):
            # self.log(len(list1_attr))
            # self.log(len(list2_attr))
            d_score = cohen_kappa_score(list(str(v) for v in list1_attr), list(str(v) for v in list2_attr))
            formatted_d_score = str(round(d_score, 2))
            kappaCorrelationBetweenLists.append(formatted_d_score)
            # AnnotatedDataHandler = round(AnnotatedDataHandler, 2)
            # print('%7.4f %7.4f' % (AnnotatedDataHandler, p_value))
        return kappaCorrelationBetweenLists

    def getDefaultThresholdValues(self, thresholdValues={}):

        # initializing default threshold values. Perhaps this should be in a separate function?
        if "0.0" not in thresholdValues or thresholdValues["0.0"] == "undefined":
            print("setting default value for 0.0", thresholdValues)
            thresholdValues["0"] = 0.6
        if "0.5" not in thresholdValues or thresholdValues["0.5"] == "undefined":
            print("setting default value for 0.5", thresholdValues)
            thresholdValues["0.5"] = 0.8
        return thresholdValues

    def convertComputedAttrValue(self, attr, thresholdValues={}):
        attr = attr.strip()
        thresholds = self.getDefaultThresholdValues(thresholdValues)

        if attr == '-1' or attr == '-' or attr == '':
            return "-"
        # elif attr == '~' or attr == '':
        #     return "0.0"
        elif self.isFloat(attr) and 0 <= float(attr) < thresholds["0.0"]:
            return "0.0"
        elif self.isFloat(attr) and thresholds["0.0"] <= float(attr) < thresholds["0.5"]:
            return "0.5"
        elif self.isFloat(attr) and thresholds["0.5"] <= float(attr):
            return "1.0"

        # check if some float value was missed
        if self.isFloat(attr):
            self.log([attr], "DEBUG")

        return attr

    @staticmethod
    def convertAnnotatedAttrValue(attr):
        attr = attr.strip()
        if attr == '-':
            attr = "-"
        elif attr == '~' or attr =='':
            attr = "-1.0"
        elif attr == '0':
            attr = "0.0"
        elif attr == '<':
            attr = "0.5"
        elif attr == '1':
            attr = "1.0"
        elif attr == '>':
            attr = "1.5"

        return attr

    @staticmethod
    def isFloat(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def readAllAnnotatorsData(self, readHeaders=False):
        # log("*****************************************Annotator 1*****************************************")
        if not readHeaders:
            self.annotator1Data = annotatedDataHandler.readCSVWithoutHeaders(
                'Data/Annotated/updated/Annotator1.csv')
        else:
            self.annotator1Data = annotatedDataHandler.readCSV(
                'Data/Annotated/updated/Annotator1.csv')
        self.log(["annotator1 data loaded: ", len(self.annotator1Data), " with readHeaders=", readHeaders])
        # log(["*"]*80)
        # log("*****************************************Annotator 2*****************************************")
        if not readHeaders:
            self.annotator2Data = annotatedDataHandler.readCSVWithoutHeaders(
                'Data/Annotated/updated/Annotator2.csv')
        else:
            self.annotator2Data = annotatedDataHandler.readCSV(
                'Data/Annotated/updated/Annotator2.csv')
        self.log(["annotator2 data loaded: ", len(self.annotator2Data), " with readHeaders=", readHeaders])
        # log(["*"]*80)
        # log("*****************************************Annotator 3*****************************************")
        if not readHeaders:
            self.annotator3Data = annotatedDataHandler.readCSVWithoutHeaders(
                'Data/Annotated/updated/Annotator3.csv')
        else:
            self.annotator3Data = annotatedDataHandler.readCSV(
                'Data/Annotated/updated/Annotator3.csv')
        self.log(["annotator3 data loaded: ", len(self.annotator3Data), " with readHeaders=", readHeaders])
        # log(["*"]*80)
        # log("*****************************************Annotator 4*****************************************")
        if not readHeaders:
            self.annotator4Data = annotatedDataHandler.readCSVWithoutHeaders(
                'Data/Annotated/updated/Annotator4.csv')
        else:
            self.annotator4Data = annotatedDataHandler.readCSV(
                'Data/Annotated/updated/Annotator4.csv')
        self.log(["annotator4 data loaded: ", len(self.annotator4Data), " with readHeaders=", readHeaders])

        self.log(["*"] * 80)

    def readMethod1ComputedData(self, readHeaders=False, thresholdValues={}):
        if not readHeaders:
            self.method1Data = annotatedDataHandler.readCSVWithoutHeaders(
                'Data/0-table-V0.5.csv', True, thresholdValues)
        else:
            self.method1Data = annotatedDataHandler.readCSV(
                'Data/0-table-V0.5.csv', True, thresholdValues)
        self.log(["method1 data loaded: ", len(self.method1Data), " with readHeaders=", readHeaders])

    def readMethod2ComputedData(self, readHeaders=False, thresholdValues={}):
        if not readHeaders:
            self.method2Data = annotatedDataHandler.readCSVWithoutHeaders(
                'Data/1-table-V0.5.csv', True,
                thresholdValues)
        else:
            self.method2Data = annotatedDataHandler.readCSV(
                'Data/1-table-V0.5.csv', True,
                thresholdValues)
        self.log(["method2 data loaded: ", len(self.method2Data), " with readHeaders=", readHeaders])

    def readMethod3ComputedData(self, readHeaders=False, thresholdValues={}):
        if not readHeaders:
            self.method3Data = annotatedDataHandler.readCSVWithoutHeaders(
                'Data/2-table-V0.5.csv', True, thresholdValues)
        else:
            self.method3Data = annotatedDataHandler.readCSV(
                'Data/2-table-V0.5.csv', True, thresholdValues)
        self.log(["method3 data loaded: ", len(self.method3Data), " with readHeaders=", readHeaders])

    def readAllComputedData(self, readHeaders=False, thresholdValues={}):
        self.readMethod1ComputedData(readHeaders, thresholdValues)
        # self.log(["*"] * 80)
        self.readMethod2ComputedData(readHeaders, thresholdValues)
        # self.log(["*"] * 80)
        self.readMethod3ComputedData(readHeaders, thresholdValues)

        # self.log(["*"] * 80)

    def calculatePearsonScoreBetweenAnnotators(self):
        allAnnotatedDataHandlers = ""
        if len(self.annotator1Data) < 1 or len(self.annotator2Data) < 1 or len(self.annotator3Data) < 1 or len(
                self.annotator4Data) < 1:
            self.log("Insufficient data for the annotators, have you read the files yet?")
            return

        # log('Pearson AnnotatedDataHandler between annotator1 and annotator2: ')
        allAnnotatedDataHandlers += "annotator1 vs annotator2," + ",".join(
            annotatedDataHandler.getPearsonCorrelationScore(self.annotator1Data, self.annotator2Data)) + "\r\n"
        # log(["*"]*80)

        # log('Pearson AnnotatedDataHandler between annotator1 and annotator3: ')
        allAnnotatedDataHandlers += "annotator1 vs annotator3," + ",".join(
            annotatedDataHandler.getPearsonCorrelationScore(self.annotator1Data, self.annotator3Data)) + "\r\n"
        # log(["*"]*80)

        # log('Pearson AnnotatedDataHandler between annotator1 and annotator4: ')
        allAnnotatedDataHandlers += "annotator1 vs annotator4," + ",".join(
            annotatedDataHandler.getPearsonCorrelationScore(self.annotator1Data, self.annotator4Data)) + "\r\n"
        # log(["*"]*80)

        # log('Pearson AnnotatedDataHandler between annotator2 and annotator3: ')
        allAnnotatedDataHandlers += "annotator2 vs annotator3," + ",".join(
            annotatedDataHandler.getPearsonCorrelationScore(self.annotator2Data, self.annotator3Data)) + "\r\n"
        # log(["*"]*80)

        # log('Pearson AnnotatedDataHandler between annotator2 and annotator4: ')
        allAnnotatedDataHandlers += "annotator2 vs annotator4," + ",".join(
            annotatedDataHandler.getPearsonCorrelationScore(self.annotator2Data, self.annotator4Data)) + "\r\n"
        # log(["*"]*80)

        # log('Pearson AnnotatedDataHandler between annotator3 and annotator4: ')
        allAnnotatedDataHandlers += "annotator3 vs annotator4," + ",".join(
            annotatedDataHandler.getPearsonCorrelationScore(self.annotator3Data, self.annotator4Data)) + "\r\n"
        self.log(["*"] * 80)
        print(allAnnotatedDataHandlers)
        self.log(["*"] * 80)

    def calculateKappaScoreBetweenAnnotators(self):
        if len(self.annotator1Data) < 1 or len(self.annotator2Data) < 1 or len(self.annotator3Data) < 1 or len(
                self.annotator4Data) < 1:
            self.log("Insufficient data for the annotators, have you read the files yet?")
            return
        allKappaAnnotatedDataHandlers = "\r\n"

        self.log('Cohen kappa score  between annotator1 and annotator2: ')
        allKappaAnnotatedDataHandlers += "annotator1 vs annotator2," + ",".join(
            annotatedDataHandler.getKappaCorrelationScore(self.annotator1Data, self.annotator2Data)) + "\r\n"
        self.log(["*"] * 80)

        self.log('Cohen kappa score  between annotator1 and annotator3: ')
        allKappaAnnotatedDataHandlers += "annotator1 vs annotator3," + ",".join(
            annotatedDataHandler.getKappaCorrelationScore(self.annotator1Data, self.annotator3Data)) + "\r\n"
        self.log(["*"] * 80)

        self.log('Cohen kappa score  between annotator1 and annotator4: ')
        allKappaAnnotatedDataHandlers += "annotator1 vs annotator4," + ",".join(
            annotatedDataHandler.getKappaCorrelationScore(self.annotator1Data, self.annotator4Data)) + "\r\n"
        self.log(["*"] * 80)

        self.log('Cohen kappa score  between annotator2 and annotator3: ')
        allKappaAnnotatedDataHandlers += "annotator2 vs annotator3," + ",".join(
            annotatedDataHandler.getKappaCorrelationScore(self.annotator2Data, self.annotator3Data)) + "\r\n"
        self.log(["*"] * 80)

        self.log('Cohen kappa score  between annotator2 and annotator4: ')
        allKappaAnnotatedDataHandlers += "annotator2 vs annotator4," + ",".join(
            annotatedDataHandler.getKappaCorrelationScore(self.annotator2Data, self.annotator4Data)) + "\r\n"
        annotatedDataHandler.log(["*"] * 80)

        self.log('Cohen kappa score  between annotator3 and annotator4: ')
        allKappaAnnotatedDataHandlers += "annotator3 vs annotator4," + ",".join(
            annotatedDataHandler.getKappaCorrelationScore(self.annotator3Data, self.annotator4Data)) + "\r\n"
        annotatedDataHandler.log(["*"] * 80)
        self.log(allKappaAnnotatedDataHandlers)

    def calculateKappaScoreBetweenComputedAndAnnotatedData(self, avgAnnotatedData=None):
        if avgAnnotatedData == None:
            if len(self.annotator1Data) < 1 or len(self.annotator2Data) < 1 or len(self.annotator3Data) < 1 or len(
                    self.annotator4Data) < 1:
                self.log("Insufficient data for the annotators, have you read the files yet?")
                return
            else:
                avgAnnotatedData = annotatedDataHandler.calculateModeScoreBetweenAllAnnotators()

        if len(self.method1Data) < 1 or len(self.method1Data) < 1 or len(self.method1Data) < 1:
            self.log("Insufficient data for the computed methods, have you read the files yet?")
            return

        resultAsCsvString = "\r\n"

        self.log('Cohen kappa score  between method1 and avg(annotators): ')
        resultAsCsvString += "method1 vs avg(annotators)," + ",".join(
            annotatedDataHandler.getKappaCorrelationScore(self.method1Data, avgAnnotatedData)) + "\r\n"
        self.log(["*"] * 80)

        self.log('Cohen kappa score  between method2 and avg(annotators): ')
        resultAsCsvString += "method2 vs avg(annotators)," + ",".join(
            annotatedDataHandler.getKappaCorrelationScore(self.method2Data, avgAnnotatedData)) + "\r\n"

        self.log(["*"] * 80)
        self.log('Cohen kappa score  between method3 and avg(annotators): ')

        resultAsCsvString += "method3 vs avg(annotators)," + ",".join(
            annotatedDataHandler.getKappaCorrelationScore(self.method3Data, avgAnnotatedData)) + "\r\n"
        self.log(["*"] * 80)

        self.log(["*"] * 80)
        self.log(resultAsCsvString)

    def calculateModeScoreBetweenAllAnnotators(self, hasHeaders=False):
        if len(self.annotator1Data) < 1 or len(self.annotator2Data) < 1 or len(self.annotator3Data) < 1 \
                or len(self.annotator4Data) < 1:
            self.log("Insufficient data for the annotators, have you read the files yet?")
            return
        averagedData = []
        rowIterator = 0
        colHeadNameList = []
        for (a1Row, a2Row, a3Row, a4Row) in zip(self.annotator1Data, self.annotator2Data, self.annotator3Data,
                                                self.annotator4Data):
            rowIterator += 1
            if hasHeaders and rowIterator == 1:
                averagedData.append(a1Row)
                continue
            colIterator = 0
            cellData = []

            for (a1Col, a2Col, a3Col, a4Col) in zip(a1Row, a2Row, a3Row, a4Row):
                colIterator += 1
                self.log(a1Col, "TRACE")
                if hasHeaders and colIterator == 1:
                    cellData.append(a1Col)
                    # print(rowHeaderName)
                elif not self.isFloat(a1Col) or not self.isFloat(a2Col) or not self.isFloat(a3Col) or not self.isFloat(
                        a4Col):
                    if a1Col == a2Col == a3Col == a4Col:
                        cellData.append(a1Col)
                    else:
                        self.log("Major ERROR in code")
                        self.log([a1Row, a2Row, a3Row, a4Row])
                        self.log(["colIterator:",colIterator])
                        self.log([a1Col, a2Col, a3Col, a4Col])
                        exit()
                else:
                    try:
                        mode_val = mode([a1Col, a2Col, a3Col, a4Col])
                    except StatisticsError:
                        self.log([a1Col, a2Col, a3Col, a4Col])
                        self.log("Major error: No mode found")
                        exit()
                    cellData.append(str(mode_val))
            averagedData.append(cellData)
        return averagedData

    def collapseDataSetTo1d(self, list2dWithHeaders):
        list1d = []
        rowIterator = 0
        colHeadNameList = []
        for row in list2dWithHeaders:
            rowIterator += 1
            # if rowIterator == 1:
            #     for attr in row:
            #         colHeadNameList.append(attr)
            # Even java has labeled loops. This sucks!
            if rowIterator == 1:
                continue
            # print("colHeadNameList", colHeadNameList)
            colIterator = 0
            # rowHeaderName = ""
            for attr in row:
                colIterator += 1
                if colIterator == 1:
                    # rowHeaderName = attr
                    # print(rowHeaderName)
                    continue
                # colHeaderName = colHeadNameList[colIterator - 1]
                # attr = self.convertAttrValue(attr)
                if not attr == "-":
                    list1d.append(attr)

        return list1d

    def collapseDataSetTo1dArrayWithHeaders(self, list2dWithHeaders):
        list1d = []
        rowIterator = 0
        colHeadNameList = []
        for row in list2dWithHeaders:
            rowIterator += 1
            if rowIterator == 1:
                for attr in row:
                    colHeadNameList.append(attr)

            # Even java has labeled loops. This sucks!
            if rowIterator == 1:
                print(len(colHeadNameList))
                continue
            # print("colHeadNameList", colHeadNameList)
            colIterator = 0
            rowHeaderName = ""
            for attr in row:
                colIterator += 1
                if colIterator == 1:
                    rowHeaderName = attr
                    # print(rowHeaderName)
                    continue
                try:
                    colHeaderName = colHeadNameList[colIterator - 1]
                except:
                    print(row)
                    print("colIterator:", colIterator, ", size of colHeadNameList:", len(colHeadNameList))
                    exit()
                # attr = self.convertAttrValue(attr)
                list1d.append([rowHeaderName, colHeaderName, attr])
        return list1d

    def compare1dLists(self, list2dWithHeader1, list2dWithHeader2):
        self.log(["len(list2dWithHeader1):", len(list2dWithHeader1)])
        self.log(["len(list2dWithHeader2):", len(list2dWithHeader2)])
        countProblematicRows = 0

        for (row1, row2) in zip(list2dWithHeader1, list2dWithHeader2):
            for (a1Col, a2Col) in zip(row1, row2):
                if not self.isFloat(a1Col) and not self.isFloat(a2Col) and not a1Col == a2Col:
                    countProblematicRows += 1
                    self.log(["row1:", row1])
                    self.log(["row2:", row2])
                    self.log(["a1Col:", a1Col, ",a2Col:", a2Col])
                    exit()
        self.log(["countProblematicRows:", countProblematicRows])


annotatedDataHandler = AnnotatedDataHandler()
# annotatedDataHandler.readAllAnnotatorsData(False)
# annotatedDataHandler.readAllComputedData(False)
# # annotatedDataHandler.calculateKappaScoreBetweenAnnotators()
# modeAnnotatedData = annotatedDataHandler.calculateModeScoreBetweenAllAnnotators()

# annotatedDataHandler.log(modeAnnotatedData, "DEBUG")
# annotatedDataHandler.calculateKappaScoreBetweenComputedAndAnnotatedData(modeAnnotatedData)

# Now to convert the datasets into 1d

annotatedDataHandler.initDataStructures()
annotatedDataHandler.readAllAnnotatorsData(True)
modeAnnotatedData = annotatedDataHandler.calculateModeScoreBetweenAllAnnotators(True)
flatAnnotatedData = annotatedDataHandler.collapseDataSetTo1d(modeAnnotatedData)

# annotatedDataHandler.readAllComputedData(True)


# print(modeAnnotatedData)
#
# flatAnnotatedData1 = annotatedDataHandler.collapseDataSetTo1d(annotatedDataHandler.annotator1Data)
# flatAnnotatedData2 = annotatedDataHandler.collapseDataSetTo1d(annotatedDataHandler.annotator2Data)
# flatAnnotatedData3 = annotatedDataHandler.collapseDataSetTo1d(annotatedDataHandler.annotator3Data)
# flatAnnotatedData4 = annotatedDataHandler.collapseDataSetTo1d(annotatedDataHandler.annotator4Data)
# flatMethod1Data = annotatedDataHandler.collapseDataSetTo1d(annotatedDataHandler.method1Data)
# flatMethod2Data = annotatedDataHandler.collapseDataSetTo1d(annotatedDataHandler.method2Data)
# flatMethod3Data = annotatedDataHandler.collapseDataSetTo1d(annotatedDataHandler.method3Data)

# annotatedDataHandler.log(["-"]*80)
# flatAnnotator1DataWithHeaders = annotatedDataHandler.collapseDataSetTo1dArrayWithHeaders(
#     annotatedDataHandler.annotator4Data)
# annotatedDataHandler.log(["-"]*80)
# flatMethod2DataWithHeaders = annotatedDataHandler.collapseDataSetTo1dArrayWithHeaders(annotatedDataHandler.method3Data)
# annotatedDataHandler.log(["-"]*80)
# annotatedDataHandler.compare1dLists(flatMethod2DataWithHeaders, flatAnnotator1DataWithHeaders)

# annotatedDataHandler.log("Annotated Data 1 vs Method 1")
# data = {'y_Actual':    flatAnnotatedData1,#["-1","0","0.5","1","1.5"],
#         'y_Predicted': flatMethod1Data#[1,0.9,0.6,0.7,0.1]
#         }
#
# df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
#
# confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
# print (confusion_matrix)
#
# annotatedDataHandler.log(["*"]*80)
# annotatedDataHandler.log("Annotated Data 1 vs Method 2")
# data = {'y_Actual':    flatAnnotatedData1,#["-1","0","0.5","1","1.5"],
#         'y_Predicted': flatMethod2Data#[1,0.9,0.6,0.7,0.1]
#         }
#
# df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
#
# confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
#
# print (np.diag(confusion_matrix).sum() / confusion_matrix.to_numpy().sum())
#
# annotatedDataHandler.log(["*"]*80)
# annotatedDataHandler.log("Annotated Data 1 vs Method 3")
# data = {'y_Actual':    flatAnnotatedData1,#["-1","0","0.5","1","1.5"],
#         'y_Predicted': flatMethod3Data#[1,0.9,0.6,0.7,0.1]
#         }
#
# df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
#
# confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
# print (confusion_matrix)
#
# annotatedDataHandler.log(["*"]*80)
#
# annotatedDataHandler.log("Annotated Data 2 vs Method 1")
# data = {'y_Actual':    flatAnnotatedData2,#["-1","0","0.5","1","1.5"],
#         'y_Predicted': flatMethod1Data#[1,0.9,0.6,0.7,0.1]
#         }
#
# df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
#
# confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
# print (confusion_matrix)
#
# annotatedDataHandler.log(["*"]*80)
# annotatedDataHandler.log("Annotated Data 2 vs Method 2")
# data = {'y_Actual':    flatAnnotatedData2,#["-1","0","0.5","1","1.5"],
#         'y_Predicted': flatMethod2Data#[1,0.9,0.6,0.7,0.1]
#         }
#
# df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
#
# confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
# print (confusion_matrix)
#
# annotatedDataHandler.log(["*"]*80)
# annotatedDataHandler.log("Annotated Data 2 vs Method 3")
# data = {'y_Actual':    flatAnnotatedData2,#["-1","0","0.5","1","1.5"],
#         'y_Predicted': flatMethod3Data#[1,0.9,0.6,0.7,0.1]
#         }
#
# df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
#
# confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
# print (confusion_matrix)
#
# annotatedDataHandler.log(["*"]*80)
#
# annotatedDataHandler.log("Annotated Data 3 vs Method 1")
# data = {'y_Actual':    flatAnnotatedData3,#["-1","0","0.5","1","1.5"],
#         'y_Predicted': flatMethod1Data#[1,0.9,0.6,0.7,0.1]
#         }
#
# df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
#
# confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
# print (confusion_matrix)
#
# annotatedDataHandler.log(["*"]*80)
# annotatedDataHandler.log("Annotated Data 3 vs Method 2")
# data = {'y_Actual':    flatAnnotatedData3,#["-1","0","0.5","1","1.5"],
#         'y_Predicted': flatMethod2Data#[1,0.9,0.6,0.7,0.1]
#         }
#
# df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
#
# confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
# print (confusion_matrix)
#
# annotatedDataHandler.log(["*"]*80)
# annotatedDataHandler.log("Annotated Data 3 vs Method 3")
# data = {'y_Actual':    flatAnnotatedData3,#["-1","0","0.5","1","1.5"],
#         'y_Predicted': flatMethod3Data#[1,0.9,0.6,0.7,0.1]
#         }
#
# df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
#
# confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
# print (confusion_matrix)
#
# annotatedDataHandler.log(["*"]*80)
#
#
# annotatedDataHandler.log("Annotated Data 4 vs Method 1")
# data = {'y_Actual':    flatAnnotatedData4,#["-1","0","0.5","1","1.5"],
#         'y_Predicted': flatMethod1Data#[1,0.9,0.6,0.7,0.1]
#         }
#
# df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
#
# confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
# print (confusion_matrix)
#
# annotatedDataHandler.log(["*"]*80)
# annotatedDataHandler.log("Annotated Data 4 vs Method 2")
# data = {'y_Actual':    flatAnnotatedData4,#["-1","0","0.5","1","1.5"],
#         'y_Predicted': flatMethod2Data#[1,0.9,0.6,0.7,0.1]
#         }
#
# df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
#
# confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
# print (confusion_matrix)
#
# annotatedDataHandler.log(["*"]*80)
# annotatedDataHandler.log("Annotated Data 4 vs Method 3")
# data = {'y_Actual':    flatAnnotatedData4,#["-1","0","0.5","1","1.5"],
#         'y_Predicted': flatMethod3Data#[1,0.9,0.6,0.7,0.1]
#         }
#
# df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
#
# confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
# print (confusion_matrix)
#
# annotatedDataHandler.log(["*"]*80)
max_epoch = 1000
max_accuracy = 0.0

annotatedDataHandler.log("Mode(Annotated Data) vs Method 1")
accuracy = 0.0
curr_epoch = 0
minFive = 0.5
maxFive = 0.8
best_threshold = thresholdsMethod1 = {"0.0": minFive, "0.5": maxFive}
best_report = {}
while curr_epoch < max_epoch:
    minFive += 0.01
    maxFive += 0.01
    if maxFive > 1:
        break
    curr_epoch += 1
    thresholdsMethod1 = {"0.0": minFive, "0.5": maxFive}
    annotatedDataHandler.readMethod2ComputedData(True, thresholdsMethod1)
    flatMethod2Data = annotatedDataHandler.collapseDataSetTo1d(annotatedDataHandler.method2Data)
    data = {'y_Actual': flatAnnotatedData,  # ["-1","0","0.5","1","1.5"],
            'y_Predicted': flatMethod2Data  # [1,0.9,0.6,0.7,0.1]
            }

    # df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    # flatAnnotatedData, flatMethod1Data
    #accuracy = accuracy_score(flatAnnotatedData, flatMethod1Data) #, rownames=['Actual'], colnames=['Predicted'])
    target_names = ['0.0', '0.5', "1.0", "1.5"]
    # report = classification_report(flatAnnotatedData, flatMethod1Data, target_names=target_names,output_dict=False,zero_division=0) #, rownames=['Actual'], colnames=['Predicted'])
    report = confusion_matrix(flatAnnotatedData, flatMethod2Data, labels=target_names) #, rownames=['Actual'], colnames=['Predicted'])

    print(report)
    exit()
    accuracy = report['accuracy']
    #accuracy = np.diag(confusion_matrix).sum() / confusion_matrix.to_numpy().sum()
    annotatedDataHandler.log(
        ["epoch:", curr_epoch, "accuracy:", accuracy, ", threshold: ", thresholdsMethod1, "max_accuracy:", max_accuracy,
         "best_threshold:", best_threshold])
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        best_threshold = thresholdsMethod1
        best_report = report

print(best_threshold)
print(max_accuracy)
print(best_report)
exit()

annotatedDataHandler.log(["*"] * 80)
annotatedDataHandler.log("Mode(Annotated Data) vs Method 2")
curr_epoch = 0
accuracy = 0.0
minFive = -0.01
maxFive = 0.0
best_threshold = thresholdsMethod2 = {"0.0": minFive, "0.5": maxFive}
while curr_epoch < max_epoch:
    minFive += 0.01
    maxFive += 0.01
    if maxFive > 1:
        break
    curr_epoch += 1
    thresholdsMethod2 = {"0.0": minFive, "0.5": maxFive}
    annotatedDataHandler.readMethod2ComputedData(True, thresholdsMethod2)
    flatMethod2Data = annotatedDataHandler.collapseDataSetTo1d(annotatedDataHandler.method2Data)
    data = {'y_Actual': flatAnnotatedData,  # ["-1","0","0.5","1","1.5"],
            'y_Predicted': flatMethod2Data  # [1,0.9,0.6,0.7,0.1]
            }
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    accuracy = np.diag(confusion_matrix).sum() / confusion_matrix.to_numpy().sum()
    annotatedDataHandler.log(
        ["epoch:", curr_epoch, "accuracy:", accuracy, ", threshold: ", thresholdsMethod2, "max_accuracy:", max_accuracy,
         "best_threshold:", best_threshold])
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        best_threshold = thresholdsMethod2

print(best_threshold)
print(max_accuracy)


annotatedDataHandler.log(["*"] * 80)
annotatedDataHandler.log("Mode(Annotated Data) vs Method 3")
curr_epoch = 0
accuracy = 0.0
minFive = -0.001
maxFive = 0.0
best_threshold = thresholdsMethod3 = {"0.0": minFive, "0.5": maxFive}
while curr_epoch < max_epoch:
    minFive += 0.01
    maxFive += 0.01
    if maxFive > 1:
        break
    curr_epoch += 1
    thresholdsMethod3 = {"0.0": minFive, "0.5": maxFive}
    annotatedDataHandler.readMethod3ComputedData(True, thresholdsMethod3)
    flatMethod3Data = annotatedDataHandler.collapseDataSetTo1d(annotatedDataHandler.method3Data)

    data = {'y_Actual': flatAnnotatedData,  # ["-1","0","0.5","1","1.5"],
            'y_Predicted': flatMethod3Data  # [1,0.9,0.6,0.7,0.1]
            }

    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])

    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    accuracy = np.diag(confusion_matrix).sum() / confusion_matrix.to_numpy().sum()
    annotatedDataHandler.log(
        ["epoch:", curr_epoch, "accuracy:", accuracy, ", threshold: ", thresholdsMethod3, "max_accuracy:", max_accuracy,
         "best_threshold:", best_threshold])
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        best_threshold = thresholdsMethod3

print(best_threshold)
print(max_accuracy)

# np.set_printoptions(precision=2)


# X = ["-1","0","0.5","1","1.5"]
# Y = [1,0.9,0.6,0.7,0.1]


# #cm = confusion_matrix(flatAnnotatedData, flatMethod1Data, sample_weight=None, labels=None, normalize=None)
# cm = confusion_matrix(X, Y, sample_weight=None, labels=None, normalize=None)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
# print(disp.confusion_matrix)
# disp.plot()

# disp.ax_.set_title("Confusion Matrix between Averaged Anntoated Data and Method 1")

# plt.show()
