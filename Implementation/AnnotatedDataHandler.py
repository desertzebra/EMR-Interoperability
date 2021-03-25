import csv
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score
import numpy as np
import re


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

    def log(self, msg, log_at="INFO"):
        if log_at in self.logLevel:
            if isinstance(msg, list):
                print(' '.join(str(v) for v in msg))
            else:
                print(msg)

    def readCSV(self, url):
        data = []
        with open(url) as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',')
            for row in csvReader:
                cellData = []
                for attr in row:
                    attr = self.convertAttrValue(attr)
                    cellData.append(attr)
                if len(cellData) > 0:
                    data.append(cellData)
        return data

    def readCSVWithoutHeaders(self, url):
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
                    attr = self.convertAttrValue(attr)

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
            (correlation_value, p_value) = pearsonr(list(float(v) for v in list1_attr), list(float(v) for v in list2_attr))
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
            d_score = cohen_kappa_score(list1_attr, list2_attr)
            formatted_d_score = str(round(d_score, 2))
            kappaCorrelationBetweenLists.append(formatted_d_score)
            # AnnotatedDataHandler = round(AnnotatedDataHandler, 2)
            # print('%7.4f %7.4f' % (AnnotatedDataHandler, p_value))
        return kappaCorrelationBetweenLists

    @staticmethod
    def convertAttrValue(attr):
        attr = attr.strip()
        if attr == '-' or attr == '~':
            attr = "-1.0"
        elif attr == '0':
            attr = "0.0"
        elif attr == '<':
            attr = "0.5"
        elif attr == '1':
            attr = "1.0"
        elif attr == '>':
            attr = "1.5"
        elif attr == '':
            attr = "-1.0"

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
            self.annotator1Data = annotatedDataHandler.readCSVWithoutHeaders('Data/Annotated/Annotator1.csv')
        else:
            self.annotator1Data = annotatedDataHandler.readCSV('Data/Annotated/Annotator1.csv')
        self.log(["annotator1 data loaded: ", len(self.annotator1Data), " with readHeaders=", readHeaders])
        # log("**********************************************************************************")
        # log("*****************************************Annotator 2*****************************************")
        if not readHeaders:
            self.annotator2Data = annotatedDataHandler.readCSVWithoutHeaders('Data/Annotated/Annotator2.csv')
        else:
            self.annotator2Data = annotatedDataHandler.readCSV('Data/Annotated/Annotator2.csv')
        self.log(["annotator2 data loaded: ", len(self.annotator2Data), " with readHeaders=", readHeaders])
        # log("**********************************************************************************")
        # log("*****************************************Annotator 3*****************************************")
        if not readHeaders:
            self.annotator3Data = annotatedDataHandler.readCSVWithoutHeaders('Data/Annotated/Annotator3.csv')
        else:
            self.annotator3Data = annotatedDataHandler.readCSV('Data/Annotated/Annotator3.csv')
        self.log(["annotator3 data loaded: ", len(self.annotator3Data), " with readHeaders=", readHeaders])
        # log("**********************************************************************************")
        # log("*****************************************Annotator 4*****************************************")
        if not readHeaders:
            self.annotator4Data = annotatedDataHandler.readCSVWithoutHeaders('Data/Annotated/Annotator4.csv')
        else:
            self.annotator4Data = annotatedDataHandler.readCSV('Data/Annotated/Annotator4.csv')
        self.log(["annotator4 data loaded: ", len(self.annotator4Data), " with readHeaders=", readHeaders])

        self.log("**********************************************************************************")

    def readAllComputedData(self, readHeaders=False):

        # log("*****************************************Annotator 1*****************************************")
        if not readHeaders:
            self.method1Data = annotatedDataHandler.readCSVWithoutHeaders('Data/0-table-V0.2.csv')
        else:
            self.method1Data = annotatedDataHandler.readCSV('Data/0-table-V0.2.csv')
        self.log(["method1 data loaded: ", len(self.method1Data), " with readHeaders=", readHeaders])
        # log("**********************************************************************************")
        # log("*****************************************Annotator 2*****************************************")
        if not readHeaders:
            self.method2Data = annotatedDataHandler.readCSVWithoutHeaders('Data/1-table-V0.2.csv')
        else:
            self.method2Data = annotatedDataHandler.readCSV('Data/1-table-V0.2.csv')
        self.log(["method2 data loaded: ", len(self.method2Data), " with readHeaders=", readHeaders])
        # log("**********************************************************************************")
        # log("*****************************************Annotator 3*****************************************")
        if not readHeaders:
            self.method3Data = annotatedDataHandler.readCSVWithoutHeaders('Data/2-table-V0.2.csv')
        else:
            self.method3Data = annotatedDataHandler.readCSV('Data/2-table-V0.2.csv')
        self.log(["method3 data loaded: ", len(self.method3Data), " with readHeaders=", readHeaders])

        self.log("**********************************************************************************")

    def calculatePearsonScoreBetweenAnnotators(self):
        if len(self.annotator1Data) < 1 or len(self.annotator2Data) < 1 or len(self.annotator3Data) < 1 or len(
                self.annotator4Data) < 1:
            self.log("Insufficient data for the annotators, have you read the files yet?")
            return

        # log('Pearson AnnotatedDataHandler between annotator1 and annotator2: ')
        allAnnotatedDataHandlers += "annotator1 vs annotator2," + ",".join(
            annotatedDataHandler.getPearsonCorrelationScore(self.annotator1Data, self.annotator2Data)) + "\r\n"
        # log("**********************************************************************************")

        # log('Pearson AnnotatedDataHandler between annotator1 and annotator3: ')
        allAnnotatedDataHandlers += "annotator1 vs annotator3," + ",".join(
            annotatedDataHandler.getPearsonCorrelationScore(self.annotator1Data, self.annotator3Data)) + "\r\n"
        # log("**********************************************************************************")

        # log('Pearson AnnotatedDataHandler between annotator1 and annotator4: ')
        allAnnotatedDataHandlers += "annotator1 vs annotator4," + ",".join(
            annotatedDataHandler.getPearsonCorrelationScore(self.annotator1Data, self.annotator4Data)) + "\r\n"
        # log("**********************************************************************************")

        # log('Pearson AnnotatedDataHandler between annotator2 and annotator3: ')
        allAnnotatedDataHandlers += "annotator2 vs annotator3," + ",".join(
            annotatedDataHandler.getPearsonCorrelationScore(self.annotator2Data, self.annotator3Data)) + "\r\n"
        # log("**********************************************************************************")

        # log('Pearson AnnotatedDataHandler between annotator2 and annotator4: ')
        allAnnotatedDataHandlers += "annotator2 vs annotator4," + ",".join(
            annotatedDataHandler.getPearsonCorrelationScore(self.annotator2Data, self.annotator4Data)) + "\r\n"
        # log("**********************************************************************************")

        # log('Pearson AnnotatedDataHandler between annotator3 and annotator4: ')
        allAnnotatedDataHandlers += "annotator3 vs annotator4," + ",".join(
            annotatedDataHandler.getPearsonCorrelationScore(self.annotator3Data, self.annotator4Data)) + "\r\n"
        self.log("**********************************************************************************")
        print(allAnnotatedDataHandlers)
        self.log("**********************************************************************************")

    def calculateKappaScoreBetweenAnnotators(self):
        if len(self.annotator1Data) < 1 or len(self.annotator2Data) < 1 or len(self.annotator3Data) < 1 or len(
                self.annotator4Data) < 1:
            self.log("Insufficient data for the annotators, have you read the files yet?")
            return
        allKappaAnnotatedDataHandlers = "\r\n"

        self.log('Cohen kappa score  between annotator1 and annotator2: ')
        allKappaAnnotatedDataHandlers += "annotator1 vs annotator2," + ",".join(
            annotatedDataHandler.getKappaCorrelationScore(self.annotator1Data, self.annotator2Data)) + "\r\n"
        self.log("**********************************************************************************")

        self.log('Cohen kappa score  between annotator1 and annotator3: ')
        allKappaAnnotatedDataHandlers += "annotator1 vs annotator3," + ",".join(
            annotatedDataHandler.getKappaCorrelationScore(self.annotator1Data, self.annotator3Data)) + "\r\n"
        self.log("**********************************************************************************")

        self.log('Cohen kappa score  between annotator1 and annotator4: ')
        allKappaAnnotatedDataHandlers += "annotator1 vs annotator4," + ",".join(
            annotatedDataHandler.getKappaCorrelationScore(self.annotator1Data, self.annotator4Data)) + "\r\n"
        self.log("**********************************************************************************")

        self.log('Cohen kappa score  between annotator2 and annotator3: ')
        allKappaAnnotatedDataHandlers += "annotator2 vs annotator3," + ",".join(
            annotatedDataHandler.getKappaCorrelationScore(self.annotator2Data, self.annotator3Data)) + "\r\n"
        self.log("**********************************************************************************")

        self.log('Cohen kappa score  between annotator2 and annotator4: ')
        allKappaAnnotatedDataHandlers += "annotator2 vs annotator4," + ",".join(
            annotatedDataHandler.getKappaCorrelationScore(self.annotator2Data, self.annotator4Data)) + "\r\n"
        self.log("**********************************************************************************")

        self.log('Cohen kappa score  between annotator3 and annotator4: ')
        allKappaAnnotatedDataHandlers += "annotator3 vs annotator4," + ",".join(
            annotatedDataHandler.getKappaCorrelationScore(self.annotator3Data, self.annotator4Data)) + "\r\n"
        self.log("**********************************************************************************")
        self.log(allKappaAnnotatedDataHandlers)

    def calculateKappaScoreBetweenComputedAndAnnotatedData(self, avgAnnotatedData=None):
        if avgAnnotatedData == None:
            if len(self.annotator1Data) < 1 or len(self.annotator2Data) < 1 or len(self.annotator3Data) < 1 or len(
                    self.annotator4Data) < 1:
                self.log("Insufficient data for the annotators, have you read the files yet?")
                return
            else:
                avgAnnotatedData = annotatedDataHandler.calculateAverageScoreBetweenAllAnnotators()

        if len(self.method1Data) < 1 or len(self.method1Data) < 1 or len(self.method1Data) < 1:
            self.log("Insufficient data for the computed methods, have you read the files yet?")
            return


        resultAsCsvString = "\r\n"

        self.log('Cohen kappa score  between method1 and avg(annotators): ')
        resultAsCsvString += "method1 vs avg(annotators)," + ",".join(
            annotatedDataHandler.getPearsonCorrelationScore(self.method1Data, avgAnnotatedData)) + "\r\n"
        self.log("**********************************************************************************")

        self.log('Cohen kappa score  between method2 and avg(annotators): ')
        resultAsCsvString += "method2 vs avg(annotators)," + ",".join(
            annotatedDataHandler.getPearsonCorrelationScore(self.method2Data, avgAnnotatedData)) + "\r\n"
        self.log("**********************************************************************************")
        self.log('Cohen kappa score  between method3 and avg(annotators): ')

        resultAsCsvString += "method3 vs avg(annotators)," + ",".join(
            annotatedDataHandler.getPearsonCorrelationScore(self.method3Data, avgAnnotatedData)) + "\r\n"
        self.log("**********************************************************************************")

        self.log("**********************************************************************************")
        self.log(resultAsCsvString)


    def calculateAverageScoreBetweenAllAnnotators(self):
        if len(self.annotator1Data) < 1 or len(self.annotator2Data) < 1 or len(self.annotator3Data) < 1 \
                or len(self.annotator4Data) < 1:
            self.log("Insufficient data for the annotators, have you read the files yet?")
            return
        averagedData = []
        rowIterator = 0
        for (a1Row, a2Row, a3Row, a4Row) in zip(self.annotator1Data, self.annotator2Data, self.annotator3Data,
                                                self.annotator4Data):
            rowIterator += 1
            colIterator = 0
            cellData = []
            for (a1Col, a2Col, a3Col, a4Col) in zip(a1Row, a2Row, a3Row, a4Row):
                colIterator += 1
                self.log(a1Col, "TRACE")
                if not self.isFloat(a1Col) or not self.isFloat(a2Col) or not self.isFloat(a3Col) or not self.isFloat(
                        a4Col):
                    if a1Col == a2Col == a3Col == a4Col:
                        cellData.append(a1Col)
                    else:
                        self.log("Major ERROR in code")
                        self.log([a1Col, a2Col, a3Col, a4Col])
                        exit()
                else:
                    avg = (float(a1Col) + float(a2Col) + float(a3Col) + float(a4Col)) / 4
                    cellData.append(avg)
            averagedData.append(cellData)
        return averagedData


annotatedDataHandler = AnnotatedDataHandler()
annotatedDataHandler.readAllAnnotatorsData(False)
annotatedDataHandler.readAllComputedData(False)
# annotatedDataHandler.calculateKappaScoreBetweenAnnotators()
avgAnnotatedData = annotatedDataHandler.calculateAverageScoreBetweenAllAnnotators()

# annotatedDataHandler.log(avgAnnotatedData, "DEBUG")

annotatedDataHandler.calculateKappaScoreBetweenComputedAndAnnotatedData(avgAnnotatedData)
