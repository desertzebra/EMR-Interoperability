import csv
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report, multilabel_confusion_matrix, \
    confusion_matrix
import numpy as np
# import matplotlib.pyplot as plt
import re
import pandas as pd
from statistics import mode, StatisticsError
import math
from matplotlib import pyplot as plt
from decimal import Decimal

np.seterr('raise')


class AnnotatedDataHandler:

    def __init__(self):
        self.logLevel = ["DEBUG",
                         "INFO"]  # ["TRACE", "DEBUG", "INFO"]       # Leave out only those levels, which should fire
        self.annotator1Data = []
        self.annotator2Data = []
        self.annotator3Data = []
        self.annotator4Data = []
        self.fuzzy_wuzzy_computed_data = []
        self.syn_sem_computed_data = []
        self.name_embedding_computed_data = []
        self.classes = ["0.0", "0.5", "1.0"]
        self.marker_set = ["o", "^", "x"]
        self.color_set = ["blue", "red", "green"]
        self.computed_method = ["fuzzy_wuzzy", "syn_and_sem", " name_embedding"]
        self.read_computed_data_from = [self.read_computed_data_from_fuzzy_wuzzy,
                                        self.read_computed_data_from_syn_and_sem,
                                        self.read_computed_data_from_name_embedding]

    def initDataStructures(self):
        self.annotator1Data = []
        self.annotator2Data = []
        self.annotator3Data = []
        self.annotator4Data = []
        self.fuzzy_wuzzy_computed_data = []
        self.syn_sem_computed_data = []
        self.name_embedding_computed_data = []

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
        elif attr == '~' or attr == '':
            attr = "-1.0"
        elif attr == '0':
            attr = "0.0"
        elif attr == '<' or attr == '>':
            attr = "0.5"
        elif attr == '1':
            attr = "1.0"
        # elif attr == '>':
        #     attr = "1.5"

        return attr

    @staticmethod
    def isFloat(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    # https://github.com/django/django/blob/main/django/utils/text.py
    @staticmethod
    def get_valid_filename(s):
        s = str(s).strip().replace(' ', '_')
        return re.sub(r'(?u)[^-\w.]', '_', s)

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

    def read_computed_data_from_fuzzy_wuzzy(self, readHeaders=False, thresholdValues={}):
        if not readHeaders:
            self.fuzzy_wuzzy_computed_data = annotatedDataHandler.readCSVWithoutHeaders(
                'Data/0-table-V0.5.csv', True, thresholdValues)
        else:
            self.fuzzy_wuzzy_computed_data = annotatedDataHandler.readCSV(
                'Data/0-table-V0.5.csv', True, thresholdValues)
        return self.fuzzy_wuzzy_computed_data
        # self.log(["method1 data loaded: ", len(self.fuzzy_wuzzy_computed_data), " with readHeaders=", readHeaders])

    def read_computed_data_from_syn_and_sem(self, readHeaders=False, thresholdValues={}):
        if not readHeaders:
            self.syn_sem_computed_data = annotatedDataHandler.readCSVWithoutHeaders(
                'Data/1-table-V0.5.csv', True,
                thresholdValues)
        else:
            self.syn_sem_computed_data = annotatedDataHandler.readCSV(
                'Data/1-table-V0.5.csv', True,
                thresholdValues)
        return self.syn_sem_computed_data
        # self.log(["method2 data loaded: ", len(self.syn_sem_computed_data), " with readHeaders=", readHeaders])

    def read_computed_data_from_name_embedding(self, readHeaders=False, thresholdValues={}):
        if not readHeaders:
            self.name_embedding_computed_data = annotatedDataHandler.readCSVWithoutHeaders(
                'Data/2-table-V0.5.csv', True, thresholdValues)
        else:
            self.name_embedding_computed_data = annotatedDataHandler.readCSV(
                'Data/2-table-V0.5.csv', True, thresholdValues)
        return self.name_embedding_computed_data
        # self.log(["method3 data loaded: ", len(self.name_embedding_computed_data), " with readHeaders=", readHeaders])

    def readAllComputedData(self, readHeaders=False, thresholdValues={}):
        self.read_computed_data_from_fuzzy_wuzzy(readHeaders, thresholdValues)
        # self.log(["*"] * 80)
        self.read_computed_data_from_syn_and_sem(readHeaders, thresholdValues)
        # self.log(["*"] * 80)
        self.read_computed_data_from_name_embedding(readHeaders, thresholdValues)

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

        if len(self.fuzzy_wuzzy_computed_data) < 1 or len(self.fuzzy_wuzzy_computed_data) < 1 or len(
                self.fuzzy_wuzzy_computed_data) < 1:
            self.log("Insufficient data for the computed methods, have you read the files yet?")
            return

        resultAsCsvString = "\r\n"

        self.log('Cohen kappa score  between method1 and avg(annotators): ')
        resultAsCsvString += "method1 vs avg(annotators)," + ",".join(
            annotatedDataHandler.getKappaCorrelationScore(self.fuzzy_wuzzy_computed_data, avgAnnotatedData)) + "\r\n"
        self.log(["*"] * 80)

        self.log('Cohen kappa score  between method2 and avg(annotators): ')
        resultAsCsvString += "method2 vs avg(annotators)," + ",".join(
            annotatedDataHandler.getKappaCorrelationScore(self.syn_sem_computed_data, avgAnnotatedData)) + "\r\n"

        self.log(["*"] * 80)
        self.log('Cohen kappa score  between method3 and avg(annotators): ')

        resultAsCsvString += "method3 vs avg(annotators)," + ",".join(
            annotatedDataHandler.getKappaCorrelationScore(self.name_embedding_computed_data, avgAnnotatedData)) + "\r\n"
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
                        self.log(["colIterator:", colIterator])
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

    def collapseDataSetTo1d(self, list2dWithoutHeaders):
        list1d = []
        rowIterator = 0
        colHeadNameList = []
        for row in list2dWithoutHeaders:
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

    def calculate_ovr_conditions(self, cm, thresholds):
        classes = ["0.0", "0.5", "1.0"]
        performance_dict = {}
        performance_dict["threshold"] = thresholds
        for pc_index, positiveClass in enumerate(classes):

            performance_dict[pc_index] = {}
            performance_dict[pc_index]["classPositive"] = positiveClass
            performance_dict[pc_index]["classNegative"] = []
            for c in classes:
                if c != positiveClass:
                    performance_dict[pc_index]["classNegative"].append(c)

            performance_dict[pc_index]["tp"] = cm[pc_index][pc_index]  # 0-0, 1-1, 2,2

            # must be a square matrix with rows = cols
            total_rows_in_cm = len(cm)
            total_cols_in_cm = len(cm[0])

            performance_dict[pc_index]["fn"] = cm[pc_index][(pc_index + 1) % total_cols_in_cm] + cm[pc_index][
                (pc_index + 2) % total_cols_in_cm]
            performance_dict[pc_index]["fp"] = cm[(pc_index + 1) % total_rows_in_cm][pc_index] + cm[(pc_index + 1) % 3][
                (pc_index + 2) % total_cols_in_cm]
            performance_dict[pc_index]["tn"] = cm[(pc_index + 2) % total_rows_in_cm][
                                                   (pc_index + 2) % total_cols_in_cm] + \
                                               cm[(pc_index + 1) % total_rows_in_cm][
                                                   (pc_index + 1) % total_cols_in_cm] + \
                                               cm[(pc_index + 2) % total_rows_in_cm][
                                                   (pc_index + 1) % total_cols_in_cm] + \
                                               cm[(pc_index + 1) % total_rows_in_cm][(pc_index + 2) % total_cols_in_cm]

            performance_dict[pc_index]["accuracy"] = (performance_dict[pc_index]["tp"] + performance_dict[pc_index][
                "tn"]) / (
                                                             performance_dict[pc_index]["tp"] +
                                                             performance_dict[pc_index]["fp"] +
                                                             performance_dict[pc_index]["fn"] +
                                                             performance_dict[pc_index]["tn"])

            if performance_dict[pc_index]["tp"] + performance_dict[pc_index]["fn"] == 0:
                performance_dict[pc_index]["sensitivity"] = 0
            else:
                performance_dict[pc_index]["sensitivity"] = (performance_dict[pc_index]["tp"]) / (
                        performance_dict[pc_index]["tp"] + performance_dict[pc_index]["fn"])

            if performance_dict[pc_index]["tn"] + performance_dict[pc_index]["fp"] == 0:
                performance_dict[pc_index]["specificity"] = 0
            else:
                performance_dict[pc_index]["specificity"] = (performance_dict[pc_index]["tn"]) / (
                        performance_dict[pc_index]["tn"] + performance_dict[pc_index]["fp"])
            if performance_dict[pc_index]["tp"] + performance_dict[pc_index]["fp"] == 0:
                performance_dict[pc_index]["precision"] = 0
            else:
                performance_dict[pc_index]["precision"] = (performance_dict[pc_index]["tp"]) / (
                        performance_dict[pc_index]["tp"] + performance_dict[pc_index]["fp"])

            if performance_dict[pc_index]["tp"] + performance_dict[pc_index]["fn"] == 0:
                performance_dict[pc_index]["recall"] = 0
            else:
                performance_dict[pc_index]["recall"] = (performance_dict[pc_index]["tp"]) / (
                        performance_dict[pc_index]["tp"] + performance_dict[pc_index]["fn"])

            performance_dict[pc_index]["f-measure"] = {}
            for B in [0.5, 1, 2]:

                f_measure_numerator = (1 + (B * B)) * (
                    performance_dict[pc_index]["tp"])
                f_measure_denominator = ((1 + (B * B)) * (performance_dict[pc_index]["tp"])) + (
                        (B * B) * performance_dict[pc_index]["fn"]) + (performance_dict[pc_index]["fp"])
                if f_measure_denominator == 0:
                    performance_dict[pc_index]["f-measure"][str(B)] = 0
                else:
                    if f_measure_numerator / f_measure_denominator > 1:
                        print(f_measure_numerator, "/", f_measure_denominator)
                    performance_dict[pc_index]["f-measure"][str(B)] = f_measure_numerator / f_measure_denominator

            performance_dict[pc_index]["G-mean1"] = math.sqrt(performance_dict[pc_index]["sensitivity"] * \
                                                              performance_dict[pc_index]["precision"])
            performance_dict[pc_index]["G-mean2"] = math.sqrt(
                performance_dict[pc_index]["sensitivity"] * performance_dict[pc_index]["specificity"])

            mcc_numerator = (performance_dict[pc_index]["tp"] * performance_dict[pc_index]["tn"]) - (
                    performance_dict[pc_index]["fp"] * performance_dict[pc_index]["fn"])
            mcc_denominator = math.sqrt((performance_dict[pc_index]["tp"] + performance_dict[pc_index]["fp"]) * (
                    performance_dict[pc_index]["tp"] + performance_dict[pc_index]["fn"]) * (
                                                performance_dict[pc_index]["tn"] + performance_dict[pc_index][
                                            "fp"]) * (performance_dict[pc_index]["tn"] + performance_dict[pc_index][
                "fn"]))
            if mcc_denominator == 0:
                performance_dict[pc_index]["mcc"] = 0
            else:
                performance_dict[pc_index]["mcc"] = mcc_numerator / mcc_denominator

        return performance_dict

    def evaluateMethod(self, annotatedData, methodIndex=1):
        annotatedDataHandler.log("Mode(Annotated Data) vs " + self.computed_method[methodIndex])
        minFive = float(0.0)
        # maxFive = 0.1
        ovr_conditions = []
        ovo_conditions = []
        while minFive < float(0.991):
            maxFive = minFive + float(0.01)
            while maxFive <= float(1.0):
                thresholds = {"0.0": minFive, "0.5": maxFive}
                conditions = {}
                data_in_2d = self.read_computed_data_from[methodIndex](True, thresholds)
                data_in_1d = annotatedDataHandler.collapseDataSetTo1d(data_in_2d)
                # data = {'y_Actual': annotatedData,  # ["-1","0","0.5","1","1.5"],
                #         'y_Predicted': data_in_1d  # [1,0.9,0.6,0.7,0.1]
                #         }
                target_names = ['0.0', '0.5', "1.0"]  # , "1.5"]

                cm = confusion_matrix(annotatedData, data_in_1d,
                                      labels=target_names)  # , rownames=['Actual'], colnames=['Predicted'])
                conditions["ovr"] = annotatedDataHandler.calculate_ovr_conditions(cm, thresholds)
                # print(thresholds)
                ovr_conditions.append(conditions["ovr"])
                maxFive = float(Decimal(maxFive) + Decimal('.01'))
            minFive = float(Decimal(minFive) + Decimal('.01'))

        print("Method " + self.computed_method[methodIndex] + " finished processing.")

        annotatedDataHandler.plot_roc(ovr_conditions,
                                      'ROC for Mode(Annotated Data) vs ' + self.computed_method[methodIndex])

        annotatedDataHandler.plot_scatter_for_mcc_vs_f1(ovr_conditions,
                                                        'MCC vs F1 for Mode(Annotated Data) vs ' + self.computed_method[
                                                            methodIndex])

        annotatedDataHandler.plot_pr(ovr_conditions,
                                     'Precision vs Recall for Mode(Annotated Data) vs ' + self.computed_method[
                                         methodIndex])

    def plot_roc(self, condition_from_experimental_iterations, plotTitle):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)
        plt.xlabel('False Positive Rate (FPR)')
        # Labeling the Y-axis
        plt.ylabel('sensitivity')
        plt.title(plotTitle)
        for positive_class, marker, color in zip(self.classes, self.marker_set, self.color_set):
            sensitivity_from_experimental_iterations = []
            fpr_from_experimental_iterations = []
            thresholds_from_experimental_iterations = []
            for cond in condition_from_experimental_iterations:
                for index in range(0, len(self.classes)):
                    class_result = cond[index]
                    if class_result['classPositive'] == positive_class:
                        sensitivity_from_experimental_iterations.append(class_result['sensitivity'])
                        fpr_from_experimental_iterations.append(1 - class_result["specificity"])
                        thresholds_from_experimental_iterations.append(cond["threshold"])

            # plt.plot(sensitivity_from_experimental_iterations, fpr_from_experimental_iterations, linestyle='dashed', linewidth=2, label=positive_class, marker=marker, markerfacecolor=color, markersize=4)
            plt.scatter(sensitivity_from_experimental_iterations, fpr_from_experimental_iterations,
                        label=positive_class, marker=marker)
            # for x, y, z in zip(sensitivity_from_experimental_iterations, fpr_from_experimental_iterations,
            #                    thresholds_from_experimental_iterations):
            #     # print(z["0.0"])
            #     ax.annotate('(T_0=%.2f,T_0.5=%.2f)' % (z["0.0"], z["0.5"]), xy=(x, y))
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
        plt.xticks(np.arange(0, 1, step=0.02), rotation=45)
        plt.yticks(np.arange(0, 1, step=0.02))
        # plt.grid(True)
        plt.legend()
        # plt.show()
        self.log("saving plot:" + plotTitle)
        fig.savefig("Results/charts/" + self.get_valid_filename(plotTitle), bbox_inches='tight')

    # Plot precision vs recall
    def plot_pr(self, condition_from_experimental_iterations, plotTitle):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(plotTitle)
        for positive_class, marker, color in zip(self.classes, self.marker_set, self.color_set):
            recall_from_experimental_iterations = []
            precision_from_experimental_iterations = []
            thresholds_from_experimental_iterations = []
            for cond in condition_from_experimental_iterations:
                for index in range(0, len(self.classes)):
                    class_result = cond[index]
                    if class_result['classPositive'] == positive_class:
                        precision_from_experimental_iterations.append(class_result['precision'])
                        recall_from_experimental_iterations.append(class_result["recall"])
                        thresholds_from_experimental_iterations.append(cond["threshold"])

            plt.plot(recall_from_experimental_iterations, precision_from_experimental_iterations, linestyle='dashed',
                     linewidth=2, label=positive_class, marker=marker, markerfacecolor=color, markersize=6)
            # for x, y, z in zip(recall_from_experimental_iterations, precision_from_experimental_iterations,
            #                    thresholds_from_experimental_iterations):
            #     # print(z["0.0"])
            #     ax.annotate('(%.1f,%.1f)' % (z["0.0"], z["0.5"]), xy=(x, y))
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
        plt.xticks(np.arange(0, 1, step=0.02), rotation=45)
        plt.yticks(np.arange(0, 1, step=0.02))
        # plt.grid(True)
        plt.legend()
        # plt.show()
        self.log("saving plot:" + plotTitle)
        fig.savefig("Results/charts/" + self.get_valid_filename(plotTitle), bbox_inches='tight')

    def plot_scatter_for_mcc_vs_f1(self, condition_from_experimental_iterations, plotTitle):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)
        plt.xlabel("MCC")
        # Labeling the Y-axis
        plt.ylabel("F-1 Score")
        plt.title(plotTitle)
        for positive_class, marker, color in zip(self.classes, self.marker_set, self.color_set):
            sensitivity_from_experimental_iterations = []
            fpr_from_experimental_iterations = []
            thresholds_from_experimental_iterations = []
            for cond in condition_from_experimental_iterations:
                for index in range(0, len(self.classes)):
                    class_result = cond[index]
                    if class_result['classPositive'] == positive_class:
                        sensitivity_from_experimental_iterations.append(class_result['mcc'])
                        fpr_from_experimental_iterations.append(class_result['f-measure']['1'])
                        thresholds_from_experimental_iterations.append(cond["threshold"])

            plt.scatter(sensitivity_from_experimental_iterations, fpr_from_experimental_iterations,
                        label=positive_class, marker=marker)
        # plt.grid(True)
        plt.xticks(np.arange(-1, 1, step=0.05), rotation=45)
        plt.yticks(np.arange(0, 1.02, step=0.02))
        plt.legend()
        # plt.show()
        self.log("saving plot:" + plotTitle)
        fig.savefig("Results/charts/" + self.get_valid_filename(plotTitle), bbox_inches='tight')

annotatedDataHandler = AnnotatedDataHandler()

# annotatedDataHandler.calculateKappaScoreBetweenAnnotators()
# annotatedDataHandler.calculateKappaScoreBetweenComputedAndAnnotatedData(modeAnnotatedData)

# Now to convert the datasets into 1d

annotatedDataHandler.initDataStructures()
annotatedDataHandler.readAllAnnotatorsData(True)
modeAnnotatedData = annotatedDataHandler.calculateModeScoreBetweenAllAnnotators(True)
flatAnnotatedData = annotatedDataHandler.collapseDataSetTo1d(modeAnnotatedData)

annotatedDataHandler.evaluateMethod(flatAnnotatedData, 0)
annotatedDataHandler.evaluateMethod(flatAnnotatedData, 1)
annotatedDataHandler.evaluateMethod(flatAnnotatedData, 2)
