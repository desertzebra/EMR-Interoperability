import csv
import os
from scipy.stats import pearsonr
import numpy as np
import re
import pandas as pd
from statistics import mode, StatisticsError
import math
# from matplotlib import pyplot as plt
from decimal import Decimal
from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report, multilabel_confusion_matrix, \
    confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

np.seterr('raise')


# list1 = [1, 2, 3, 4, 5, 6, 8, 1, 2]
# list2 = [1, 3, 5, 2, 5, 6, 8, 2 ,1 ]
# score = cohen_kappa_score(list1, list2)
# print("score: ", score)
# exit()

def get_kappa_correlation_score(list1, list2):
    kappaCorrelationBetweenLists = []
    list1Class = [t[2] for t in list1]
    list2Class = [t[2] for t in list2]
    d_score = cohen_kappa_score(list1Class, list2Class)
    # formatted_d_score = str(round(d_score, 2))
    return d_score


def get_pearson_correlation_score(list1, list2):
    correlationBetweenLists = []
    for list1_attr, list2_attr in zip(list1, list2):
        (correlation_value, p_value) = pearsonr(list(float(v) for v in list1_attr),
                                                list(float(v) for v in list2_attr))
        formatted_correlation_value = str(round(correlation_value, 2))
        correlationBetweenLists.append(formatted_correlation_value)
        # AnnotatedDataHandler = round(AnnotatedDataHandler, 2)
        # print('%7.4f %7.4f' % (AnnotatedDataHandler, p_value))
    return correlationBetweenLists


class AnnotatedDataHandler:

    def __init__(self):
        self.logDEBUG = "DEBUG"
        self.logINFO = "INFO"
        self.logTRACE = "TRACE"
        self.logLevel = ["DEBUG",
                         "INFO"]  # ["TRACE", "DEBUG", "INFO"]       # Leave out only those levels, which should fire

        # Annotations
        self.annotator1Data = []
        self.annotator2Data = []
        self.annotator3Data = []
        self.annotator4Data = []

        # Computational accessories
        self.result_indexes = ["0.0-1.0", "0.1-0.9", "0.2-0.8", "0.3-0.7", "0.4-0.6", "0.5-0.5", "0.6-0.4", "0.7-0.3",
                               "0.8-0.2", "0.9-0.1"]
        self.result_file_index = "0.0-1.0"
        self.computational_iteration = "1.6"
        self.result_iteration = ".2"
        self.notSynAndSem = ["FUZZY_MATCH", "Word2Vec"]
        self.computed_method = ['bert-base-nli-stsb-mean-tokens',
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
        # Computed Data Structures
        self.computed_data = {}
        for m in self.notSynAndSem:
            self.computed_data[m] = {}
        for m in self.computed_method:
            self.computed_data[m] = {}

        self.class_equal = "1.0"
        self.class_related = "0.5"
        self.class_unrelated = "0.0"
        # self.roc_dict = {}
        # self.max_roc_dict = {}
        # self.prc_dict = {}
        # self.max_prc_dict = {}
        # Plot accessories
        self.classes = [self.class_unrelated, self.class_related, self.class_equal]
        self.marker_set = ["o", "^", "x"]
        self.color_set = ["blue", "red", "green"]
        # Directories
        self.raw_dict_result_dir = "Results/raw_dicts/" + str(self.computational_iteration) + str(
            self.result_iteration) + "/"
        self.resultDir = "Results/charts/" + str(self.computational_iteration) + str(self.result_iteration) + "/"
        self.roc_result_dir = self.resultDir + "/roc/"
        self.prc_result_dir = self.resultDir + "/prc/"
        # Make sure the folder for results exists
        if not os.path.exists(self.roc_result_dir):
            os.makedirs(self.roc_result_dir)
        if not os.path.exists(self.prc_result_dir):
            os.makedirs(self.prc_result_dir)
        if not os.path.exists(self.raw_dict_result_dir):
            os.makedirs(self.raw_dict_result_dir)


    def log(self, msg, log_at="INFO"):
        if log_at in self.logLevel:
            if isinstance(msg, list):
                print(' '.join(str(v) for v in msg))
            else:
                print(str(msg))

    def readCSV(self, url, is_computed=False, _thresholds={}):
        data = []
        with open(url) as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',')
            for row in csvReader:
                if len(row) == 0:
                    continue
                cellData = []
                for attr in row:
                    if not is_computed:
                        attr = self.convertAnnotatedAttrValue(attr)
                    else:
                        attr = self.convert_computed_attr_value(attr, _thresholds)
                        # attr = attr.strip()
                    if attr == '-':
                        continue
                    else:
                        cellData.append(attr)
                    # cellData.append(attr)
                if len(cellData) > 0:
                    data.append(cellData)
        return data

    def readCSVWithoutHeaders(self, url, is_computed=False, _thresholds={}):
        data = []
        with open(url) as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',')
            rowIterator = 0
            for row in csvReader:
                if len(row) == 0:
                    continue
                rowIterator += 1
                if rowIterator == 1:
                    continue
                colIterator = 0
                cellData = []
                for attr in row:
                    colIterator += 1
                    if colIterator == 1:
                        continue

                    # Convert the attribute values, according to _thresholds
                    if not is_computed:
                        attr = self.convertAnnotatedAttrValue(attr)
                    else:
                        attr = self.convert_computed_attr_value(attr, _thresholds)
                        # attr = attr.strip()

                    if attr == '':
                        self.log(["rowIterator:", rowIterator, ",colIterator:", colIterator, ",attr:", attr], "DEBUG")
                        # colIterator = colIterator-1
                        continue
                    elif attr == '-':
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

    def get_default_threshold_values(self, threshold_values={}):

        # initializing default threshold values. Perhaps this should be in a separate function?
        if self.class_unrelated not in threshold_values or threshold_values[self.class_unrelated] == "undefined":
            print("setting default value for 0.0", threshold_values)
            threshold_values[self.class_unrelated] = 0.6
        if self.class_related not in threshold_values or threshold_values[self.class_related] == "undefined":
            print("setting default value for 0.5", threshold_values)
            threshold_values[self.class_related] = 0.8
        return threshold_values

    def convert_computed_attr_value(self, attr, threshold_values={}):
        if not self.isFloat(attr):
            attr = attr.strip()

        _thresholds = self.get_default_threshold_values(threshold_values)

        if attr == '-1' or attr == '-' or attr == '':
            return "-"
        # elif self.isFloat(attr) and float(attr) < 0.0:
        #     self.log(["attr:", attr])
        #     return self.class_unrelated
        # elif attr == '~' or attr == '':
        #     return "0.0"
        # elif self.isFloat(attr) and 0 <= float(attr) < thresholds["0.0"]:
        elif self.isFloat(attr) and float(attr) < _thresholds[self.class_unrelated]:
            return self.class_unrelated
        elif self.isFloat(attr) and _thresholds[self.class_unrelated] <= float(attr) < _thresholds[self.class_related]:
            return self.class_related
        elif self.isFloat(attr) and _thresholds[self.class_related] <= float(attr):
            return self.class_equal

        # check if some float value was missed
        if self.isFloat(attr):
            self.log(["attr:", attr], "DEBUG")

        return attr

    def convertAnnotatedAttrValue(self, attr):
        attr = attr.strip()
        if attr == '-':
            attr = "-"
        elif attr == '~' or attr == '':
            attr = "-1.0"
        elif attr == '0':
            attr = self.class_unrelated
        elif attr == '<' or attr == '>':
            attr = self.class_related
        elif attr == '1':
            attr = self.class_equal
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
        filename = re.sub(r'(?u)[^-\w.]', '_', s)
        return filename + ".png"

    def readAllAnnotatorsData(self, read_headers=False):
        # log("*****************************************Annotator 1*****************************************")
        if not read_headers:
            self.annotator1Data = annotatedDataHandler.readCSVWithoutHeaders(
                'Data/Annotated/updated/Annotator1.csv')
        else:
            self.annotator1Data = annotatedDataHandler.readCSV(
                'Data/Annotated/updated/Annotator1.csv')
        self.log(["annotator1 data loaded: ", len(self.annotator1Data), " with read_headers=", read_headers])
        # log(["*"]*80)
        # log("*****************************************Annotator 2*****************************************")
        if not read_headers:
            self.annotator2Data = annotatedDataHandler.readCSVWithoutHeaders(
                'Data/Annotated/updated/Annotator2.csv')
        else:
            self.annotator2Data = annotatedDataHandler.readCSV(
                'Data/Annotated/updated/Annotator2.csv')
        self.log(["annotator2 data loaded: ", len(self.annotator2Data), " with read_headers=", read_headers])
        # log(["*"]*80)
        # log("*****************************************Annotator 3*****************************************")
        if not read_headers:
            self.annotator3Data = annotatedDataHandler.readCSVWithoutHeaders(
                'Data/Annotated/updated/Annotator3.csv')
        else:
            self.annotator3Data = annotatedDataHandler.readCSV(
                'Data/Annotated/updated/Annotator3.csv')
        self.log(["annotator3 data loaded: ", len(self.annotator3Data), " with read_headers=", read_headers])
        # log(["*"]*80)
        # log("*****************************************Annotator 4*****************************************")
        if not read_headers:
            self.annotator4Data = annotatedDataHandler.readCSVWithoutHeaders(
                'Data/Annotated/updated/Annotator4.csv')
        else:
            self.annotator4Data = annotatedDataHandler.readCSV(
                'Data/Annotated/updated/Annotator4.csv')
        self.log(["annotator4 data loaded: ", len(self.annotator4Data), " with read_headers=", read_headers])

        self.log(["*"] * 80)

    def read_computed_data_for_model(self, model_as_filename, read_headers=False, syn_sem_threshold="0-0",
                                     _thresholds={}):
        self.log(["Reading data produced by ", model_as_filename, " at syn-sem-threshold:", syn_sem_threshold], \
                 self.logTRACE)

        if not read_headers:
            return annotatedDataHandler.readCSVWithoutHeaders(
                'Data/' + self.computational_iteration + "/" + str(
                    model_as_filename) + '-table-V-' + syn_sem_threshold + '.csv',
                True, _thresholds)
        else:
            return annotatedDataHandler.readCSV(
                'Data/' + self.computational_iteration + "/" + str(
                    model_as_filename) + '-table-V-' + syn_sem_threshold + '.csv',
                True, _thresholds)

    def read_all_computed_data(self, read_headers=False, _thresholds={}):
        # Read own method
        annotatedDataHandler.log(['*'] * 80)
        for model in self.computed_method:
            for syn_sem_threshold in self.result_indexes:
                self.computed_data[model][syn_sem_threshold] = self.read_computed_data_for_model(model + "_syn_sem",
                                                                                                 read_headers,
                                                                                                 syn_sem_threshold,
                                                                                                 _thresholds)
        # Read other methods
        annotatedDataHandler.log(['-'] * 80)
        for model in self.notSynAndSem:
            syn_sem_threshold = "0-0"
            self.computed_data[model][syn_sem_threshold] = self.read_computed_data_for_model(model, read_headers,
                                                                                             syn_sem_threshold,
                                                                                             _thresholds)
        annotatedDataHandler.log(['*'] * 80)

    def calculate_pearson_score_between_annotators(self):
        allAnnotatedDataHandlers = ""
        if len(self.annotator1Data) < 1 or len(self.annotator2Data) < 1 or len(self.annotator3Data) < 1 or len(
                self.annotator4Data) < 1:
            self.log("Insufficient data for the annotators, have you read the files yet?")
            return

        # log('Pearson AnnotatedDataHandler between annotator1 and annotator2: ')
        allAnnotatedDataHandlers += "annotator1 vs annotator2," + ",".join(
            annotatedDataHandler.get_pearson_correlation_score(self.annotator1Data, self.annotator2Data)) + "\r\n"
        # log(["*"]*80)

        # log('Pearson AnnotatedDataHandler between annotator1 and annotator3: ')
        allAnnotatedDataHandlers += "annotator1 vs annotator3," + ",".join(
            annotatedDataHandler.get_pearson_correlation_score(self.annotator1Data, self.annotator3Data)) + "\r\n"
        # log(["*"]*80)

        # log('Pearson AnnotatedDataHandler between annotator1 and annotator4: ')
        allAnnotatedDataHandlers += "annotator1 vs annotator4," + ",".join(
            annotatedDataHandler.get_pearson_correlation_score(self.annotator1Data, self.annotator4Data)) + "\r\n"
        # log(["*"]*80)

        # log('Pearson AnnotatedDataHandler between annotator2 and annotator3: ')
        allAnnotatedDataHandlers += "annotator2 vs annotator3," + ",".join(
            annotatedDataHandler.get_pearson_correlation_score(self.annotator2Data, self.annotator3Data)) + "\r\n"
        # log(["*"]*80)

        # log('Pearson AnnotatedDataHandler between annotator2 and annotator4: ')
        allAnnotatedDataHandlers += "annotator2 vs annotator4," + ",".join(
            annotatedDataHandler.get_pearson_correlation_score(self.annotator2Data, self.annotator4Data)) + "\r\n"
        # log(["*"]*80)

        # log('Pearson AnnotatedDataHandler between annotator3 and annotator4: ')
        allAnnotatedDataHandlers += "annotator3 vs annotator4," + ",".join(
            annotatedDataHandler.get_pearson_correlation_score(self.annotator3Data, self.annotator4Data)) + "\r\n"
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
        allKappaAnnotatedDataHandlers += "annotator1 vs annotator2," + annotatedDataHandler.get_kappa_correlation_score(
            self.annotator1Data, self.annotator2Data) + "\r\n"
        self.log(["*"] * 80)

        self.log('Cohen kappa score  between annotator1 and annotator3: ')
        allKappaAnnotatedDataHandlers += "annotator1 vs annotator3," + \
                                         annotatedDataHandler.get_kappa_correlation_score(self.annotator1Data,
                                                                                          self.annotator3Data) + "\r\n"
        self.log(["*"] * 80)

        self.log('Cohen kappa score  between annotator1 and annotator4: ')
        allKappaAnnotatedDataHandlers += "annotator1 vs annotator4," + \
                                         annotatedDataHandler.get_kappa_correlation_score(self.annotator1Data,
                                                                                          self.annotator4Data) + "\r\n"
        self.log(["*"] * 80)

        self.log('Cohen kappa score  between annotator2 and annotator3: ')
        allKappaAnnotatedDataHandlers += "annotator2 vs annotator3," + \
                                         annotatedDataHandler.get_kappa_correlation_score(self.annotator2Data,
                                                                                          self.annotator3Data) + "\r\n"
        self.log(["*"] * 80)

        self.log('Cohen kappa score  between annotator2 and annotator4: ')
        allKappaAnnotatedDataHandlers += "annotator2 vs annotator4," + \
                                         annotatedDataHandler.get_kappa_correlation_score(self.annotator2Data,
                                                                                          self.annotator4Data) + "\r\n"
        annotatedDataHandler.log(["*"] * 80)

        self.log('Cohen kappa score  between annotator3 and annotator4: ')
        allKappaAnnotatedDataHandlers += "annotator3 vs annotator4," + \
                                         annotatedDataHandler.get_kappa_correlation_score(self.annotator3Data,
                                                                                          self.annotator4Data) + "\r\n"
        annotatedDataHandler.log(["*"] * 80)
        self.log(allKappaAnnotatedDataHandlers)

    def calculateKappaScoreBetweenComputedAndAnnotatedData(self, annotated_data=None):
        if annotated_data is None:
            if len(self.annotator1Data) < 1 or len(self.annotator2Data) < 1 or len(self.annotator3Data) < 1 or len(
                    self.annotator4Data) < 1:
                self.log("Insufficient data for the annotators, have you read the files yet?")
                return
            else:
                annotated_data = annotatedDataHandler.calculateModeScoreBetweenAllAnnotators()

        # result_as_csv_str = "\r\n"
        result_as_dict = {}
        # Base methods
        for m in self.notSynAndSem:
            result_as_dict[m] = {}

            if len(self.computed_data[m]) < 1:
                self.log("Insufficient data for the computed methods, have you read the files yet?")
                return
            # self.log('Cohen kappa score  between ' + m + ' and avg(annotators): ')
            data_in_1d = self.collapseDataSetTo1dArrayWithHeaders(self.computed_data[m]["0-0"])
            result_as_dict[m]["0-0"] = get_kappa_correlation_score(data_in_1d, annotated_data)
            # result_as_csv_str += m + "-0-0 vs mode(annotators)," + get_kappa_correlation_score(data_in_1d,
            #                                                                                    annotated_data) + "\r\n"


        # print(self.computed_data)
        for m in self.computed_method:
            result_as_dict[m] = {}
            for _syn_sem_threshold in self.result_indexes:
                if len(self.computed_data[m]) < 1:
                    self.log("Insufficient data for the computed methods, have you read the files yet?")
                    return
                # self.log('Cohen kappa score  between ' + m + ' and avg(annotators): ')
                data_in_1d = self.collapseDataSetTo1dArrayWithHeaders(self.computed_data[m][_syn_sem_threshold])
                result_as_dict[m][_syn_sem_threshold] = get_kappa_correlation_score(data_in_1d, annotated_data)
                # result_as_csv_str += m + "-" + str(syn_sem_threshold) + " vs mode(annotators)," + \
                #                      get_kappa_correlation_score(data_in_1d, annotated_data) + "\r\n"

        # print("result_as_csv_str:", result_as_csv_str)
        return result_as_dict
        # self.log(["*"] * 80)

    def calculateModeScoreBetweenAllAnnotators(self, has_headers=False):
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
            if has_headers and rowIterator == 1:
                averagedData.append(a1Row)
                continue
            colIterator = 0
            cellData = []

            for (a1Col, a2Col, a3Col, a4Col) in zip(a1Row, a2Row, a3Row, a4Row):
                colIterator += 1
                self.log(a1Col, "TRACE")
                if has_headers and colIterator == 1:
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
                        return  # Hack to avoid these stupid formatting notices
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
                # print("collapseDataSetTo1dArrayWithHeaders:", len(colHeadNameList))
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
                    print("collapseDataSetTo1dArrayWithHeaders except:", row)
                    print("collapseDataSetTo1dArrayWithHeaders except:colIterator:", colIterator,
                          ", size of colHeadNameList:", len(colHeadNameList))
                    exit()
                    return  # Hack to avoid these stupid formatting notices
                # attr = self.convertAttrValue(attr)
                list1d.append([rowHeaderName, colHeaderName, attr])
        return list1d

    def compare1dLists(self, list2dWithHeader1, list2dWithHeader2):

        if (len(list2dWithHeader1) != len(list2dWithHeader2)):
            self.log(["len(list2dWithHeader1):", len(list2dWithHeader1)])
            self.log(["len(list2dWithHeader2):", len(list2dWithHeader2)])
            exit()
        countProblematicRows = 0

        for (row1, row2) in zip(list2dWithHeader1, list2dWithHeader2):
            for (a1Col, a2Col) in zip(row1, row2):
                if not self.isFloat(a1Col) and not self.isFloat(a2Col) and not a1Col == a2Col:
                    countProblematicRows += 1
                    self.log(["row1:", row1])
                    self.log(["row2:", row2])
                    self.log(["a1Col:", a1Col, ",a2Col:", a2Col])
                    exit()
        if (countProblematicRows > 0):
            self.log(["countProblematicRows:", countProblematicRows])
            exit()

    # Calcualte performance metrics from confusion matrix
    def calculate_ovr_conditions(self, cm, thresholds):
        performance_dict = {}
        performance_dict["threshold"] = thresholds
        for pc_index, positiveClass in enumerate(self.classes):

            performance_dict[pc_index] = {}
            performance_dict[pc_index]["classPositive"] = positiveClass
            performance_dict[pc_index]["classNegative"] = []
            for c in self.classes:
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
                        print("calculate_ovr_conditions:", f_measure_numerator, "/", f_measure_denominator)
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

    def writeDetailedDictToCsv(self, dict={}, dict_name="result", type="kappa", is_max_score=False):
        file_name = self.raw_dict_result_dir + "" + dict_name
        dict_table = []

        with open(file_name, 'w') as csv_file:
            self.log("Saving CSV data file for: " + dict_name)
            csv_writer = csv.writer(csv_file, delimiter=',')
            # pd.DataFrame(annotatedDataHandler.roc_dict).to_csv('roc_dict')
            if type == "kappa":
                csv_writer.writerow(["Outer Threshold", "Method name", "Inner Threshold", "d score (kappa)"])
            elif type == "condition":
                csv_writer.writerow(["Outer Threshold", "Method name", "Inner Threshold", "AUC method",
                                     " AUC Score - " + self.class_unrelated, " AUC Score - " + self.class_related,
                                     " AUC Score - " + self.class_equal,
                                     "Conditions - positive class=" + self.class_unrelated
                                        , "Conditions - positive class=" + self.class_related
                                        , "Conditions - positive class=" + self.class_equal])
            # else:
                # No header?

            for k_outer_threshold, v_outer_threshold in dict.items():
                # print("k_outer_threshold:", k_outer_threshold, ", items: ", len(v_outer_threshold.items()))
                for k_method_name, v_method_name in v_outer_threshold.items():
                    # print("k_method_name:",k_method_name)
                    if not isinstance(v_method_name, list):
                        if type == "kappa":
                            for k_inner_threshold, v_inner_threshold in v_method_name.items():
                                csv_writer.writerow(
                                    [k_outer_threshold, k_method_name, k_inner_threshold,v_inner_threshold])

            print("done")

    # def plot_scatter_for_mcc_vs_f1(self, condition_from_experimental_iterations, plotTitle):
    #     fig = plt.figure(figsize=(20, 10))
    #     ax = fig.add_subplot(111)
    #     plt.xlabel("MCC")
    #     # Labeling the Y-axis
    #     plt.ylabel("F-1 Score")
    #     plt.title(plotTitle)
    #     for positive_class, marker, color in zip(self.classes, self.marker_set, self.color_set):
    #         sensitivity_from_experimental_iterations = []
    #         fpr_from_experimental_iterations = []
    #         thresholds_from_experimental_iterations = []
    #         for cond in condition_from_experimental_iterations:
    #             for index in range(0, len(self.classes)):
    #                 class_result = cond[index]
    #                 if class_result['classPositive'] == positive_class:
    #                     sensitivity_from_experimental_iterations.append(class_result['mcc'])
    #                     fpr_from_experimental_iterations.append(class_result['f-measure']['1'])
    #                     thresholds_from_experimental_iterations.append(cond["threshold"])
    #
    #         plt.scatter(sensitivity_from_experimental_iterations, fpr_from_experimental_iterations,
    #                     label=positive_class, marker=marker)
    #     # plt.grid(True)
    #     plt.xticks(np.arange(-1, 1, step=0.05), rotation=45)
    #     plt.yticks(np.arange(0, 1.02, step=0.02))
    #     plt.legend()
    #     # plt.show()
    #     self.log("saving plot:" + plotTitle)
    #     fig.savefig(self.resultDir + self.get_valid_filename(plotTitle), bbox_inches='tight')
    #     plt.close(fig)


# Main script
annotatedDataHandler = AnnotatedDataHandler()
# Now to convert the datasets into 1d
annotatedDataHandler.readAllAnnotatorsData(True)
# annotatedDataHandler.calculateKappaScoreBetweenAnnotators()

modeAnnotatedData = annotatedDataHandler.calculateModeScoreBetweenAllAnnotators(True)
# print(modeAnnotatedData)
flatAnnotatedData = annotatedDataHandler.collapseDataSetTo1dArrayWithHeaders(modeAnnotatedData)
# print("len(flatAnnotatedData):", len(flatAnnotatedData))
dataset = {}
dataset['dev_x_index'], dataset['test_x_index'], dataset['dev_y'], dataset['test_y'] = train_test_split(
    range(len(flatAnnotatedData)), flatAnnotatedData, test_size=0.4)

minFive = 0.0
maxFive = 0.1
step = 0.1
kappa_score_at_thresholds = {}
while minFive < 1:
    maxFive = minFive + step
    while maxFive <= 1:
        annotatedDataHandler.log(['*'] * 80)
        thresholds = {annotatedDataHandler.class_unrelated: minFive, annotatedDataHandler.class_related: maxFive}
        annotatedDataHandler.log('Thresholds:' + str(thresholds))
        annotatedDataHandler.read_all_computed_data(read_headers=True, _thresholds=thresholds)

        # Read other models
        for baseline_method in annotatedDataHandler.notSynAndSem:
            # Check if there are any inconsistencies between the 2 lists
            annotatedDataHandler.compare1dLists(annotatedDataHandler.collapseDataSetTo1dArrayWithHeaders(
                annotatedDataHandler.computed_data[baseline_method]["0-0"]), flatAnnotatedData)

        # Read own models
        for own_method in annotatedDataHandler.computed_method:
            for syn_sem_threshold in annotatedDataHandler.result_indexes:
                # Check if there are any inconsistencies between the 2 lists
                annotatedDataHandler.compare1dLists(annotatedDataHandler.collapseDataSetTo1dArrayWithHeaders(
                    annotatedDataHandler.computed_data[own_method][syn_sem_threshold]), flatAnnotatedData)

        kappa_score_at_thresholds[str(thresholds)] = annotatedDataHandler.calculateKappaScoreBetweenComputedAndAnnotatedData(flatAnnotatedData)
        annotatedDataHandler.log(['Kappa Score for thresholds:', str(thresholds), " = ", kappa_score_at_thresholds[str(thresholds)]])
        # annotatedDataHandler.log(['-'] * 80)
        maxFive = maxFive + step
    minFive = minFive + step

annotatedDataHandler.log("Printing Final Results")
annotatedDataHandler.log([kappa_score_at_thresholds])
annotatedDataHandler.writeDetailedDictToCsv(kappa_score_at_thresholds, "kappa_score"+annotatedDataHandler.computational_iteration, type="kappa")
#
#

#
# annotatedDataHandler.log(['Development: Class0=%d, Class1=%d, Class2=%d' % (
#             len([t for t in dataset['dev_y'] if t == annotatedDataHandler.class_unrelated]),
#             len([t for t in dataset['dev_y'] if t == annotatedDataHandler.class_related]),
#             len([t for t in dataset['dev_y'] if t == annotatedDataHandler.class_equal]))], annotatedDataHandler.logTRACE)
#
# # print('Threshold Selection: Class0=%d, Class1=%d, Class2=%d' %
# #       (len([t for t in threshold_selection_y if t == "0.0"]),
# #        len([t for t in threshold_selection_y if t == "0.5"]),
# #        len([t for t in threshold_selection_y if t == self.class_equal])))
#
# annotatedDataHandler.log(['Test Selection: Class0=%d, Class1=%d, Class2=%d' % (
#     len([t for t in dataset['test_y'] if t == annotatedDataHandler.class_unrelated]),
#     len([t for t in dataset['test_y'] if t == annotatedDataHandler.class_related]),
#     len([t for t in dataset['test_y'] if t == annotatedDataHandler.class_equal]))], annotatedDataHandler.logTRACE)
#
#