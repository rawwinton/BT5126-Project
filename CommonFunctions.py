from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, PrecisionRecallDisplay, precision_score, \
    average_precision_score, auc, precision_recall_curve, make_scorer, SCORERS, recall_score,f1_score
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd


def aucrocCurve(y_test, y_proba):
    # roc curve for classes
    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh = {}
    n_class = 4
    for i in range(n_class):
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_proba[:, i], pos_label=i)

    # plotting
    plt.plot(fpr[0], tpr[0], linestyle='-', color='orange', label='Wait Level 0 vs Rest')
    plt.plot(fpr[1], tpr[1], linestyle='-', color='green', label='Wait Level 1 vs Rest')
    plt.plot(fpr[2], tpr[2], linestyle='-', color='blue', label='Wait Level 2 vs Rest')
    plt.plot(fpr[3], tpr[3], linestyle='-', color='yellow', label='Wait Level 3 vs Rest')
    plt.title('Wait Levels AUC-ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')


def aucprCurve(y_test, y_proba):
    # roc curve for classes
    precision = {}
    recall = {}
    thresh = {}
    n_class = 4
    for i in range(n_class):
        precision[i], recall[i], thresh[i] = precision_recall_curve(y_test, y_proba[:, i], pos_label=i)

    # plotting
    plt.plot(recall[0], precision[0], linestyle='-', color='orange', label='Wait Level 0 vs Rest')
    plt.plot(recall[1], precision[1], linestyle='-', color='green', label='Wait Level 1 vs Rest')
    plt.plot(recall[2], precision[2], linestyle='-', color='blue', label='Wait Level 2 vs Rest')
    plt.plot(recall[3], precision[3], linestyle='-', color='yellow', label='Wait Level 3 vs Rest')

    plt.title('Multiclass PR-AUC curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')


def confusionMatrix(y_test, y_pred):
    df = pd.DataFrame({'y_Actual': y_test, 'y_Predicted': y_pred})
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True, fmt='g')
    plt.show()


def PrintSummary(y_train, y_trainPred, y_trainProba, y_test, y_pred, y_proba):
    accuracy, averagePrecision, aucScore, prAucScore, averageRecall, averageF1Score = GetSummary(y_train, y_trainPred, y_trainProba)
    # summarize scores
    print('Accuracy score on Train dataset : ', accuracy)
    print('Average precision score on Train dataset: %.3f' % averagePrecision)
    print('Weighted recall score on Train dataset: %.3f' % averageRecall)
    print('Weighted F1 score on Train dataset: %.3f' % averageF1Score)
    print('ROC AUC score on Train dataset: %.3f' % aucScore)
    print('PR AUC score on Train dataset: %.3f' % prAucScore)
    print('')

    accuracy, averagePrecision, aucScore, prAucScore, averageRecall, averageF1Score = GetSummary(y_test, y_pred, y_proba)
    # summarize scores
    print('Accuracy score on test dataset : ', accuracy)
    print('Average precision score on test dataset: %.3f' % averagePrecision)
    print('Weighted recall score on test dataset: %.3f' % averageRecall)
    print('Weighted F1 score on test dataset: %.3f' % averageF1Score)
    print('ROC AUC score on test dataset: %.3f' % aucScore)
    print('PR AUC score on test dataset: %.3f' % prAucScore)


def GetSummary2(y_train, y_trainPred, y_trainProba, y_test, y_pred, y_proba):
    return GetSummary(y_train, y_trainPred, y_trainProba) + GetSummary(y_test, y_pred, y_proba)


def GetSummary(y_test, y_pred, y_proba):
    accuracy = accuracy_score(y_test, y_pred)
    averageRecall = recall_score(y_test, y_pred, average="weighted")
    averageF1Score = f1_score(y_test, y_pred, average="weighted")
    averagePrecision = 0
    prAucScore = 0
    if len(y_proba.shape) == 2:
        aucScore = roc_auc_score(y_test, y_proba, multi_class="ovr")
        for i in range(y_proba.shape[1]):
            averagePrecision += average_precision_score(y_test == i, y_proba[:, i])
            precision, recall, thresholds = precision_recall_curve(y_test == i, y_proba[:, i], pos_label=1)
            prAucScore += auc(recall, precision)
        averagePrecision = averagePrecision / y_proba.shape[1]
        prAucScore = prAucScore / y_proba.shape[1]

    else:
        aucScore = roc_auc_score(y_test, y_proba)
        averagePrecision += average_precision_score(y_test, y_proba)
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba, pos_label=1)
        prAucScore += auc(recall, precision)

    return [accuracy, averagePrecision, aucScore, prAucScore, averageRecall, averageF1Score]


def plotStackingGraph(preprocessingName, finalEstimators, finalPipe, finalParam, X_train, y_train, X_test, y_test):
    for estimator in finalEstimators:
        modelParam = {}
        for param, grid in finalParam.items():
            if param.startswith(estimator[0]):
                modelParam[param[len(estimator[0]) + 2:]] = grid
            elif param.startswith(preprocessingName):
                modelParam[param] = grid
        # print(estimator[1].get_params().keys())
        estimator[1].set_params(**modelParam)
        estimator[1].fit(X_train, y_train)
    finalPipe.set_params(**finalParam)
    finalPipe.fit(X_train, y_train)

    resultList = []
    trainProbaList = []
    testProbaList = []
    trainPredList = []
    testPredList = []
    i = 0
    for estimator in finalEstimators:
        resultList.append((estimator[0], GetSummary2(y_train, estimator[1].predict(X_train),
                                                     estimator[1].predict_proba(X_train), y_test,
                                                     estimator[1].predict(X_test),
                                                     estimator[1].predict_proba(X_test))))
        trainProbaList.append((estimator[0], estimator[1].predict_proba(X_train)))
        testProbaList.append((estimator[0], estimator[1].predict_proba(X_test)))
        trainPredList.append((estimator[0], estimator[1].predict(X_train)))
        testPredList.append((estimator[0], estimator[1].predict(X_test)))
        i += 1
    resultList.append(("Stack Classifier", GetSummary2(y_train, finalPipe.predict(X_train),
                                                       finalPipe.predict_proba(X_train), y_test,
                                                       finalPipe.predict(X_test),
                                                       finalPipe.predict_proba(X_test))))
    trainProbaList.append(("Stack Classifier", finalPipe.predict_proba(X_train)))
    testProbaList.append(("Stack Classifier", finalPipe.predict_proba(X_test)))
    trainPredList.append(("Stack Classifier", finalPipe.predict(X_train)))
    testPredList.append(("Stack Classifier", finalPipe.predict(X_test)))

    barWidth = 0.07
    # set heights of bars
    names = []
    bars = [[], [], [], [], [], [], [], [], [], [], [], []]
    for name, result in resultList:
        names.append(name)
        bars[0].append(result[0])
        bars[1].append(result[1])
        bars[2].append(result[2])
        bars[3].append(result[3])
        bars[4].append(result[4])
        bars[5].append(result[5])
        bars[6].append(result[6])
        bars[7].append(result[7])
        bars[8].append(result[8])
        bars[9].append(result[9])
        bars[10].append(result[10])
        bars[11].append(result[11])

    # Set position of bar on X axis
    r1 = np.arange(len(bars[0]))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    r5 = [x + barWidth for x in r4]
    r6 = [x + barWidth for x in r5]
    r7 = [x + barWidth for x in r6]
    r8 = [x + barWidth for x in r7]
    r9 = [x + barWidth for x in r8]
    r10 = [x + barWidth for x in r9]
    r11 = [x + barWidth for x in r10]
    r12 = [x + barWidth for x in r11]

    # Make the plot
    finalPlotLabels = ['Acc(Train)', 'precision(Train)', 'ROC-AUC (Train)', 'PR-AUC(Train)', 'Average Recall(Train)', 'F1 Score(Train)',
                       'Acc(Test)', 'precision(Test)', 'ROC-AUC (Test)', 'PR-AUC(Test)', 'Average Recall(Test)', 'F1 Score(Test)']
    colours = ['red', 'orange', 'yellow', 'green', 'blue', '#558f2d', '#2d9f5e', '#2db95e','Magenta','Cyan','gold','purple','tomato']
    r = [r1, r2, r3, r4, r5, r6, r7, r8,r9,r10,r11,r12]
    i = 0

    f = plt.figure()
    f.set_figwidth(20)
    for finalPlotLabel in finalPlotLabels:
        plt.bar(r[i], bars[i], color=colours[i], width=barWidth, edgecolor='white', label=finalPlotLabel)
        i += 1

    # Add xticks on the middle of the group bars
    plt.xlabel('Method', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars[0]))], names)

    # Create legend & Show graphic
    plt.legend()
    plt.show()

    fig = plt.figure()
    fig.set_figwidth(20)
    fig.set_figheight(27)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    # fig.subplots_adjust(hspace=0.4, wspace=0.4)
    i = 1
    for name, proba in trainProbaList:
        ax = fig.add_subplot(6, 5, i)
        ax = plotAUC(ax, y_train, proba)
        ax.set_title(name + ' AUC-ROC (Train)')
        i += 1

    for name, proba in testProbaList:
        ax = fig.add_subplot(6, 5, i)
        ax = plotAUC(ax, y_test, proba)
        ax.set_title(name + ' AUC-ROC (Test)')
        i += 1

    for name, proba in trainProbaList:
        ax = fig.add_subplot(6, 5, i)
        ax = plotPRAUC(ax, y_train, proba)
        ax.set_title(name + ' PR-AUC (Train)')
        i += 1

    for name, proba in testProbaList:
        ax = fig.add_subplot(6, 5, i)
        ax = plotPRAUC(ax, y_test, proba)
        ax.set_title(name + ' PR-AUC (Test)')
        i += 1

    for name, proba in trainPredList:
        ax = fig.add_subplot(6, 5, i)
        ax = plotConfusionMatrix(ax, y_train, proba)
        ax.set_title(name + ' CM (Train)')
        i += 1

    for name, proba in testPredList:
        ax = fig.add_subplot(6, 5, i)
        ax = plotConfusionMatrix(ax, y_test, proba)
        ax.set_title(name + ' CM (Test)')
        i += 1

    plt.show()
    print("Summary Stats:")
    return pd.DataFrame(bars, index=finalPlotLabels, columns=names)


def roc_auc(clf, x, y_true):
    return roc_auc_score(y_true=y_true, y_score=clf.predict_proba(x), multi_class="ovr")


def plotAUC(ax, y_test, y_proba):
    colour = ['red', 'orange', 'yellow', 'green', 'blue']
    for i in range(y_proba.shape[1]):
        fpr, tpr, thresh = roc_curve(y_test, y_proba[:, i], pos_label=i)
        aucScore = roc_auc_score(y_test == i, y_proba[:, i])
        ax.plot(fpr, tpr, linestyle='-', color=colour[i], label='Wait Level ' + str(i) + ' (%.3f)' % aucScore)

    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
    # plotting
    ax.set_title('Wait Levels AUC-ROC')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive rate')
    ax.legend(loc='best')
    return ax


def plotPRAUC(ax, y_test, y_proba):
    colour = ['red', 'orange', 'yellow', 'green', 'blue']
    for i in range(y_proba.shape[1]):
        precision, recall, thresh = precision_recall_curve(y_test, y_proba[:, i], pos_label=i)
        prAucScore = auc(recall, precision)
        ax.plot(recall, precision, linestyle='-', color=colour[i],
                label='Wait Level ' + str(i) + ' (%.3f)' % prAucScore)

    ax.set_title('Multiclass PR-AUC curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='best')
    return ax


def plotConfusionMatrix(ax, y_test, y_pred):
    df = pd.DataFrame({'y_Actual': y_test, 'y_Predicted': y_pred})
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True, fmt='g', ax=ax)
    return ax
