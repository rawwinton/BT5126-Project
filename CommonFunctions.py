from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, PrecisionRecallDisplay, precision_score, \
    average_precision_score, auc, precision_recall_curve, make_scorer, SCORERS
import matplotlib.pyplot as plt
import seaborn as sn
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
    accuracy, averagePrecision, aucScore, prAucScore = GetSummary(y_train, y_trainPred, y_trainProba)
    # summarize scores
    print('Accuracy score on Train dataset : ', accuracy)
    print('Average precision score on Train dataset: %.3f' % averagePrecision)
    print('ROC AUC score on Train dataset: %.3f' % aucScore)
    print('PR AUC score on Train dataset: %.3f' % prAucScore)
    print('')

    accuracy, averagePrecision, aucScore, prAucScore = GetSummary(y_test, y_pred, y_proba)
    # summarize scores
    print('Accuracy score on test dataset : ', accuracy)
    print('Average precision score on test dataset: %.3f' % averagePrecision)
    print('ROC AUC score on test dataset: %.3f' % aucScore)
    print('PR AUC score on test dataset: %.3f' % prAucScore)


def GetSummary2(y_train, y_trainPred, y_trainProba, y_test, y_pred, y_proba):
    return GetSummary(y_train, y_trainPred, y_trainProba)+GetSummary(y_test, y_pred, y_proba)


def GetSummary(y_test, y_pred, y_proba):
    accuracy = accuracy_score(y_test, y_pred)
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

    return [accuracy, averagePrecision, aucScore, prAucScore]


def GetStackingSummary(clf, X_train, y_train, X_test, y_test):

    return [accuracy, averagePrecision, aucScore, prAucScore]


def roc_auc(clf, x, y_true):
    return roc_auc_score(y_true=y_true, y_score=clf.predict_proba(x), multi_class="ovr")
