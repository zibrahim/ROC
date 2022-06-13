from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, \
    brier_score_loss, auc, confusion_matrix, cohen_kappa_score, roc_curve, precision_recall_curve, roc_auc_score, \
    make_scorer


def confusion_matrix_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return {'tn': cm[0, 0], 'fp': cm[0, 1],
            'fn': cm[1, 0], 'tp': cm[1, 1]}


def confusion_matrix_scorer_tn(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return cm[0, 0]

def confusion_matrix_scorer_fp(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return cm[0, 1]

def confusion_matrix_scorer_fn(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return cm[1, 0]

def confusion_matrix_scorer_tp(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return cm[1, 1]

def pr_auc_score(y_true, y_score):
    """
    Generates the Area Under the Curve for precision and recall.
    """
    precision, recall, thresholds = \
        precision_recall_curve(y_true, y_score)
    return auc(recall, precision)

def pr_auc_score_alt(x, y):
    score=roc_auc_score(x, y)
    return score
scoring_list = {'f1_score_macro' : make_scorer(f1_score, average='macro'),
                'accuracy_score' : make_scorer(accuracy_score),
                'recall_score_macro' : make_scorer(recall_score, average='macro'),
                'precision_score_macro' : make_scorer(precision_score, average='macro', zero_division=0),
                'auc_score' : make_scorer(roc_auc_score),
                'pr_auc_score': make_scorer(pr_auc_score),
                'confusion_tp' : confusion_matrix_scorer_tp,
                'confusion_fp' : confusion_matrix_scorer_fp,
                'confusion_tn' : confusion_matrix_scorer_tn,
                'confusion_fn' : confusion_matrix_scorer_fn
                }

def performance_metrics(testing_y, y_pred_binary,yhat):
    F1Macro = f1_score(testing_y, y_pred_binary, average='macro')
    Accuracy = accuracy_score(testing_y, y_pred_binary)
    BS = brier_score_loss(testing_y, yhat)
    CK = cohen_kappa_score(testing_y, y_pred_binary)
    ClassificationReport = classification_report(testing_y, y_pred_binary)
    CM = confusion_matrix(testing_y, y_pred_binary)

    TN, FP, FN, TP = CM.ravel()
    precision_rt, recall_rt, threshold_rt = precision_recall_curve(testing_y,
                                                                   yhat)
    pr_auc = auc(recall_rt, precision_rt)
    false_pos_rate, true_pos_rate, thresholds = roc_curve(testing_y, yhat)
    roc_auc = auc(false_pos_rate, true_pos_rate, )
    roc_auc_using_builtin = roc_auc_score(testing_y, yhat)


    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)
    FPR = FP/(FP+TN)
    FNR = FN/(FN+TP)

    performance_row = {
        "F1-Macro" : F1Macro,
        "Accuracy" : Accuracy,
        "ClassificationReport" : ClassificationReport,
        "PPV":PPV,
        "NPV":NPV,
        "Brier's":BS,
        "Cohen's": CK,
        "TP": TP,
        "TN":TN,
        "FP":FP,
        "FN":FN,
        "PR-AUC-Trapezoid":pr_auc,
        "ROC-AUC-Trapezoid":roc_auc,
        "ROC-AUC-Built-in":roc_auc_using_builtin,
        "TPR": TPR,
        "TNR": TNR,
        "FPR" : FPR,
        "FNR": FNR

    }

    return performance_row
