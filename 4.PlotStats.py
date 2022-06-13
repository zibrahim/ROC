import itertools
import json

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

np.seterr(divide='ignore')

from numpy.random import seed
from sklearn.ensemble import RandomForestClassifier

seed(7)

rcParams['figure.figsize'] = 8, 6
LABELS = ["0", "1"]

def main () :
    configs = json.load(open('Utils/Configuration.json', 'r'))
    data_path = configs['paths']['data_path']
    timeseries_path = data_path+"Training/"
    output_path = data_path+"Output/"

    features = configs['data']['features']
    outcome = configs['data']['classification_outcome']

    prevalence_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    plt.plot()
    prev = []
    tps = []
    fps = []
    tns = []
    fns = []
    tprs = []
    fprs = []
    fnrs = []
    tnrs = []

    ppvs = []
    npvs = []
    ticks = []
    i = 1
    for p in prevalence_rates:
        input_file = output_path+"SVM"+str(p)+".json"
        metrics = json.load(open(input_file, 'r'))
        tp = sum(metrics['test_confusion_tp'])/len(metrics['test_confusion_tp'])
        fp = sum(metrics['test_confusion_fp'])/len(metrics['test_confusion_fp'])
        tn = sum(metrics['test_confusion_tn'])/len(metrics['test_confusion_tn'])
        fn = sum(metrics['test_confusion_fn'])/len(metrics['test_confusion_fn'])

        print(tp, fp, tn, fn)
        tpr = tp/(tp+fn)
        fpr = fp/(tn+fp)
        fnr = fn/(tp+fn)
        tnr = tn/(tn+fp)

        ppv = tp/(tp+fp)
        npv = tn/(tn+fn)

        tprs.append(tpr)
        fprs.append(fpr)
        fnrs.append(fnr)
        tnrs.append(tnr)

        tps.append(tp)
        fps.append(fp)
        tns.append(tn)
        fns.append(fn)

        ppvs.append(ppv)
        npvs.append(npv)
        ticks.append(i)
        i = i +1

    plt.plot(tprs)
    plt.plot(fprs)
    plt.plot(fnrs)
    plt.plot(tnrs)
    plt.plot(ppvs)
    plt.plot(npvs)
    plt.legend(['TPR', 'FPR', 'FNR', 'TNR', 'PPV', 'NPV'])
    plt.xticks(ticks = ticks,  labels=prev)
    plt.savefig(output_path+"/Figures/SVMPositives.png")



if __name__ == '__main__' :
    main()
