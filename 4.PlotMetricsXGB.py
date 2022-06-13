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
    prec = []
    ticks = []
    i = 1
    for p in prevalence_rates:
        input_file = output_path+"XGB_weighted"+str(p)+".json"
        metrics = json.load(open(input_file, 'r'))
        precision = metrics['test_precision_score_macro']
        x = list(itertools.repeat(p, len(precision)))
        prev.append(p)
        prec.append(precision)
        ticks.append(i)
        i = i +1

    plt.boxplot(prec)
    plt.xticks(ticks = ticks,  labels=prev)
    plt.savefig(output_path+"/Figures/XGBprecision.png")


    plt.plot()
    prev = []
    rec = []
    ticks = []
    i = 1
    for p in prevalence_rates:
        input_file = output_path+"XGB_weighted"+str(p)+".json"
        metrics = json.load(open(input_file, 'r'))
        recall = metrics['test_recall_score_macro']
        x = list(itertools.repeat(p, len(recall)))
        prev.append(p)
        rec.append(recall)
        ticks.append(i)
        i = i +1
    plt.clf()
    plt.boxplot(rec)
    plt.xticks(ticks = ticks,  labels=prev)
    plt.savefig(output_path+"/Figures/XGBrecall.png")


    plt.plot()
    prev = []
    aucs = []
    ticks = []
    i = 1
    for p in prevalence_rates:
        input_file = output_path+"XGB_weighted"+str(p)+".json"
        metrics = json.load(open(input_file, 'r'))
        auc = metrics['test_auc_score']
        x = list(itertools.repeat(p, len(recall)))
        prev.append(p)
        aucs.append(auc)
        ticks.append(i)
        i = i +1
    plt.clf()
    plt.boxplot(aucs)
    plt.xticks(ticks = ticks,  labels=prev)
    plt.savefig(output_path+"/Figures/XGBauc.png")

    plt.plot()
    prev = []
    aucs = []
    ticks = []
    i = 1
    for p in prevalence_rates :
        input_file = output_path + "XGB_weighted" + str(p) + ".json"
        metrics = json.load(open(input_file, 'r'))
        auc = metrics['test_pr_auc_score']
        x = list(itertools.repeat(p, len(recall)))
        prev.append(p)
        aucs.append(auc)
        ticks.append(i)
        i = i + 1
    plt.clf()
    plt.boxplot(aucs)
    plt.xticks(ticks=ticks, labels=prev)
    plt.savefig(output_path + "/Figures/XGBprauc.png")
if __name__ == '__main__' :
    main()
