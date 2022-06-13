import json

from random import sample

from Utils.Data import get_distribution_scalars, scale, impute
import pandas as pd


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

import numpy as np
np.seterr(divide='ignore')

from numpy.random import seed


seed(7)


def main () :
    configs = json.load(open('Utils/Configuration.json', 'r'))
    features = configs['data']['features']

    outcome = configs['data']['classification_outcome']
    timeseries_path = configs['paths']['data_path']

    data = pd.read_csv(timeseries_path+"AggregatedSeries.csv")

    y = [int(x) for x in data[outcome].values]
    print("Outcome Distribution: ", get_distribution_scalars(y))
    number_patients = data.shape[0]
    print("number of patients: ", number_patients)
    negative_patients = data.loc[data[outcome] == 0]
    number_neg_patients = len(negative_patients)
    print("number of negative patients: ", number_neg_patients)

    positive_patients = data.loc[data[outcome] == 1]
    num_positive_patients = len(positive_patients)
    print("number of positive patients: ", num_positive_patients)

    ##ZI Change this in the real thing
    dataset_size = num_positive_patients*2
    #dataset_size = num_positive_patients/2
    prevalence_rates = [0.05, 0.1, 0.15, 0.2, 0.25,  0.3, 0.35, 0.4, 0.45, 0.5]
    print(set(positive_patients.Patient_id))
    for p in prevalence_rates:
        num_positives = int(p*dataset_size)
        positive_samples = sample(population = set(positive_patients.Patient_id), k = num_positives)
        num_negatives = int((1-p)*dataset_size)
        negative_samples = sample(population = set(negative_patients.Patient_id), k = num_negatives)
        print(" Prevelance: ", p, "Num Positives: ", len(positive_samples), "Num Negatives: ", len(negative_samples),
              "Total (Sanity Check): ", len(positive_samples)+len(negative_samples))

        positive_df =  data.loc[data.Patient_id.isin(positive_samples)]
        negative_df = data.loc[data.Patient_id.isin(negative_samples)]

        rate_df = positive_df.append(negative_df, ignore_index=True)
        ids = rate_df.Patient_id
        outcomes = rate_df[outcome]
        rate_df[outcome] = outcomes
        rate_df['Patient_id'] = ids
        rate_df.to_csv(timeseries_path+"Training/"+"TimeSeries"+str(p)+".csv", index=False)

if __name__ == '__main__' :
    main()
