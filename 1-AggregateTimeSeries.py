import pandas as pd
pd.set_option('display.max_rows', None)

import json

from Utils.CleanTimeSeries import remove_alpha, remove_nacolumns
from Utils.Dictionary import aggregation
def main():

    configs = json.load(open('Utils/Configuration.json', 'r'))
    data_path = configs['paths']['data_path']

    time_series = pd.read_csv(data_path+"sepsis_challenge.csv")

    time_series = remove_alpha(time_series)
    time_series = remove_nacolumns(time_series)

    aggregated_data = time_series.groupby('Patient_id', as_index=False).agg(func=aggregation)

    aggregated_data.dropna(axis=1, how='all', inplace=True)
    col_names = [x[0] for x in aggregated_data.columns]
    aggregated_data.columns = col_names
    aggregated_data.to_csv(data_path+"AggregatedSeries.csv", index=False)

if __name__ == "__main__" :
    main()