import csv
import os
import tqdm
import pandas as pd

from pm4py.objects.log.importer.csv import importer as csv_importer
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.petrinet import visualizer as petri_visual
from pm4py.evaluation.generalization import evaluator as calc_generaliz
from pm4py.evaluation.precision import evaluator as calc_precision
from pm4py.evaluation.replay_fitness import evaluator as calc_fitness
from pm4py.evaluation.simplicity import evaluator as calc_simplic

# Sublogs to log format
def data2log(data):
    data['activity'] = ('incident_state' + '-' + data['incident_state'] + '--' +
                        'category' + '-' + data['category'] + '--' +
                        'priority' + '-' + data['priority'])

    data['sys_updated_on'] = pd.to_datetime(data['sys_updated_on'], dayfirst=True)
    log = data.sort_values(by=['sys_updated_on'])
    return log[['number', 'activity']].dropna()

def check_log_path(log_path):
    if not os.path.exists(log_path):
        data_path = "../datasets/incident_log.csv"
        data = pd.read_csv(data_path)
        log = data2log(data)
        log.to_csv(log_path, index=False)
        log.shape

# Model and metrics with all data
log_path = "../datasets/incident_log-logformat.csv"
check_log_path(log_path)

parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'number'}
log = log_converter.apply(log,parameters=parameters)

parameters = {inductive_miner.Variants.DFG_BASED.value.Parameters.CASE_ID_KEY: 'number',
             inductive_miner.Variants.DFG_BASED.value.Parameters.ACTIVITY_KEY: 'activity',}
petrinet_res = inductive_miner.apply(log, parameters=parameters)

#fitness = calc_fitness.apply(log, *petrinet_res, parameters=parameters)
#print('Conformidade',round(fitness['average_trace_fitness'],4))
precision = calc_precision.apply(log, *petrinet_res, parameters=parameters)
print('Precisao', round(precision,4))
simplic = calc_simplic.apply(petrinet_res[0], parameters=parameters)
print('Simplicidade', round(simplic,4))
generaliz = calc_generaliz.apply(log, *petrinet_res, parameters=parameters)
print('Generalização', round(generaliz,4))

# Precisao 0.1023
# Simplicidade 0.5802
# Generalização 0.575


# ----------------------------------------------------------------------------
# Metrics for kmeans
# Generate a csv file with the metrics for all the kmeans results

def attributes_selection(filename):
    if 'specialist' in filename:
        return ['number', 'incident_state', 'priority', 'category', 'sys_updated_on']
    if 'alg1' in filename:
        return ['number', 'caller_id', 'assigned_to', 'sys_updated_on']
    return ['number', 'incident_state', 'location', 'sys_updated_on']


sublogs_dir = "kmeans_results/sublogs/"
sublogs = [filename for filename in os.listdir(sublogs_dir) if 'csv' in filename]

generaliz_mean = []
precision_mean = []
fitness_mean = []
simplic_mean = []

with open('kmeans_results/pm_metrics.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(['data', 'generalization', 'precision', 'fitness_perc', 'fitness_avg', 'simplic'])
    for sublog_filename in tqdm.tqdm_notebook(sublogs):
        parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'number'}
        data = pd.read_csv(sublogs_dir + sublog_filename)
        log = data2log(data)
        log = log_converter.apply(log, parameters=parameters)

        parameters = {inductive_miner.Variants.DFG_BASED.value.Parameters.CASE_ID_KEY: 'number',
                      inductive_miner.Variants.DFG_BASED.value.Parameters.ACTIVITY_KEY: 'activity', }
        petrinet_res = inductive_miner.apply(log, parameters=parameters)

        fitness = calc_fitness.apply(log, *petrinet_res, parameters=parameters)
        precision = calc_precision.apply(log, *petrinet_res, parameters=parameters)
        simplic = calc_simplic.apply(petrinet_res[0], parameters=parameters)
        generaliz = calc_generaliz.apply(log, *petrinet_res, parameters=parameters)

        generaliz_mean.append(generaliz)
        precision_mean.append(precision)
        fitness_mean.append(fitness)
        simplic_mean.append(simplic)

        spamwriter.writerow([sublog_filename.replace('.csv', ''), round(generaliz, 4), round(precision, 4),
                             round(fitness['percFitTraces'], 4), round(fitness['averageFitness'], 4),
                             round(simplic, 4)])

