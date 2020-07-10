import src.create_matrices as cm
import src.kmeans as km
import pandas as pd




# Loading data
log = pd.read_csv("./datasets/incident_log.csv").reset_index()

cm.create_matrices(log)
km.aplic_kmeans(log)
