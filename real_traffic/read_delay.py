import math
import numpy as np
import os,tarfile
import matplotlib.pyplot as plt
'''
file_path = "predictions_delay_real_traces.npy"  # Remplacez par le chemin réel
data = np.load(file_path)


print("Contenu du fichier :")
print(data)


print("\nDimensions :", data.shape) #est égal au nombre de link to path sur toutes les config du sample (plusieurs network possible pour un meme sample)
'''

#lire et decoder les inputs :
root = "../data/TON23/real_traces/test/test"
tar = tarfile.open(os.path.join(root, 'results_geant_1000_0_1.tar.gz'), 'r:gz')
dir_info = tar.next()
results_file = tar.extractfile(dir_info.name+"/simulationResults.txt")
traffic_file = tar.extractfile(dir_info.name+"/traffic.txt")
status_file = tar.extractfile(dir_info.name+"/stability.txt")
input_files = tar.extractfile(dir_info.name+"/input_files.txt")


################ Simulation results
first_params = results_file.readline().decode()[:-2].split('|')[0].split(',')
first_params = list(map(float,first_params ))
r = results_file.readline().decode()[:-2][results_file.readline().decode()[:-2].find('|')+1:].split(';')
########
traffic_line = traffic_file.readline().decode()[:-1]
ptr = traffic_line.find('|')
t = traffic_line[ptr+1:].split(';')
maxAvgLambda = float(traffic_line[:ptr])
######*#################
