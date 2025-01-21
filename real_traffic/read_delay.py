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
link_file = tar.extractfile(dir_info.name+"/linkUsage.txt")


################ Simulation results
text  = results_file.readline().decode()[:-2]
first_params = text.split('|')[0].split(',')
first_params = list(map(float,first_params ))
second_params = text.split(';')[10].split(":")
L = text.split(';')
print(second_params)
print(len(L))
'''
R = []
for x in r :
    x = x.split(',')
    R.append([float(elmt) for elmt in x])'''
########
traffic_line = traffic_file.readline().decode()[:-1]
ptr = traffic_line.find('|')
t = traffic_line[ptr+1:].split(';')
maxAvgLambda = float(traffic_line[:ptr])
######*#################
'''
link_file = open("../data/TON23/real_traces/test/test/results_geant_1000_0_1/linkUsage.txt","r")
i,j = 0,1
for line in link_file:
    l  = line.split(";")
    while i <= 484 :
        if l[i] != str(-1) and len(l[i]) > 1 :
            print(i,type(l[i]),l[i].split(":"))
            if len(l[i].split(":")) < 2 : print('')
        #j += 1
        i+=1
    print(len(l))
    print(j)
    i,j = 0,1
'''
