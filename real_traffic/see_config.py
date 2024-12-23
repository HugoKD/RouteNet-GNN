import pandas as pd
import os

path_dir= f'../data/TON23/real_traces/test/test/results_geant_1000_0_1'

lst_dir = os.listdir(path_dir)
dfs = {}
for tm in lst_dir :
    print(tm)
    dfs[tm] = pd.read_csv(path_dir+ '/' + tm, delimiter=';')  # if it's tab-delimited

print(dfs[lst_dir[2]].head())
