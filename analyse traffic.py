import os

import pandas as pd

def parse_traffic_file(input_file, output_file):
    data_rows = []

    with open(input_file, 'r') as file:
        for i, line in enumerate(file):
            print(f"\n🔹 Ligne {i} brute : {line.strip()}")  # Voir la ligne brute

            parts = line.strip().split('|')  # Pour séparer timestamp et flux
            if len(parts) < 2:
                continue  # Si ligne mal formée

            timestamp = float(parts[0])  
            flows = parts[1].split(';')  # On sépare chaque flux

            for flow in flows:
                values = flow.split(',')
                print(f"Flow décomposé : {values}")  # Vérifier la séparation

                if len(values) < 5:  
                    print("Erreur : données incomplètes, passage au suivant.")
                    continue  

                time_dist = int(values[0])  
                EqLambda = ExpMaxFactor = SizeDistribution = AvgPacketSize = PacketSize1 = PacketSize2 = ToS = None

                # Extraction selon TimeDistribution
                try:
                    if time_dist == 6:  # TRACE_T
                        EqLambda = float(values[1])
                        index_offset = 2
                    elif time_dist in [0, 1, 2, 3]:  
                        EqLambda = float(values[1])
                        ExpMaxFactor = float(values[-1])  
                        index_offset = len(values)
                    elif time_dist in [4, 5]:  
                        EqLambda = float(values[1])
                        ExpMaxFactor = float(values[-1])
                        index_offset = len(values)
                    else:
                        index_offset = len(values)  
                except IndexError:
                    print("Erreur index TimeDist")
                    continue  

                # Extraction SizeDistribution
                try:
                    SizeDistribution = int(values[index_offset])
                    AvgPacketSize = float(values[index_offset + 1])
                    if SizeDistribution == 2:  
                        PacketSize1 = float(values[index_offset + 2])
                        PacketSize2 = float(values[index_offset + 3])
                except (IndexError, ValueError):
                    print("Erreur index SizeDist")
                    continue  

                # Extraction ToS
                try:
                    ToS = int(values[-1])  
                except (IndexError, ValueError):
                    ToS = None

                # Ajouter la ligne de données
                data_rows.append([
                    timestamp, time_dist, EqLambda, ExpMaxFactor,
                    SizeDistribution, AvgPacketSize, PacketSize1, PacketSize2, ToS
                ])

    # Création DataFrame
    df = pd.DataFrame(data_rows, columns=[
        "Timestamp", "TimeDistribution", "EqLambda", "ExpMaxFactor",
        "SizeDistribution", "AvgPacketSize", "PacketSize1", "PacketSize2", "ToS"
    ])

    print("\nAperçu des données avant export :")
    print(df.head())  

    df.to_csv(output_file, index=False, sep=';', na_rep='')
    print(f"Fichier exporté : {output_file}")

# Test du script
print(os.getcwd())
input_file = "../data/TON23/real_traces/test/test/results_geant_1000_0_1/traffic.txt"
output_file = "traffic_analysis.csv"
parse_traffic_file(input_file, output_file)
