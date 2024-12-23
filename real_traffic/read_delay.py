import numpy as np

file_path = "predictions_delay_real_traces.npy"  # Remplacez par le chemin réel
data = np.load(file_path)


print("Contenu du fichier :")
print(data)


print("\nDimensions :", data.shape) #est égal au nombre de link to path sur toutes les config du sample (plusieurs network possible pour un meme sample)

