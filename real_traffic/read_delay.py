import numpy as np

file_path = "predictions_delay_real_traces.npy"  # Remplacez par le chemin réel
data = np.load(file_path)


print("Contenu du fichier :")
print(data)


print("\nDimensions :", data.shape)
print("Type des données :", data.dtype)
