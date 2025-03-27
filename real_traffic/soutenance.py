import matplotlib.pyplot as plt
import os
import json
import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot(features):
    iterations = features['iterations']
    dims = features['dim']

    val_loss, loss, labels = [], [], []

    for dim in dims:
        for iteration in iterations:
            path_dim = dim["path_state_dim"]
            link_dim = dim["link_state_dim"]
            queue_dim = dim["queue_state_dim"]

            file_prefix = f"{iteration}it_{path_dim}p_{link_dim}l_{queue_dim}q"
            ckpt_history = os.path.join('./ckpts', file_prefix)
            history_path = os.path.join(ckpt_history, f"history_{file_prefix}.json")

            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history = json.load(f)

                loss.append(history['loss'][-1])
                val_loss.append(history['val_loss'][-1])
                labels.append(file_prefix)
            else:
                print(f"❌ Fichier {history_path} introuvable, ignoré.")

    # --- Bar Plot ---
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, loss, width, label='Training Loss')
    plt.bar(x + width/2, val_loss, width, label='Validation Loss')

    plt.ylabel("Loss")
    plt.title("Comparaison des pertes finales par configuration")
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.show()


import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_for_iteration(features):
    iterations = features['iterations']
    dims = features['dim']

    val_loss = {it: [] for it in iterations}
    loss = {it: [] for it in iterations}
    labels = []

    for dim in dims:
        label = f"{dim['path_state_dim']}p_{dim['link_state_dim']}l_{dim['queue_state_dim']}q"
        labels.append(label)

        for iteration in iterations:
            path_dim = dim["path_state_dim"]
            link_dim = dim["link_state_dim"]
            queue_dim = dim["queue_state_dim"]

            file_prefix = f"{iteration}it_{path_dim}p_{link_dim}l_{queue_dim}q"
            ckpt_history = os.path.join('./ckpts', file_prefix)
            history_path = os.path.join(ckpt_history, f"history_{file_prefix}.json")

            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history = json.load(f)

                loss[iteration].append(history['loss'][-1])
                val_loss[iteration].append(history['val_loss'][-1])
            else:
                print(f"❌ Fichier {history_path} introuvable → ignoré")
                loss[iteration].append(None)
                val_loss[iteration].append(None)

    # --- Bar Plot ---
    x = np.arange(len(labels))
    width = 0.15

    plt.figure(figsize=(14, 6))

    for idx, iteration in enumerate(iterations):
        offset = (idx - len(iterations)/2) * width + width/2
        plt.bar(x + offset, val_loss[iteration], width, label=f'val_loss - {iteration} it')
        # Optionnel : afficher loss aussi
        # plt.bar(x + offset, loss[iteration], width, label=f'loss - {iteration} it', alpha=0.5)

    plt.ylabel("Validation Loss")
    plt.title("Validation loss par configuration (groupé par état dim)")
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_for_embedding_dim(features, fixed_iteration=15):
    dims = features['dim']

    val_loss = []
    loss = []
    labels = []

    for dim in dims:
        path_dim = dim["path_state_dim"]
        link_dim = dim["link_state_dim"]
        queue_dim = dim["queue_state_dim"]

        label = f"{path_dim}p_{link_dim}l_{queue_dim}q"
        labels.append(label)

        file_prefix = f"{fixed_iteration}it_{path_dim}p_{link_dim}l_{queue_dim}q"
        ckpt_history = os.path.join('./ckpts', file_prefix)
        history_path = os.path.join(ckpt_history, f"history_{file_prefix}.json")

        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)

            loss.append(history['loss'][-1])
            val_loss.append(history['val_loss'][-1])
        else:
            print(f"❌ Fichier {history_path} introuvable → ignoré")
            loss.append(None)
            val_loss.append(None)

    # --- Bar Plot ---
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, loss, width, label='Training Loss')
    plt.bar(x + width/2, val_loss, width, label='Validation Loss')

    plt.ylabel("Loss")
    plt.title(f"Influence des dimensions d'embedding (iteration = {fixed_iteration})")
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


features = {
    'iterations': [5, 8, 10, 15],
    "dim": [
        {'path_state_dim': 16, 'link_state_dim': 16, 'queue_state_dim': 16},
        {'path_state_dim': 32, 'link_state_dim': 32, 'queue_state_dim': 32},
        {'path_state_dim': 64, 'link_state_dim': 64, 'queue_state_dim': 64},
        {'path_state_dim': 128, 'link_state_dim': 128, 'queue_state_dim': 128},
    ]
}

plot_for_embedding_dim(features)