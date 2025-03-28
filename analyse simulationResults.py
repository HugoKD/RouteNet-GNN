import pandas as pd

def parse_simulation_results(file_path,file_path_stability):
    """
    Analyse et extrait les informations du fichier simulationResults.txt.

    Parameters
    ----------
    file_path : str
        Chemin du fichier simulationResults.txt.

    Returns
    -------
    global_stats : dict
        Statistiques globales (nombre de paquets, pertes, délai moyen).
    agg_results : list of dict
        Liste des résultats agrégés pour chaque flux avec un identifiant.
    """
    global_stats = {}
    agg_results = []

    with open(file_path, 'r') as file:
        config_id = 1
        lines = file.read().split("\n")
        for line in lines:
            # Séparation des statistiques globales et des résultats détaillés
            global_part, detailed_part = line.split('|', 1)
            detailed_part = detailed_part[:-1]
            # Extraction des statistiques globales
            global_values = list(map(float, global_part.split(',')))
            if len(list(global_stats.keys())) == 0 :
                global_stats = {
                    "Metric":  ["TotalPackets_"+str(config_id), "TotalLosses_"+str(config_id), "AvgGlobalDelay_"+str(config_id)],
                    "Value_config_1": global_values,
                }
            else :
                global_stats["Value_config_"+str(config_id)] = global_values

            # Extraction des résultats par source-destination
            flux_id = 1  # Compteur pour identifier les flux
            for src_dst_entry in detailed_part.split(';'):
                # Vérifier si l'entrée est valide (peut contenir -1 pour ignorer)
                if src_dst_entry.startswith('-1'):
                    continue

                try:
                    try :
                        with open(file_path_stability, 'r') as file_stability:
                            sim_time = float(file_stability.read().split("\n")[0].split(";")[0])
                    except FileNotFoundError:
                        raise ValueError

                    # Découper les valeurs agrégées
                    aux_agg = list(map(float, src_dst_entry.split(',')))
                    agg_result = {'ConfigID': config_id,
                        "FluxID": flux_id,  'PktsDrop': aux_agg[2], "AvgDelay": aux_agg[3], "AvgLnDelay": aux_agg[4],
                        "p10": aux_agg[5], "p20": aux_agg[6], "p50": aux_agg[7], "p80": aux_agg[8],
                        "p90": aux_agg[9], "Jitter": aux_agg[10],
                        'AvgBw': aux_agg[0] * 1000,
                        'PktsGen': aux_agg[1],
                        'TotalPktsGen': aux_agg[1] * sim_time
                        }

                    agg_results.append(agg_result)
                    flux_id += 1  # Incrémenter l'identifiant du flux
                except ValueError:
                    print(f"Avertissement : Impossible d'analyser cette entrée : {src_dst_entry}")
            flux_id = 1
            config_id += 1

    return global_stats, agg_results


def save_to_excel(global_stats, agg_results, output_path="simulationResults.xlsx"):
    """
    Sauvegarde les résultats dans un fichier Excel

    Parameters
    ----------
    global_stats : dict
        Statistiques globales.
    agg_results : list of dict
        Résultats par source-destination.
    output_path : str
        Chemin du fichier Excel à générer.
    """
    # Créer des DataFrames pour les données
    global_df = pd.DataFrame(global_stats)
    agg_results_df = pd.DataFrame(agg_results)

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        # Combiner les données dans une seule feuille
        workbook = writer.book
        worksheet = workbook.add_worksheet("Results")
        writer.sheets["Results"] = worksheet

        # Écrire les statistiques globales
        global_df.to_excel(writer, sheet_name="Results", startrow=0, index=False, header=True)

        # Ajouter un séparateur visuel
        separator_row = len(global_df) + 2  # Deux lignes de séparation
        worksheet.write(separator_row - 1, 0, "Aggregated Results")  # Titre

        # Écrire les résultats agrégés
        agg_results_df.to_excel(
            writer, sheet_name="Results", startrow=separator_row, index=False, header=True
        )

    print(f"Fichier Excel généré : {output_path}")


file_path = "../data/TON23/real_traces/test/test/results_geant_1000_0_1/simulationResults.txt"  # Chemin du fichier à analyser
output_path = "simulationResults_output2.xlsx"  # Chemin du fichier Excel de sortie
file_path_stability = "../data/TON23/real_traces/test/test/results_geant_1000_0_1/stability.txt"
global_stats, agg_results = parse_simulation_results(file_path,file_path_stability = file_path_stability)
save_to_excel(global_stats, agg_results, output_path)
