import csv


def analyze_link_usage(input_file, output_file):
    with open(input_file, 'r') as file:
        data = file.read().strip()

    # Split configurations based on double newline (each configuration should be separated)
    configurations = data.split("\n")

    processed_data = []

    for config_id, config in enumerate(configurations):
        rows = config.strip().split(";")
        if len(rows[-1]) == 0:
            rows = rows[:-1]  # Remove trailing empty entries

        num_links = int(len(rows) ** 0.5)  # Compute network size
        print(f"Processing configuration {config_id}: {len(rows)} entries, estimated num_links: {num_links}")

        for i in range(num_links):
            for j in range(num_links):
                try:
                    index = i * num_links + j
                    if index >= len(rows):
                        print(f"Index out of bounds: {index}, max size: {len(rows)}")
                        continue

                    params_lst = rows[index].split(":")
                    params = params_lst[0].split(",")

                    if params[0] == "-1":
                        continue  # Ignore missing links

                    # Extract link-level metrics
                    link_utilization = float(params[0]) if params[0] else None
                    link_losses = float(params[1]) if len(params) > 1 else None
                    link_avg_packet_size = float(params[2]) if len(params) > 2 else None

                    qos_queues = []
                    for qos_params in params_lst[1:]:
                        queue_params = qos_params.split(",")
                        if len(queue_params) < 5:
                            print(f"Malformed QoS queue: {queue_params}")
                            continue

                        qos_queues.append({
                            "queue_utilization": float(queue_params[0]) if queue_params[0] else None,
                            "queue_losses": float(queue_params[1]) if queue_params[1] else None,
                            "queue_avg_occupancy": float(queue_params[2]) if queue_params[2] else None,
                            "queue_max_occupancy": float(queue_params[3]) if queue_params[3] else None,
                            "queue_avg_packet_size": float(queue_params[4]) if queue_params[4] else None,
                        })

                    processed_data.append({
                        "config_id": config_id,
                        "src_node": i,
                        "dst_node": j,
                        "link_utilization": link_utilization,
                        "link_losses": link_losses,
                        "link_avg_packet_size": link_avg_packet_size,
                        "qos_queues": qos_queues,
                    })

                except Exception as e:
                    print(f"Error processing link {i}-{j} in config {config_id} (index {index}): {e}")

    # Write to a CSV file, handling multiple QoS queues properly
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            "config_id", "src_node", "dst_node",
            "link_utilization", "link_losses", "link_avg_packet_size",
            "queue_utilization", "queue_losses",
            "queue_avg_occupancy", "queue_max_occupancy", "queue_avg_packet_size"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()

        for entry in processed_data:
            for queue in entry["qos_queues"]:
                writer.writerow({
                    "config_id": entry["config_id"],
                    "src_node": entry["src_node"],
                    "dst_node": entry["dst_node"],
                    "link_utilization": entry["link_utilization"],
                    "link_losses": entry["link_losses"],
                    "link_avg_packet_size": entry["link_avg_packet_size"],
                    "queue_utilization": queue["queue_utilization"],
                    "queue_losses": queue["queue_losses"],
                    "queue_avg_occupancy": queue["queue_avg_occupancy"],
                    "queue_max_occupancy": queue["queue_max_occupancy"],
                    "queue_avg_packet_size": queue["queue_avg_packet_size"],
                })


if __name__ == "__main__":
    analyze_link_usage("../data/TON23/real_traces/test/test/results_geant_1000_0_1/linkUsage.txt",
                       "linkUsage_output.csv")
