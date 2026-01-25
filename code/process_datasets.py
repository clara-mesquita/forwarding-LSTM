import os
from itertools import product

import pandas as pd


def get_all_links():
    states = [
        "ac",
        "al",
        "am",
        "ap",
        "ba",
        "ce",
        "df",
        "es",
        "go",
        "ma",
        "mg",
        "ms",
        "mt",
        "pa",
        "pb",
        "pe",
        "pi",
        "pr",
        "rj",
        "rn",
        "ro",
        "rr",
        "rs",
        "sc",
        "se",
        "sp",
        "to",
    ]
    all_links = list(product(states, repeat=2))
    return [f"{origem}-{destino}" for origem, destino in all_links if origem != destino]


def process_and_save_datasets():
    base_dir = "datasets/datasets-todos-links"
    year_folders = ["datasets-2024", "datasets-2025", "datasets-2025-2"]
    output_dir_base = "datasets/treated"

    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)

    links = get_all_links()

    for protocol in ["cubic", "bbr"]:
        output_dir_protocol = os.path.join(output_dir_base, protocol)
        if not os.path.exists(output_dir_protocol):
            os.makedirs(output_dir_protocol)

        for link in links:
            dfs = []
            for year_folder in year_folders:
                # Caminho pode variar um pouco, então vamos construir dinamicamente
                for data_type in [
                    "datasets vazao",
                    "datasets atraso",
                    "datasets perda",
                ]:
                    search_path = os.path.join(
                        base_dir, year_folder, "originals", data_type, protocol
                    )
                    if not os.path.exists(search_path):
                        search_path = os.path.join(
                            base_dir, year_folder, "originals", data_type
                        )

                    if os.path.exists(search_path):
                        for root, _, files in os.walk(search_path):
                            for file in files:
                                if link in file and file.startswith(protocol):
                                    file_path = os.path.join(root, file)
                                    try:
                                        df = pd.read_csv(file_path)
                                        dfs.append(df)
                                    except Exception as e:
                                        print(f"Erro ao ler o arquivo {file_path}: {e}")

            if not dfs:
                continue

            combined_df = pd.concat(dfs, ignore_index=True)

            # Assegura que a coluna de timestamp existe e está no formato correto
            if "Timestamp" not in combined_df.columns:
                print(
                    f"Alerta: 'Timestamp' não encontrado para o link {link} e protocolo {protocol}. Pulando."
                )
                continue

            combined_df["Timestamp"] = pd.to_datetime(combined_df["Timestamp"])
            combined_df = combined_df.sort_values(by="Timestamp").reset_index(drop=True)

            # Divide em intervalos de 6 horas
            combined_df.set_index("Timestamp", inplace=True)
            intervals = combined_df.resample("6H")

            for i, (name, interval_df) in enumerate(intervals):
                if interval_df.empty:
                    continue

                # Verifica se há buracos maiores que 1 minuto (ajustável)
                time_diffs = interval_df.index.to_series().diff().dropna()
                max_gap = time_diffs.max()

                # Se o buraco for maior que 1 minuto, descarte
                if max_gap > pd.Timedelta(minutes=1):
                    print(
                        f"Intervalo de {link} ({protocol}) em {name} descartado por buraco grande: {max_gap}"
                    )
                    continue

                # Verifica o número de amostras
                if len(interval_df) < 600:
                    print(
                        f"Intervalo de {link} ({protocol}) em {name} descartado por poucas amostras: {len(interval_df)}"
                    )
                    continue

                # Salva o intervalo válido
                interval_filename = f"{link}_{protocol}_interval_{i}.csv"
                output_path = os.path.join(output_dir_protocol, interval_filename)
                interval_df.to_csv(output_path)
                print(f"Salvo {output_path}")


if __name__ == "__main__":
    process_and_save_datasets()
    print("Processamento concluído.")
