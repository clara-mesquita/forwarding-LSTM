"""
SBRC 2026 Paper - Result Processing Script

This script automates the analysis of experimental results for the SBRC 2026 paper
on robust network performance prediction. It processes JSON and log files from
the proposed model and baseline models (GRU, LSTM, SARIMA).

The script performs the following actions:
1.  Loads performance metrics from all specified JSON files.
2.  For the proposed model, selects the best result per link based on the lowest RMSE
    across different measurement intervals.
3.  Aggregates the data from all models into a single CSV file.
4.  Calculates global mean and standard deviation for key metrics (RMSE, MAE, NRMSE,
    Fuzzy Accuracy, Execution Time) for each model.
5.  Generates and prints publication-ready LaTeX code for two tables:
    - Table 1: A global performance summary comparing all models.
    - Table 2: A robustness comparison showing the RMSE of each model on specific
      strategic links.
6.  Generates and prints Python code snippets for creating visualizations with
    Seaborn and Matplotlib:
    - A boxplot for NRMSE comparison.
    - A bar chart for Fuzzy Accuracy (FAM) comparison.
    - A bar chart for inference time comparison.

Inputs:
- Proposal Model: 'results/cubic/evaluation.json'
- Baselines: 'results2/cubic/evaluation_*.json'

Output:
- A CSV file 'final_comparison_results.csv' with the aggregated data.
- LaTeX table strings printed to the console.
- Python plotting code printed to the console.
"""
import json
import os
import pandas as pd
import numpy as np

# --- Configuration: File Paths ---
# Using a dictionary to map model names to their respective data files
DATA_FILES = {
    "Proposal": "results/cubic/evaluation.json",
    "GRU": "results2/cubic/evaluation_gru.json",
    "LSTM": "results2/cubic/evaluation_lstm.json",
    "SARIMA": "results2/cubic/evaluation_sarima.json",
}

OUTPUT_CSV = "final_comparison_results.csv"

# --- Data Loading and Processing Functions ---

def parse_link_from_key(key):
    """Extracts the link identifier from a composite key."""
    # Assumes key format like "pb-sp_cubic_interval_0" -> "pb-sp"
    return key.split("_")[0]

def load_proposal_data(file_path):
    """
    Loads the proposal's evaluation data, selecting the best result (lowest RMSE)
    for each link if multiple intervals are present.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

    best_results = {}
    for key, metrics in data.items():
        link = parse_link_from_key(key)
        # Ensure essential metrics exist
        if "RMSE" not in metrics:
            continue
        
        # If we haven't seen this link, or if the current entry has a lower RMSE, store it.
        if link not in best_results or metrics["RMSE"] < best_results[link]["RMSE"]:
            best_results[link] = metrics
            best_results[link]["Link"] = link

    if not best_results:
        return pd.DataFrame()

    df = pd.DataFrame(list(best_results.values()))
    df["Model"] = "Proposal"
    return df

def load_baseline_data(file_path, model_name):
    """Loads evaluation data for a given baseline model."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

    processed_data = []
    for key, metrics in data.items():
        link = parse_link_from_key(key)
        metrics["Link"] = link
        processed_data.append(metrics)

    if not processed_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(processed_data)
    df["Model"] = model_name
    return df

# --- LaTeX and Visualization Code Generation ---

def generate_latex_summary_table(summary_df):
    """Generates LaTeX code for the global performance summary table."""
    
    header = r"""
\begin{table}[h]
\centering
\caption{Global Performance Summary of Prediction Models.}
\label{tab:global_performance}
\begin{tabular}{l c c c c c}
\toprule
\textbf{Model} & \textbf{RMSE} & \textbf{MAE} & \textbf{NRMSE} & \textbf{Fuzzy Acc.} & \textbf{Time (ms)} \\
\midrule
"""
    
    footer = r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    body = ""
    for model_name, row in summary_df.iterrows():
        body += f"{model_name} & "
        body += f"{row[('RMSE', 'mean')]:.3f} $\pm$ {row[('RMSE', 'std')]:.3f} & "
        body += f"{row[('MAE', 'mean')]:.3f} $\pm$ {row[('MAE', 'std')]:.3f} & "
        body += f"{row[('NRMSE', 'mean')]:.3f} $\pm$ {row[('NRMSE', 'std')]:.3f} & "
        body += f"{row[('Fuzzy_Accuracy', 'mean')]:.3f} $\pm$ {row[('Fuzzy_Accuracy', 'std')]:.3f} & "
        # Execution time is often less variable, so std might be less relevant here, but kept for consistency.
        body += f"{row[('Execution_Time', 'mean')]:.2f} $\pm$ {row[('Execution_Time', 'std')]:.2f} \\\\n"
        
    return header + body + footer

def generate_latex_robustness_table(data_df):
    """Generates LaTeX code for the robustness comparison table (RMSE per link)."""
    
    # Pivot table to have links as rows and models as columns for RMSE
    pivot_df = data_df.pivot_table(index="Link", columns="Model", values="RMSE")
    
    # Ensure the desired column order
    model_order = ["Proposal", "GRU", "LSTM", "SARIMA"]
    pivot_df = pivot_df.reindex(columns=model_order)

    header = r"""
\begin{table}[h]
\centering
\caption{Robustness Comparison: RMSE Across Strategic Links.}
\label{tab:robustness_comparison}
\begin{tabular}{l c c c c}
\toprule
\textbf{Link} & \textbf{Proposal} & \textbf{GRU} & \textbf{LSTM} & \textbf{SARIMA} \\
\midrule
"""
    
    footer = r"""
\bottomrule
\end{tabular}
\end{table}
"""

    body = ""
    for link, row in pivot_df.iterrows():
        body += f"{link.upper()} & "
        body += f"{row['Proposal']:.3f} & "
        body += f"{row['GRU']:.3f} & "
        body += f"{row['LSTM']:.3f} & "
        body += f"{row['SARIMA']:.3f} \\\\n"

    return header + body + footer

def generate_plotting_code(csv_path):
    """Generates Python code for creating specified visualizations."""
    
    code = f'''
"""
Visualization Code for SBRC 2026 Paper
Generated by process_results.py

This script loads the aggregated results from '{csv_path}'
and generates the following plots:
1. Boxplot of NRMSE scores for all models.
2. Bar chart of mean Fuzzy Accuracy (FAM) for all models.
3. Bar chart of mean Inference Time for all models.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load Data ---
try:
    df = pd.read_csv("{csv_path}")
except FileNotFoundError:
    print(f"Error: The data file '{csv_path}' was not found.")
    exit()

# --- Plotting Style ---
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")
palette = "viridis"

# --- 1. NRMSE Boxplot ---
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Model", y="NRMSE", palette=palette)
plt.title("NRMSE Distribution Across All Links")
plt.xlabel("Model")
plt.ylabel("Normalized Root Mean Square Error (NRMSE)")
plt.tight_layout()
plt.savefig("nrmse_comparison_boxplot.png", dpi=300)
print("Generated nrmse_comparison_boxplot.png")

# --- 2. Fuzzy Accuracy (FAM) Bar Chart ---
agg_metrics = df.groupby("Model")["Fuzzy_Accuracy"].agg(['mean', 'std']).reset_index()

plt.figure(figsize=(10, 6))
barplot = sns.barplot(data=agg_metrics, x="Model", y="mean", palette=palette)
# Add error bars
plt.errorbar(x=agg_metrics["Model"], y=agg_metrics["mean"], yerr=agg_metrics["std"], fmt='none', c='black', capsize=5)
plt.title("Mean Fuzzy Accuracy (FAM) Comparison")
plt.xlabel("Model")
plt.ylabel("Mean Fuzzy Accuracy")
# Add values on top of bars
for index, row in agg_metrics.iterrows():
    barplot.text(index, row['mean'] + 0.01, f"{{row['mean']:.3f}}", color='black', ha="center")
plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig("fuzzy_accuracy_comparison_barchart.png", dpi=300)
print("Generated fuzzy_accuracy_comparison_barchart.png")


# --- 3. Inference Time Bar Chart ---
agg_time = df.groupby("Model")["Execution_Time"].agg(['mean', 'std']).reset_index()

plt.figure(figsize=(10, 6))
time_barplot = sns.barplot(data=agg_time, x="Model", y="mean", palette=palette)
plt.title("Mean Inference Time Comparison")
plt.xlabel("Model")
plt.ylabel("Mean Inference Time (ms)")
plt.yscale('log') # Use log scale if times vary drastically
# Add values on top of bars
for index, row in agg_time.iterrows():
    time_barplot.text(index, row['mean'], f"{{row['mean']:.2f}}", color='black', ha="center", va='bottom')
plt.tight_layout()
plt.savefig("inference_time_comparison_barchart.png", dpi=300)
print("Generated inference_time_comparison_barchart.png")

plt.show()

'''
    return code


# --- Main Execution Logic ---

def main():
    """Main function to run the data processing and generation workflow."""
    all_dfs = []

    # Load proposal data
    print("Processing Proposal data...")
    proposal_df = load_proposal_data(DATA_FILES["Proposal"])
    if not proposal_df.empty:
        all_dfs.append(proposal_df)
    
    # Load baseline data
    for model_name, file_path in DATA_FILES.items():
        if model_name == "Proposal":
            continue
        print(f"Processing {model_name} data...")
        baseline_df = load_baseline_data(file_path, model_name)
        if not baseline_df.empty:
            all_dfs.append(baseline_df)

    if not all_dfs:
        print("No data could be loaded. Exiting.")
        return

    # Combine all data into a single DataFrame
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # --- Data Cleaning and Transformation ---
    # Convert Execution_Time from seconds to milliseconds
    if "Execution_Time" in final_df.columns:
        final_df["Execution_Time"] = final_df["Execution_Time"] * 1000
    
    # Rename columns for clarity in outputs
    final_df.rename(columns={"Fuzzy_Accuracy": "Fuzzy_Accuracy"}, inplace=True)
    
    # Ensure all required metric columns exist, fill with NaN if missing
    required_metrics = ["RMSE", "MAE", "NRMSE", "Fuzzy_Accuracy", "Execution_Time"]
    for metric in required_metrics:
        if metric not in final_df.columns:
            final_df[metric] = np.nan
            print(f"Warning: Metric '{metric}' not found in all files. Column added with NaNs.")

    # Save aggregated data to CSV
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSuccessfully generated aggregated data file: {OUTPUT_CSV}")

    # --- Generate Outputs ---
    
    # 1. Global Aggregation
    # Group by Model and calculate mean and std for all relevant metrics
    global_summary = final_df.groupby("Model")[required_metrics].agg([np.mean, np.std])
    # For display purposes, fill NaN with 0
    global_summary.fillna(0, inplace=True)


    # 2. LaTeX Table 1: Global Summary
    print("\n" + "="*20 + " LaTeX: Global Performance Summary " + "="*20)
    latex_summary = generate_latex_summary_table(global_summary)
    print(latex_summary)
    
    # 3. LaTeX Table 2: Robustness per Link
    print("\n" + "="*20 + " LaTeX: Robustness Comparison (RMSE) " + "="*20)
    # Filter out links where not all models have data, to make the table complete
    link_counts = final_df.groupby('Link')['Model'].nunique()
    complete_links = link_counts[link_counts == len(DATA_FILES)].index
    robustness_data = final_df[final_df['Link'].isin(complete_links)]
    
    if not robustness_data.empty:
        latex_robustness = generate_latex_robustness_table(robustness_data)
        print(latex_robustness)
    else:
        print("Could not generate robustness table: No links with complete data across all models found.")

    # 4. Visualization Code
    print("\n" + "="*20 + " Python Code for Visualizations " + "="*20)
    plotting_code = generate_plotting_code(OUTPUT_CSV)
    print(plotting_code)
    # Save the plotting code to a file for convenience
    with open("generate_plots.py", "w") as f:
        f.write(plotting_code)
    print("\nVisualization code has also been saved to 'generate_plots.py'")


if __name__ == "__main__":
    main()
