import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean
import warnings

# Correctly import the refactored models
from code.models.recursive_lstm import RecursiveLSTM
from code.models.standard_lstm import StandardLSTM
from code.models.gru import GRUModel
from code.models.sarima import SARIMAModel

# Dynamic Time Warping - using a library if available, or a simple implementation
try:
    from dtaidistance import dtw
    def calculate_dtw_distance(y_true, y_pred):
        return dtw.distance(y_true, y_pred)
except ImportError:
    print("Warning: dtaidistance not found. Using a fallback for DTW. pip install dtaidistance for better performance.")
    try:
        from fastdtw import fastdtw
        def calculate_dtw_distance(y_true, y_pred):
            distance, _ = fastdtw(y_true, y_pred, dist=euclidean)
            return distance
    except ImportError:
        print("Warning: fastdtw not found. Skipping DTW analysis. pip install fastdtw.")
        def calculate_dtw_distance(y_true, y_pred):
            return np.nan


warnings.filterwarnings("ignore")
# Constants
HORIZON = 28
TRAIN_WINDOW_SIZE = 200  # Number of data points to train on for each window
TEST_WINDOW_SIZE = HORIZON # We test on the next HORIZON points
SLIDING_STEP = HORIZON # How many steps to slide the window forward

def load_dataset(dataset_path):
    """Loads a dataset from a given path."""
    df = pd.read_csv(dataset_path, index_col='Timestamp', parse_dates=True)
    df.rename(columns={'Vazao_1': 'Vazao'}, inplace=True) # Standardize column name
    if '0' in df.columns:
        df = df.drop('0', axis=1)
    df['Vazao'] = df['Vazao'] / 1_000_000  # Bits to Megabits
    df['Vazao'] = df['Vazao'].replace(-1, np.nan)
    # Simple linear interpolation for now
    df['Vazao'] = df['Vazao'].interpolate(method='linear', limit_direction='both')
    return df

def calculate_rmse(y_true, y_pred):
    """Calculates Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred)**2))

def calculate_nrmse(y_true, y_pred):
    """Calculates Normalized Root Mean Squared Error."""
    rmse = calculate_rmse(y_true, y_pred)
    range = y_true.max() - y_true.min()
    return rmse / range if range != 0 else 0

# --- Analysis Functions ---

def horizon_degradation_analysis(results_df, output_dir):
    """Analyzes and plots error vs. forecast horizon."""
    horizon_errors = results_df.melt(id_vars='model', value_vars=[f'rmse_h{h}' for h in range(1, HORIZON + 1)],
                                     var_name='horizon', value_name='rmse')
    horizon_errors['horizon'] = horizon_errors['horizon'].str.replace('rmse_h', '').astype(int)
    
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=horizon_errors, x='horizon', y='rmse', hue='model', marker='o', ci='sd')
    plt.title('Horizon Degradation Analysis (Error vs. Horizon)')
    plt.xlabel('Forecast Horizon (h steps ahead)')
    plt.ylabel('Average RMSE (Mbits/s)')
    plt.xticks(np.arange(1, HORIZON + 1, 2))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'horizon_degradation.png'))
    plt.close()

def error_distribution_analysis(results_df, output_dir):
    """Analyzes and plots the distribution of NRMSE."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=results_df, x='model', y='nrmse')
    plt.title('Error Distribution (NRMSE)')
    plt.xlabel('Model')
    plt.ylabel('Normalized RMSE')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'nrmse_boxplot.png'))
    plt.close()

def shape_similarity_analysis(results_df, output_dir):
    """Analyzes and plots DTW distances."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=results_df, x='model', y='dtw')
    plt.title('Shape Similarity Analysis (DTW Distance)')
    plt.xlabel('Model')
    plt.ylabel('Dynamic Time Warping Distance')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dtw_boxplot.png'))
    plt.close()

# --- Sparsity Stress Test ---
def inject_nans(data, sparsity_level):
    """Injects NaNs into a copy of the data."""
    data_with_nans = data.copy()
    n_values_to_nan = int(len(data_with_nans) * sparsity_level)
    nan_indices = np.random.choice(data_with_nans.index, n_values_to_nan, replace=False)
    data_with_nans.loc[nan_indices, 'Vazao'] = np.nan
    return data_with_nans

def sparsity_stress_test(models, dataset_path, output_dir):
    print("\nStarting Sparsity Stress Test...")
    sparsity_levels = [0.1, 0.3, 0.5, 0.7]
    full_df = load_dataset(dataset_path)
    
    # We need a continuous segment of data for this test
    test_segment = full_df.iloc[- (TRAIN_WINDOW_SIZE + HORIZON) : ]
    if len(test_segment) < TRAIN_WINDOW_SIZE + HORIZON:
        print("Skipping sparsity test: Not enough data in the chosen dataset.")
        return

    train_data_full = test_segment.iloc[:TRAIN_WINDOW_SIZE]
    y_true = test_segment.iloc[TRAIN_WINDOW_SIZE : TRAIN_WINDOW_SIZE + HORIZON]['Vazao'].values
    
    sparsity_results = []

    for sparsity in tqdm(sparsity_levels, desc="Sparsity Levels"):
        # Inject NaNs into the training data
        train_data_sparse = inject_nans(train_data_full, sparsity)
        
        # Interpolate the sparse data before feeding to models
        train_data_interpolated = train_data_sparse.interpolate(method='linear', limit_direction='both')

        for model_name, model in models.items():
            model.train(train_data_interpolated)
            y_pred = model.predict(train_data_interpolated, HORIZON) # predict based on sparse, interpolated data
            
            rmse = calculate_rmse(y_true, y_pred)
            sparsity_results.append({
                'model': model_name,
                'sparsity': sparsity,
                'rmse': rmse
            })
    
    sparsity_df = pd.DataFrame(sparsity_results)
    
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=sparsity_df, x='sparsity', y='rmse', hue='model', marker='o')
    plt.title('Sparsity Stress Test (Robustness to Missing Data)')
    plt.xlabel('Sparsity Level (Fraction of Missing Data in Training Window)')
    plt.ylabel('RMSE on Clean Test Set')
    plt.gca().set_xticklabels([f'{x:.0%}' for x in plt.gca().get_xticks()])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sparsity_stress_test.png'))
    plt.close()
    print("Sparsity Stress Test complete.")


# --- Main Execution ---

def main(dataset_dir, output_dir, limit_files):
    """Main function to run the benchmarking suite."""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    models = {
        "Recursive-LSTM": RecursiveLSTM(look_back=28), # Using a look_back of 28 for the recursive model
        "Standard-LSTM": StandardLSTM(look_back=28),
        "GRU": GRUModel(look_back=28),
        "SARIMA": SARIMAModel()
    }
    
    all_results = []
    
    dataset_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
    if limit_files > 0:
        dataset_files = dataset_files[:limit_files]
        print(f"Limiting processing to the first {limit_files} files.")

    for dataset_file in tqdm(dataset_files, desc="Processing datasets"):
        dataset_path = os.path.join(dataset_dir, dataset_file)
        df = load_dataset(dataset_path)

        if len(df) < (TRAIN_WINDOW_SIZE + TEST_WINDOW_SIZE):
            print(f"Skipping {dataset_file}: not enough data.")
            continue
            
        # Sliding window validation
        for i in range(0, len(df) - TRAIN_WINDOW_SIZE - TEST_WINDOW_SIZE + 1, SLIDING_STEP):
            train_window = df.iloc[i : i + TRAIN_WINDOW_SIZE]
            test_window = df.iloc[i + TRAIN_WINDOW_SIZE : i + TRAIN_WINDOW_SIZE + TEST_WINDOW_SIZE]
            y_true = test_window['Vazao'].values
            
            for model_name, model in models.items():
                # Train the model on the current training window
                model.train(train_window)
                
                # Predict the future horizon
                y_pred = model.predict(train_window, HORIZON)
                
                # Calculate metrics for the window
                nrmse = calculate_nrmse(y_true, y_pred)
                dtw_dist = calculate_dtw_distance(y_true, y_pred)
                
                result_row = {'model': model_name, 'dataset': dataset_file, 'window': i, 'nrmse': nrmse, 'dtw': dtw_dist}
                
                # Calculate and add per-horizon RMSE
                for h in range(HORIZON):
                    rmse_h = calculate_rmse(y_true[h:h+1], y_pred[h:h+1])
                    result_row[f'rmse_h{h+1}'] = rmse_h
                
                all_results.append(result_row)

    if not all_results:
        print("No results were generated. Check dataset paths and sizes.")
        return
        
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, 'comparison_metrics.csv'), index=False)
    
    print(f"\nBenchmark complete. Metrics saved to {os.path.join(output_dir, 'comparison_metrics.csv')}")

    # --- Generate Plots from all collected data ---
    print("Generating analysis plots...")
    horizon_degradation_analysis(results_df, output_dir)
    error_distribution_analysis(results_df, output_dir)
    shape_similarity_analysis(results_df, output_dir)
    print("Analysis plots saved.")

    # --- Run Sparsity Test on a single representative dataset ---
    if dataset_files:
        first_dataset_path = os.path.join(dataset_dir, dataset_files[0])
        sparsity_stress_test(models, first_dataset_path, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmarking suite for time series models.')
    parser.add_argument('--dataset-dir', type=str, default='datasets/choosen', help='Directory containing the datasets.')
    parser.add_argument('--output-dir', type=str, default='results/comparison', help='Directory to save results and plots.')
    parser.add_argument('--limit-files', type=int, default=0, help='Limit the number of dataset files to process for a quick run. Set to 0 to run all.')
    args = parser.parse_args()
    
    main(args.dataset_dir, args.output_dir, args.limit_files)