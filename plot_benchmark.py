
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_benchmark_results(csv_path, output_dir='plots'):
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # 样式
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.dpi'] = 150
    
    # 1. NAC Score Comparison
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x='perturbation', y='nac_mean', hue='model')
    plt.xticks(rotation=45, ha='right')
    plt.title('NAC Coverage Score across Perturbations')
    plt.ylabel('Mean NAC Score')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/nac_comparison.png')
    
    # 2. Accuracy Comparison
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x='perturbation', y='accuracy', hue='model')
    plt.xticks(rotation=45, ha='right')
    plt.title('Model Accuracy across Perturbations')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_comparison.png')
    
    # 3. Accuracy vs NAC Scatter
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='accuracy', y='nac_mean', hue='perturbation', style='model', s=100)
    plt.title('Accuracy vs NAC Coverage')
    plt.xlabel('Top-1 Accuracy')
    plt.ylabel('Mean NAC Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/acc_vs_nac.png')
    
    # 4. NAC Delta (Relative to Clean)
    # 计算 Delta
    clean_scores = df[df['perturbation'] == 'Clean'].set_index('model')['nac_mean']
    df['nac_delta'] = df.apply(lambda row: row['nac_mean'] - clean_scores[row['model']], axis=1)
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df[df['perturbation'] != 'Clean'], x='perturbation', y='nac_delta', hue='model')
    plt.xticks(rotation=45, ha='right')
    plt.title('NAC Delta relative to Clean Data')
    plt.ylabel('Delta NAC')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/nac_delta.png')
    
    print(f"Plots saved to {output_dir}/")

if __name__ == "__main__":
    plot_benchmark_results('total_benchmark_results/benchmark_results.csv')
