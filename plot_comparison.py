
import json
import matplotlib.pyplot as plt
import os
import sys

def plot_metrics(logdir_base):
    rssm_log = os.path.join(logdir_base, 'rssm', 'metrics.jsonl')
    context_log = os.path.join(logdir_base, 'context', 'metrics.jsonl')
    
    metrics = ['episode/score', 'train/kl_loss', 'train/context_change_rate']
    
    def load_data(path):
        data = []
        if not os.path.exists(path):
            return data
        with open(path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    rssm_data = load_data(rssm_log)
    context_data = load_data(context_log)
    
    if not rssm_data and not context_data:
        print("No data found to plot.")
        return

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 15))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        if rssm_data:
            steps = [d['step'] for d in rssm_data if metric in d]
            values = [d[metric] for d in rssm_data if metric in d]
            ax.plot(steps, values, label='RSSM')
            
        if context_data:
            steps = [d['step'] for d in context_data if metric in d]
            values = [d[metric] for d in context_data if metric in d]
            ax.plot(steps, values, label='C-RSSM')
            
        ax.set_title(metric)
        ax.legend()
        ax.set_xlabel('Steps')

    plt.tight_layout()
    plot_path = os.path.join(logdir_base, 'comparison_plot.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_comparison.py <logdir_base>")
    else:
        plot_metrics(sys.argv[1])
