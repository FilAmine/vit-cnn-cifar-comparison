import subprocess
import json
import matplotlib.pyplot as plt

def run_data_efficiency_experiment():
    """Run experiments with different data proportions"""
    models = ['resnet50', 'efficientnet_b0', 'vit_base_patch16_224']
    data_proportions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = {}
    
    for model in models:
        results[model] = {}
        for proportion in data_proportions:
            print(f"Training {model} with {proportion*100}% data")
            
            # Run training command
            cmd = [
                'python', 'train.py',
                '--model', model,
                '--dataset', 'cifar100',
                '--epochs', '100',
                '--data-proportion', str(proportion)
            ]
            
            # Execute and capture results
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse final accuracy from output
            # In practice, you'd parse from saved results
            final_acc = 50.0 + (proportion * 25)  # Placeholder
            results[model][proportion] = final_acc
    
    # Save results
    with open('data_efficiency_results.json', 'w') as f:
        json.dump(results, f)
    
    # Plot results
    plot_data_efficiency(results)

def plot_data_efficiency(results):
    """Plot data efficiency results"""
    plt.figure(figsize=(10, 6))
    
    for model, data in results.items():
        proportions = list(data.keys())
        accuracies = list(data.values())
        plt.plot(proportions, accuracies, marker='o', label=model, linewidth=2)
    
    plt.xlabel('Training Data Proportion')
    plt.ylabel('Accuracy (%)')
    plt.title('Data Efficiency Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('data_efficiency_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    run_data_efficiency_experiment()
