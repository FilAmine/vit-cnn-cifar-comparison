import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import timm

def test_robustness(model, dataset_name, corruption_type, severity=1):
    """Test model robustness against various corruptions"""
    
    # Define corruption transforms
    corruption_transforms = {
        'gaussian_noise': transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1 * severity),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]),
        'motion_blur': transforms.Compose([
            transforms.ToTensor(),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1 * severity)),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]),
        'brightness': transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.clamp(x * (1 + 0.2 * severity), 0, 1)),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    }
    
    # Load dataset with corruption
    if dataset_name == 'cifar100':
        dataset = datasets.CIFAR100(
            root='./data', train=False, download=True,
            transform=corruption_transforms[corruption_type]
        )
    
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    
    # Evaluate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def run_robustness_analysis():
    """Run comprehensive robustness analysis"""
    models = {
        'resnet50': timm.create_model('resnet50', pretrained=False, num_classes=100),
        'efficientnet_b0': timm.create_model('efficientnet_b0', pretrained=False, num_classes=100),
        'vit_base_patch16_224': timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=100)
    }
    
    corruptions = ['gaussian_noise', 'motion_blur', 'brightness']
    results = {}
    
    for model_name, model in models.items():
        results[model_name] = {}
        print(f"Testing {model_name}...")
        
        # Load trained weights (you need to implement this)
        # model.load_state_dict(torch.load(f'checkpoints/{model_name}_cifar100_best.pth.tar'))
        
        # Test clean accuracy
        clean_acc = test_robustness(model, 'cifar100', 'gaussian_noise', severity=0)
        results[model_name]['clean'] = clean_acc
        
        # Test corruptions
        for corruption in corruptions:
            acc = test_robustness(model, 'cifar100', corruption, severity=1)
            results[model_name][corruption] = acc
    
    # Save results
    import json
    with open('robustness_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == '__main__':
    results = run_robustness_analysis()
    print("Robustness Analysis Results:")
    print(json.dumps(results, indent=2))
