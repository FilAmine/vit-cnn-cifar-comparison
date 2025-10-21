# Experiment configuration
EXPERIMENT_CONFIG = {
    'models': [
        'resnet50',
        'efficientnet_b0', 
        'vit_base_patch16_224',
        'deit_base_patch16_224',
        'convnext_base'
    ],
    'datasets': ['cifar10', 'cifar100'],
    'seeds': [42, 123, 456],
    'training': {
        'epochs': 200,
        'batch_size': 128,
        'learning_rate': 0.001,
        'weight_decay': 0.05
    },
    'data_augmentation': {
        'random_crop': True,
        'horizontal_flip': True,
        'cutmix': True
    }
}
