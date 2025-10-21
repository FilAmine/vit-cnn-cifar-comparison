import timm
import torch.nn as nn

class CIFARModel(nn.Module):
    """Base model adapter for CIFAR datasets"""
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.model = base_model
        # Adjust classifier for CIFAR
        if hasattr(self.model, 'classifier'):
            if isinstance(self.model.classifier, nn.Linear):
                self.model.classifier = nn.Linear(
                    self.model.classifier.in_features, num_classes
                )
            elif hasattr(self.model, 'head'):
                self.model.head = nn.Linear(self.model.head.in_features, num_classes)
    
    def forward(self, x):
        # Resize input for models expecting 224x224
        if x.shape[-1] != 224:
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear')
        return self.model(x)

def get_model(model_name, num_classes):
    """Get model by name"""
    if model_name == 'resnet50':
        model = timm.create_model('resnet50', pretrained=False, num_classes=num_classes)
    elif model_name == 'efficientnet_b0':
        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
    elif model_name == 'vit_base_patch16_224':
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
    elif model_name == 'deit_base_patch16_224':
        model = timm.create_model('deit_base_patch16_224', pretrained=False, num_classes=num_classes)
    elif model_name == 'convnext_base':
        model = timm.create_model('convnext_base', pretrained=False, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return CIFARModel(model, num_classes)
