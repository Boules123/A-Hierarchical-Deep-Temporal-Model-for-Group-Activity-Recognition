import torch, torch.nn as nn, torchvision.models as models

class PersonClassifier(nn.Module):
    def __init__(self, num_classes=9):
        super(PersonClassifier, self).__init__()

        # fine-tune the resnet50 model
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # change the fc to match number classes 
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=2048, out_features=num_classes))

    def forward(self, x):
        x = self.backbone(x) 
        return x

