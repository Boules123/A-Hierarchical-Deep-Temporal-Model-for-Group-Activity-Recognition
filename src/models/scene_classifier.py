import torch , torch.nn as nn, torchvision.models as models

class SceneClassifier_B1(nn.Module):
    def __init__(self, num_classes=8):
        super(SceneClassifier_B1, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # replace the fc layer to match the num_classes
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )
    
    def forward(self, x):
        x = self.backbone(x) #[b, num_classes]
        return x


class SceneClassifier_B3(nn.Module):
    def __init__(self, model_person , num_classes=8):
        super(SceneClassifier_B3, self).__init__()
        self.backbone = model_person.backbone
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1]) # remove the fc layer classification

        # freeze the backbone parameters 
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.pooling = nn.AdaptiveMaxPool1d(1) # max pool over 12 players
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        b, n, c, h, w = x.shape
        x = x.view(b*n, c,h,w)
        x = self.backbone(x)    #[b*n, 2048, 1, 1]
        x = x.view(b, n, -1)
        x = x.permute(0, 2, 1)      #[b, 2048, n]
        x = self.pooling(x).squeeze(-1)     #[b, 2048]
        return self.fc(x)    #[b, num_classes]

class SceneClassifier_B5(nn.Module):
    def __init__(self, player_classifier_model, num_classes=8):
        super(SceneClassifier_B5, self).__init__()
        
        # use the trained models from PersonTempClassifier
        self.resnet50 = player_classifier_model.backbone
        self.lstm = player_classifier_model.lstm

        for module in [self.resnet50,  self.lstm]:
            for param in module.parameters():
                param.requires_grad = False

        # max pool over 12 players [b, 12, hidden_size] -> [b, 1, 2048]
        self.pool = nn.AdaptiveMaxPool2d((1, 2048))  

        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes), 
        )
    
    def forward(self, x):
        b, n, t, c, h, w = x.shape 
        x = x.view(b*n*t, c, h, w) # [b*n*t, c, h, w]
        x1 = self.resnet50(x) # [b * n * t, 2048, 1 , 1]

        x1 = x1.view(b*n, t, -1) #[b * n, t, 2048]
        x2, (h , c) = self.lstm(x1) # [b * n, t, hidden_size]

        x = torch.cat([x1, x2], dim=2) # Concat the visual representation and temporal representation
        x = x.contiguous()
        x = x[:, -1, :]   # [b*n, hidden_size+2048]

        x = x.view(b, n, -1) # [b, n, hidden_size + 2048]
        x = self.pool(x) # [b, 1, 2048]
        x = x.squeeze(dim=1) # [b, 2048]

        x = self.fc(x) # [b, num_classes]
        return x