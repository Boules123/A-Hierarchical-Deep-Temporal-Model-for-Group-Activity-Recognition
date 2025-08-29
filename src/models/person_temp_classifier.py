import torch.nn as nn, torchvision.models as models

class PersonTempClassifier(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, num_classes=9):
        super(PersonTempClassifier, self).__init__()
        
        # pretrained model resnet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # remove fc layer classification
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        b,n,t,c,h,w = x.shape
        x = x.view(b*n*t, c, h,w)
        x = self.backbone(x) #[b*n*t, 2048, 1, 1]
        x = x.contiguous()
        x = x.view(b*n, t, -1) 
        out, (_, _) = self.lstm(x) #[b*n, t, hidden_dim]
        out = out[:, -1, :]  # take the last step in seq -> [b*n, hidden_dim]
        x = self.fc(out) #[b*n, num_classes]
        return x