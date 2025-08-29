import torch, torch.nn as nn, torchvision.models as models


class SceneTempClassifier_B4(nn.Module):
    def __init__(self, num_classes=8):
        super(SceneTempClassifier_B4, self).__init__()
        
        # trained models 
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        self.rnn = nn.LSTM(
            input_size=2048,
            hidden_size=512,
            num_layers=1, 
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w) 
        x = self.backbone(x) #[b*t, 2048, 1, 1]
        x = x.view(b, t, -1)
        out, (_,_) = self.rnn(x) #[b, t, 512]
        return self.fc(out[:, -1, :]) #[b, num_classes]


class SceneTempClassifier_B6(nn.Module):
    def __init__(self, model_person, input_dim=2048, hidden_dim=512, num_classes=8):
        super(SceneTempClassifier_B6, self).__init__()
        
        self.backbone = model_person.backbone
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.pool = nn.AdaptiveMaxPool1d(1) #max pool
        
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
        b, n, t, c, h, w = x.shape
        x = x.reshape(b * n * t, c, h, w)
        x = self.backbone(x) #[b*n*t, 2048, 1, 1]

        x = x.view(b * t, n, -1)
        x = x.permute(0, 2, 1) # [b*t, 2048, n]
        
        x = self.pool(x)
        x = x.squeeze(dim=2) #[b*t, 2048]
        
        x = x.view(b, t, -1) 
        x, (_,_) = self.lstm(x) #[b, t, hidden_dim]
        x = x[:, -1, :] #take last step -> [b, hidden_dim]

        x = self.fc(x) #[b, num_classes]
        return x



class SceneTempClassifier_B7(nn.Module):
    def __init__(self,player_classifier_model, num_classes=8):
        super(SceneTempClassifier_B7, self).__init__()
        
        self.feature_extract = player_classifier_model.backbone
        self.lstm_person = player_classifier_model.lstm # extract the person feature by cnn + lstm

        # freeze the person feature extractor (resnet50)
        for param in self.feature_extract.parameters():
            param.requires_grad = False

        # freeze the lstm_person 
        for param in self.lstm_person.parameters():
            param.requires_grad = False

        # max pool over 12 player and each player has 2048 features
        self.pooling = nn.AdaptiveMaxPool2d((1, 2048))

        # lstm for group after pooling player
        self.lstm_group = nn.LSTM(
            input_size=2048,
            hidden_size=512,
            num_layers=1,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        b,n,t,c,h,w = x.shape
        x = x.view(b*n*t, c,h,w)
        x = self.feature_extract(x)     #[b*n*t, 2048, 1, 1]

        x = x.view(b*n, t, -1)
        out_person, (_,_) = self.lstm_person(x)     #[b*n, t, 512]
        
        out_person = torch.cat([x, out_person], dim=2)   #[b*n, t, 2048+512]
        out_person = out_person.contiguous()
        
        out_person = out_person.view(b*t, n, -1) #[b*t, n, -1]

        out_person = self.pooling(out_person)   #[b*t, 1, 2048]
        out_person = out_person.contiguous()
        
        out_person = out_person.view(b,t,-1)
        out_group ,(_,_) = self.lstm_group(out_person)      #[b, t, 512]
        out_group = out_group[:, -1, :]     #[b, 512]
        
        x = self.fc(out_group)      #[b, num_classes]
        return x


class SceneTempClassifier_B8(nn.Module):
    def __init__(self, person_classifier_model, num_classes=8):
        super(SceneTempClassifier_B8, self).__init__()
        
        self.feature_extract = person_classifier_model.backbone
        self.lstm_person = person_classifier_model.lstm_person
        
        for param in self.feature_extract.parameters():
            param.requires_grad = False

        for param in self.lstm_person.parameters():
            param.requires_grad = False
            
        self.pooling = nn.AdaptiveMaxPool1d(1)

        self.lstm_group = nn.LSTM(
            input_size=5120, 
            hidden_size=512,
            num_layers=1,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        b, n, t, c, h, w = x.shape
        
        x = x.view(b * n * t, c, h, w)
        x = self.feature_extract(x)
        
        x = x.view(b * n, t, -1)
        x = x.contiguous()
        out, (_, _) = self.lstm_person(x) #[b*n, t, 512]
        
        out = out.contiguous()
        out = torch.cat([x, out], dim=-1) #[b*t, 12, 2048+512]
        out = out.view(b, n, t, -1)       #[b, 12, t, 2560]

        out = out.permute(0, 2, 1, 3) #[b, t, 12, 2560]

        out = out.contiguous().view(b * t, n, -1) #[b*t, 12, 2560]

        left_team = out[:, :6, :]   #[b*t, 6, 2560]
        right_team = out[:, 6:, :]  #[b*t, 6, 2560]
        
        left_team = self.pooling(left_team.permute(0, 2, 1)).squeeze(-1)    #[b*t, 2560]
        right_team = self.pooling(right_team.permute(0, 2, 1)).squeeze(-1)  #[b*t, 2560]
        
        x = torch.cat([left_team, right_team], dim=1)   #[b*t, 5120]
        x = x.contiguous()
        x = x.view(b, t, -1)             #[b, t, 5120]
        out_group, (_, _) = self.lstm_group(x) #[b, t, 512]
        
        out_group = out_group[:, -1, :]     #[b, 512]
        x = self.fc(out_group)              #[b, num_classes]
        
        return x

