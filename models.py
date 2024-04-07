import torch
import torch.nn as nn
import torch.nn.functional as F

class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        # Basic network, no T-nets
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        x = points.transpose(1, 2)            # (B, 3, N)
        x = self.encoder(x)                   # (B, 1024, N)
        x = torch.max(x, 2, keepdim=True)[0]  # (B, 1024, 1)
        x = x.view(-1, 1024)                  # (B, 1024)
        x = self.classifier(x)                # (B, num_classes)
        return x

class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        # Basic network, no T-nets
        self.layer1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024)
        )
        self.segmentation = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, num_seg_classes, 1)
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        x = points.transpose(1, 2)                          # (B, 3, N)
        x1 = self.layer1(x)                                 # (B, 64, N)
        x2 = self.layer2(x1)                                # (B, 1024, N)
        global_feat = torch.max(x2, 2, keepdim=True)[0]     # (B, 1024, 1)
        global_feat = global_feat.repeat(1, 1, x2.size(2))  # (B, 1024, N)
        feat = torch.cat([x1, global_feat], 1)              # (B, 1088, N)
        out = self.segmentation(feat)                       # (B, num_seg_classes, N)
        out = out.transpose(1, 2).contiguous()              # (B, N, num_seg_classes)
        return out
