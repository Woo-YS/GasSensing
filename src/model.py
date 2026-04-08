import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 1. 모델 팩토리 (통합 진입점)
# -----------------------------------------------------------------------------
def create_model(model: str, input_length: int, output_dim: int):
    """
    args:
        model (str): 모델 이름 ('cnn1d', 'mlp', 'acnn', 'bcnn')
        input_length (int): 입력 데이터 길이 (스펙트럼 포인트 수)
        output_dim (int): 출력 노드 수 (회귀=1, 분류=클래스 개수)
    """
    if model == 'cnn1d':
        return CNN1D(input_length, output_dim)
    elif model == 'mlp':
        return MLP(input_length, output_dim)
    elif model == 'acnn':
        return AttentionCNN1D(input_length, output_dim)
    elif model == 'bcnn':
        return BiLSTM_CNN1D(input_length, output_dim)
    elif model == 'acnn1d':
        return ACNN1D(input_length, output_dim)
    elif model == 'bcnn1d':
        return BCNN1D(input_length, output_dim)
    elif model == 'cnncls':
        return CNNDClassifier(input_length, output_dim)
    elif model == 'cnnaa':
        return CNN1Daa(input_length, output_dim)
    else:
        raise ValueError(f"Unknown model name: {model}")


# -----------------------------------------------------------------------------
# 2. [Upgrade] CNN1D (파라미터 대폭 증가 버전)
#    - 기존 300K -> 수백만(M) 단위로 체급을 키워 MLP와 대등하게 만듦
# -----------------------------------------------------------------------------
class CNN1D(nn.Module):
    def __init__(self, input_length, output_dim):
        super().__init__()
        
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 2
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 3
            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 4
            nn.Conv1d(256, 512,kernel_size=7, padding=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 5 (깊이 추가)
            nn.Conv1d(512, 512,kernel_size=7, padding=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim) # 1 or 3
        )

    def forward(self, x):
        # 입력 차원 보정: (Batch, Length) -> (Batch, 1, Length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.features(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)


# -----------------------------------------------------------------------------
# 3. MLP (Fully Connected Network)
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_length, output_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_length, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        if x.dim() == 3: # (Batch, 1, Length) -> (Batch, Length)
            x = x.squeeze(1)
        return self.net(x)


# -----------------------------------------------------------------------------
# 4. Attention CNN (ACNN) - 팀원 코드 통합
# -----------------------------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class AttentionCNN1D(nn.Module):
    def __init__(self, input_length, output_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.se1 = SEBlock(16)
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.se2 = SEBlock(32)
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.se3 = SEBlock(64)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 64(Avg) + 64(Max) = 128 input
        self.fc = nn.Linear(64 * 2, output_dim)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.se1(x)
        x = self.conv2(x)
        x = self.se2(x)
        x = self.conv3(x)
        x = self.se3(x)
        
        x_avg = self.global_pool(x).squeeze(-1)
        x_max = self.max_pool(x).squeeze(-1)
        x = torch.cat([x_avg, x_max], dim=1)
        
        return self.fc(x)


# -----------------------------------------------------------------------------
# 5. BiLSTM + CNN (BCNN) - 팀원 코드 통합
# -----------------------------------------------------------------------------
class BiLSTM_CNN1D(nn.Module):
    def __init__(self, input_length, output_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, 
                            batch_first=True, bidirectional=True)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # LSTM output(128) * 2(Pooling) = 256 input
        self.fc = nn.Linear(128 * 2, output_dim)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        
        # CNN Feature Extractor
        x = self.cnn(x) # [Batch, 128, Len']
        
        # LSTM (Batch, Seq, Feature)로 변환 필요
        x = x.permute(0, 2, 1) # [Batch, Len', 128]
        
        x, _ = self.lstm(x) # [Batch, Len', 128(64*2)]
        
        # 다시 Pooling을 위해 (Batch, Feature, Seq)로 원복
        x = x.permute(0, 2, 1)
        
        x_avg = self.global_pool(x).squeeze(-1)
        x_max = self.max_pool(x).squeeze(-1)
        x = torch.cat([x_avg, x_max], dim=1)
        
        return self.fc(x)
    
    
class ACNN1D(nn.Module):
    def __init__(self, input_length, output_dim):
        super().__init__()
        
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 64, kernel_size=50, padding=25, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 2
            nn.Conv1d(64, 128, kernel_size=10, padding=5, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 4
            nn.Conv1d(256, 256,kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
        )
        
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim) # 1 or 3
        )

    def forward(self, x):
        # 입력 차원 보정: (Batch, Length) -> (Batch, 1, Length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.features(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)
    
    
    
class BCNN1D(nn.Module):
    def __init__(self, input_length, output_dim):
        super().__init__()
        
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 64, kernel_size=50, padding=25, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 2
            nn.Conv1d(64, 128, kernel_size=10, padding=5, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 4
            nn.Conv1d(256, 512,kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 4
            nn.Conv1d(512, 512,kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
        )
        
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim) # 1 or 3
        )

    def forward(self, x):
        # 입력 차원 보정: (Batch, Length) -> (Batch, 1, Length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.features(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)
    
    
class CNNDClassifier(nn.Module):
    def __init__(self, input_length: int, num_classes: int):
        super().__init__()
        
        # ⚡ 요청하신 대로 nn.Sequential로 묶어서 정의
        self.features = nn.Sequential(
            # Conv1 (idx 0)
            nn.Conv1d(1, 16, kernel_size=7, padding=3, dilation=1),
            nn.ReLU(),   # (idx 1)
            
            # Conv2 (idx 2)
            nn.Conv1d(16, 32, kernel_size=7, padding=6, dilation=2),
            nn.ReLU(),   # (idx 3)
            
            # Conv3 (idx 4)
            nn.Conv1d(32, 64, kernel_size=7, padding=12, dilation=4),
            nn.ReLU(),   # (idx 5)
            
            # Flatten & FC
            nn.Flatten(), # (idx 6)
            nn.Linear(64 * input_length, num_classes) # (idx 7)
        )

    def forward(self, x):
        # 입력 차원 보정 (Batch, Length) -> (Batch, 1, Length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        return self.features(x)
    
    
    
class CNNDClassifier(nn.Module):
    def __init__(self, input_length: int, num_classes: int):
        super().__init__()
        
        # ⚡ [핵심] 체크포인트의 "model.model.0..." 구조와 맞추기 위해
        # 반드시 변수명을 'self.model'로 하고 nn.Sequential을 사용해야 함
        self.model = nn.Sequential(
            # Conv1 (idx 0)
            nn.Conv1d(1, 16, kernel_size=7, padding=3, dilation=1),
            nn.ReLU(),   # (idx 1)
            
            # Conv2 (idx 2)
            nn.Conv1d(16, 32, kernel_size=7, padding=6, dilation=2),
            nn.ReLU(),   # (idx 3)
            
            # Conv3 (idx 4)
            nn.Conv1d(32, 64, kernel_size=7, padding=12, dilation=4),
            nn.ReLU(),   # (idx 5)
            
            # Flatten & FC
            nn.Flatten(), # (idx 6)
            nn.Linear(64 * input_length, num_classes) # (idx 7)
        )

    def forward(self, x):
        # 입력 차원 보정 (Batch, Length) -> (Batch, 1, Length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # ⚡ self.model 통과
        return self.model(x)
    
    
    
class CNN1Daa(nn.Module):
    def __init__(self, input_length, output_dim):
        super().__init__()
        
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 2
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 3
            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 4
            nn.Conv1d(256, 512,kernel_size=7, padding=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 5 (깊이 추가)
            nn.Conv1d(512, 512,kernel_size=7, padding=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(512, 256,kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(256, 128,kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 64,kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            
        )
        
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim) # 1 or 3
        )

    def forward(self, x):
        # 입력 차원 보정: (Batch, Length) -> (Batch, 1, Length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.features(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)