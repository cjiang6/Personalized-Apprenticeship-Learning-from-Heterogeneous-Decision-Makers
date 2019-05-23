import torch
import torch.nn as nn

# will do embedding for now

class EmbeddingModule(nn.Module):
    def __init__(self, n=36):
        super(EmbeddingModule, self).__init__()
        self.embedding = nn.Parameter(torch.randn(1, n))

    def forward(self):
        return







class LearnActionEmbeddings(nn.Module):
    def __init__(self):
        super(LearnActionEmbeddings, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.relu1 = nn.ReLU()
        self.fc1a = nn.Linear(128,128)
        self.relu1a = nn.ReLU()
        self.fc1b = nn.Linear(164,128)
        self.relu1b = nn.ReLU()
        self.fc1c = nn.Linear(128, 64)
        self.dropout1 = nn.Dropout(p=.2)
        self.relu1c = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 128)  # predicting player state, unit type counts, enemy player counts, and 80 image #s
        self.relu4 = nn.ReLU()
        self.fc4 = nn.Linear(128,256)
        self.EmbeddingList = nn.ModuleList([EmbeddingModule() for i in range(1)])

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)
        w = self.EmbeddingList[0].embedding
        x = self.fc1a(x)
        x = self.relu1a(x)
        w = w.reshape((36))
        x = torch.cat([x, w], dim=0)
        x = self.fc1b(x)
        x = self.relu1b(x)
        x = self.fc1c(x)
        x = self.dropout1(x)
        x = self.relu1c(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu4(x)
        x = self.fc4(x)

        return x  # returns next state

class PairwiseIsActionTakenBase(nn.Module):
    def __init__(self):
        super(PairwiseIsActionTakenBase, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.relu1 = nn.ReLU()
        self.fc1a = nn.Linear(164, 128)
        self.relu1a = nn.ReLU()
        self.fc1b = nn.Linear(128, 128)
        self.relu1b = nn.ReLU()
        self.fc1c = nn.Linear(128, 64)
        self.relu1c = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)  # predicting player state, unit type counts, enemy player counts, and 80 image #s
        self.relu4 = nn.ReLU()
        self.fc4 = nn.Linear(32, 16)
        self.relu5 = nn.ReLU()
        self.fc5 = nn.Linear(16, 1)
        self.sig = nn.Sigmoid()


    def forward(self, x, w):
        x = self.fc1(x)
        x = self.relu1(x)
        x = torch.cat([x, w], dim=0)
        x = self.fc1a(x)
        x = self.relu1a(x)
        x = self.fc1b(x)
        x = self.relu1b(x)
        x = self.fc1c(x)
        x = self.relu1c(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu4(x)
        x = self.fc4(x)
        x = self.relu5(x)
        x = self.fc5(x)

        return self.sig(x)  # returns next state

class PairwiseIsActionTakenBayesian(nn.Module):
    def __init__(self):
        super(PairwiseIsActionTakenBayesian, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.relu1 = nn.ReLU()
        self.fc1a = nn.Linear(164, 128)
        self.relu1a = nn.ReLU()
        self.fc1b = nn.Linear(128, 128)
        self.relu1b = nn.ReLU()
        self.fc1c = nn.Linear(140, 64)
        self.relu1c = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)  # predicting player state, unit type counts, enemy player counts, and 80 image #s
        self.relu4 = nn.ReLU()
        self.fc4 = nn.Linear(32, 16)
        self.relu5 = nn.ReLU()
        self.fc5 = nn.Linear(16, 1)
        self.sig = nn.Sigmoid()
        self.action_EmbeddingList = nn.ModuleList([EmbeddingModule() for i in range(1)])
        self.EmbeddingList = nn.ModuleList([EmbeddingModule(n=12) for i in range(1)]) # game

    def forward(self, x, w):
        x = self.fc1(x)
        x = self.relu1(x)
        x = torch.cat([x, w], dim=0)
        x = self.fc1a(x)
        x = self.relu1a(x)
        x = self.fc1b(x)
        x = self.relu1b(x)
        w_game = self.EmbeddingList[0].embedding
        x = torch.cat([x, w_game.reshape(12)], dim=0)
        x = self.fc1c(x)
        x = self.relu1c(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu4(x)
        x = self.fc4(x)
        x = self.relu5(x)
        x = self.fc5(x)

        return self.sig(x)  # returns next state



class PairwiseIsActionTakenBLSTM(nn.Module):
    def __init__(self):
        super(PairwiseIsActionTakenBLSTM, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.relu1 = nn.ReLU()
        self.fc1a = nn.Linear(164, 128)
        self.relu1a = nn.ReLU()
        self.fc1b = nn.Linear(128, 128)
        self.relu1b = nn.ReLU()
        self.fc1c = nn.Linear(140, 64)
        self.relu1c = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 32)
        self.relu4 = nn.ReLU()
        self.LSTM_layer = nn.LSTMCell(input_size=32, hidden_size=32)
        self.hidden = (torch.randn(1, 32), torch.randn(1, 32))
        self.fc4 = nn.Linear(32, 32)
        self.relu5 = nn.ReLU()
        self.fc5 = nn.Linear(32, 1)
        self.sig = nn.Sigmoid()
        self.action_EmbeddingList = nn.ModuleList([EmbeddingModule() for i in range(1)])
        self.EmbeddingList = nn.ModuleList([EmbeddingModule(n=12) for i in range(1)]) # game

    def forward(self, x, w, previous_hidden_state):
        x = self.fc1(x)
        x = self.relu1(x)
        x = torch.cat([x, w], dim=0)
        x = self.fc1a(x)
        x = self.relu1a(x)
        x = self.fc1b(x)
        x = self.relu1b(x)
        w_game = self.EmbeddingList[0].embedding
        x = torch.cat([x, w_game.reshape(12)], dim=0)
        x = self.fc1c(x)
        x = self.relu1c(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu4(x)
        x, c_x = self.LSTM_layer(x.reshape(1,32), previous_hidden_state)
        self.hidden = (x, c_x)
        x = self.fc4(x)
        x = self.relu5(x)
        x = self.fc5(x)

        return self.sig(x)  # returns next state

class PairwiseIsActionTakenLSTM(nn.Module):
    def __init__(self):
        super(PairwiseIsActionTakenLSTM, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.relu1 = nn.ReLU()
        self.fc1a = nn.Linear(164, 128)
        self.relu1a = nn.ReLU()
        self.fc1b = nn.Linear(128, 128)
        self.relu1b = nn.ReLU()
        self.fc1c = nn.Linear(128, 64)
        self.relu1c = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 32)
        self.relu4 = nn.ReLU()
        self.LSTM_layer = nn.LSTMCell(input_size=32, hidden_size=32)
        self.hidden = (torch.randn(1, 32), torch.randn(1, 32))
        self.fc4 = nn.Linear(32, 32)
        self.relu5 = nn.ReLU()
        self.fc5 = nn.Linear(32, 1)
        self.sig = nn.Sigmoid()
        self.action_EmbeddingList = nn.ModuleList([EmbeddingModule() for i in range(1)])

    def forward(self, x, w, previous_hidden_state):
        x = self.fc1(x)
        x = self.relu1(x)
        x = torch.cat([x, w], dim=0)
        x = self.fc1a(x)
        x = self.relu1a(x)
        x = self.fc1b(x)
        x = self.relu1b(x)
        x = self.fc1c(x)
        x = self.relu1c(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu4(x)
        x, c_x = self.LSTM_layer(x.reshape(1,32), previous_hidden_state)
        self.hidden = (x, c_x)
        x = self.fc4(x)
        x = self.relu5(x)
        x = self.fc5(x)

        return self.sig(x)  # returns next state
