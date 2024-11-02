#   Imports and device
import torch
import numpy as np
import pandas as pd
from torch.nn.modules import rnn
from tqdm import tqdm
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

#   Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#   TimeGAN
class TimeGANModule(nn.Module):
    def __init__(self, inputSize, outputSize, hiddenDim, nLayers,
                 activationFunc = torch.sigmoid, rnnType = 'GRU') -> None:
        super(TimeGANModule, self).__init__()
        #   Params
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenDim = hiddenDim
        self.actFunc = activationFunc
        self.rnnType = rnnType
        self.nLayers = nLayers
        #   RNN Layer
        match rnnType:
            case 'RNN':
                self.rnn = nn.RNN(inputSize, hiddenDim, nLayers, batch_first = True)
            case 'LSTM':
                self.rnn = nn.LSTM(inputSize, hiddenDim, nLayers, batch_first = True)
            case 'GRU':
                self.rnn = nn.GRU(inputSize, hiddenDim, nLayers, batch_first = True)
        #   FC Layer 
        self.fc = nn.Linear(hiddenDim, outputSize)

    def forward(self, X):
        batchSize = X.size(0)
        if self.rnnType == 'LSTM':
            #   Hidden State & Cell State
            h0 = torch.zeros(2 * self.nStacked, batchSize, self.hiddenDim).to(device).float()
            c0 = torch.zeros(2 * self.nStacked, batchSize, self.hiddenDim).to(device).float()
            hidden = (h0, c0)
        else:
            hidden = torch.zeros(self.nStacked, batchSize, self.hiddenDim).to(device).float()
        out, hidden = self.rnn(X, hidden)

        out = out.contiguous().view(-1, self.hiddenDim)
        out = self.fc(out)

        if self.actFunc == self.Identity:
            identity = nn.Identity()
            return identity(out)

        out = self.actFunc(out)
        return out, hidden

def TimeGAN(data, params):
    #   Params
    hiddenDim = params['hiddenDim']
    nLayer = params['nLayers']
    iters = params['iters']
    batchSize = params['batchSize']
    module = params['module']
    numEpochs = params['numEpochs']
    no, seqLen, dim = np.asarray(data).shape
    zDim = dim
    gamma = 1
    checkPoints = {}

    #   Models
    Embedder = TimeGANModule(zDim, hiddenDim, hiddenDim, nLayer)
    Recovery = TimeGANModule(hiddenDim, dim, hiddenDim, nLayer)
    Generator = TimeGANModule(dim, hiddenDim, hiddenDim, nLayer)
    Supervisor = TimeGANModule(hiddenDim, hiddenDim, hiddenDim, nLayer)
    Discriminator = TimeGANModule(hiddenDim, 1, hiddenDim, nLayer, nn.Identity)

    #   Optimizers
    embdOptim = optim.Adam(Embedder.parameters(), lr = 0.001)
    recOptim = optim.Adam(Recovery.parameters(), lr = 0.001)
    genOptim = optim.Adam(Generator.parameters(), lr = 0.001)
    supOptim = optim.Adam(Supervisor.parameters(), lr = 0.001)
    disOptim = optim.Adam(Discriminator.parameters(), lr = 0.001)
    BCELoss = nn.BCEWithLogitsLoss()
    MSELoss = nn.MSELoss()
    trainLoader = DataLoader(data, params['batchSize'], shuffle = True)
    randomNoise = torch.randn(())

    #   Embedding Network Training (AutoEncoder / Embedder + Recovery Training)
    print('Started Embedding Network Training!\n')
    for epoch in tqdm(range(numEpochs), leave = False):
        loop = tqdm(enumerate(trainLoader), leave = False)
        for _, X in loop:
            #   Hidden encoded latent space
            H, _ = Embedder(X).float()
            H = torch.reshape(H, (batchSize, seqLen, dim))
            #   Decoded to the real data
            XHat, _ = Recovery(H).float()
            XHat = torch.reshape(XHat, (batchSize, seqLen, dim))
            #   AutoEncoder section's loss
            AELoss = 10 * torch.sqrt(MSELoss(X, XHat))

            Embedder.zero_grad()
            Recovery.zero_grad()

            AELoss.backward(retain_graph = True)

            embdOptim.step()
            recOptim.step()
            loop.set_description(f'Epoch[{epoch} / {numEpochs}]')
            loop.set_postfix(loss = AELoss.item())
    print('Finished Embedding Network Training!\n')
