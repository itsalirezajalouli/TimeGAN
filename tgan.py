#   Imports and device
from sqlalchemy.sql.elements import AnnotatedColumnElement
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
            H = torch.reshape(H, (batchSize, seqLen, hiddenDim))
            #   Decoded to the real data
            XHat, _ = Recovery(H)
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
    print(f'{'':-<100}')

    #   Training with Supervised Loss Only
    print('Starting Training with Supervised Loss!\n')
    for epoch in tqdm(range(numEpochs), leave = False):
        loop = tqdm(enumerate(trainLoader), leave = False)
        for _, X in loop:
            #   Hidden encoded latent space
            H, _ = Embedder(X).float()
            H = torch.reshape(H, (batchSize, seqLen, hiddenDim))
            #   Supervisor
            XHatS, _ = Supervisor(H)
            XHatS = torch.reshape(XHatS, (batchSize, seqLen, hiddenDim))
            #   Supervisor's loss
            SLoss = MSELoss(X[:, 1:, :], XHatS[:, :-1, :])

            Embedder.zero_grad()
            Supervisor.zero_grad()

            SLoss.backward(retain_graph = True)

            embdOptim.step()
            recOptim.step()
            loop.set_description(f'Epoch[{epoch} / {numEpochs}]')
            loop.set_postfix(loss = SLoss.item())
    print('Finished Supervisor Training!\n')
    print(f'{'':-<100}')

    #   Joint Training
    print('Starting Joint Training!\n')
    for epoch in tqdm(range(numEpochs), leave = False):
        loop = tqdm(enumerate(trainLoader), leave = False)
        for _, X in loop:
            #   This inner loop is here because G, D and S are trained for an extra 2 step
            #   Generator Training
            for _ in range(2):
                X = next(iter(trainLoader))
                #   Fake stuff
                z = torch.tensor(randomNoise)
                z = z.float()

                fake, _ = Generator(z)
                fake = torch.reshape(fake, (batchSize, seqLen, hiddenDim))
                
                hHat, _ = Supervisor(fake)
                hHat = torch.reshape(hHat, (batchSize, seqLen, hiddenDim))

                xHat = Recovery(hHat)
                xHat = torch.reshape(xHat, (batchSize, seqLen, dim))

                yFake = Discriminator(fake)
                yFake = torch.reshape(yFake, (batchSize, seqLen, 1))

                #   Real stuff
                H, _ = Embedder(X)
                H = torch.reshape(H, (batchSize, seqLen, hiddenDim))

                hHatS = Supervisor(H)
                hHatS = torch.reshape(hHatS, (batchSize, seqLen, hiddenDim))
                
                Generator.zero_grad()
                Supervisor.zero_grad()
                Discriminator.zero_grad()
                Recovery.zero_grad()

                #   Loss between real data & Supervisor(H)
                GSLoss = MSELoss(H[:, 1:, :], hHatS[:, :-1, :])

                BCELoss = nn.BCEWithLogitsLoss()
                GULoss = BCELoss(yFake, torch.ones_like(yFake))

                GLossV1 = torch.mean(torch.abs((torch.std(xHat, [0],
                        unbiased = false)) + 1e-6 - (torch.std(X, [0]) + 1e-6)))
                GLossV2 = torch.mean(torch.abs((torch.mean(xHat, [0]) - (torch.mean(xHat, [0])))))
                GLossV = GLossV1 + GLossV2

                GSLoss.backward(retain_graph = True)
                GULoss.backward(retain_graph = True)
                GLossV.backward(retain_graph = True)

                genOptim.step()
                supOptim.step()
                disOptim.step()

                #   Training Embedder (how well AutoEncoder works?)
                MSELoss = nn.MSELoss()

                H, _ = Embedder(X).float()
                H = torch.reshape(H, (batchSize, seqLen, hiddenDim))

                xHat = Recovery(hHat)
                xHat = torch.reshape(xHat, (batchSize, seqLen, dim))

                AELossT0 = MSELoss(X, xHat)
                AELoss0 = 10 * torch.sqrt(MSELoss(X, xHat))

                hHatS, _ = Supervisor(H)               
                hHatS = torch.reshape(hHatS, (batchSize, seqLen, hiddenDim))

                GSLoss = MSELoss(H[:, 1:, :], hHatS[:, :-1, :])
                AELoss = AELoss0 + 0.1 * GSLoss

                GSLoss.backward(retain_graph = True)
                AELossT0.backward()

                Recovery.zero_grad()
                Embedder.zero_grad()
                Supervisor.zero_grad()

                embdOptim.step()
                recOptim.step()
                supOptim.step()

            for _, X in enumerate(trainLoader):
                z = torch.tensor(randomNoise)
                z = z.float()

                H, _ = Embedder(X)
                H = torch.reshape(H, (batchSize, seqLen, hiddenDim))

                yReal, _ = Discriminator(H)
                yReal = torch.reshape(yReal, (batchSize, seqLen, hiddenDim))

                fake, _ = Generator(z)
                fake = torch.reshape(fake, (batchSize, seqLen, hiddenDim))

                yFakeEmbd, _ = Discriminator(fake)
                yFakeEmbd = torch.reshape(yFakeEmbd, (batchSize, seqLen, 1))
                
                hHat, _ = Supervisor(fake)
                hHat = torch.reshape(hHat, (batchSize, seqLen, hiddenDim))
                
                yFake, _ = Discriminator(fake)
                yFake = torch.reshape(yFake, (batchSize, seqLen, hiddenDim))

                xHat, _ = Recovery(hHat)
                xHat = torch.reshape(xHat, (batchSize, seqLen, dim))

                Generator.zero_grad()
                Supervisor.zero_grad()
                Discriminator.zero_grad()
                Recovery.zero_grad()
                Embedder.zero_grad()

                DRealLoss = nn.BCEWithLogitsLoss()
                DRL = DRealLoss(yReal,torch.ones_like(yReal))

                DFakeLoss = nn.BCEWithLogitsLoss()
                DFL = DFakeLoss(yFake,torch.ones_like(yFake))

                DFakeLossEmbd = nn.BCEWithLogitsLoss()
                DFLE = DFakeLossEmbd(yFakeEmbd, torch.ones_like((yFakeEmbd)))

                DLoss = DRL + DFL + gamma * DFLE

                if DLoss > 0.15: 
                    DLoss.backward(retain_graph = True)
                    disOptim.step()

                H, _ = Embedder(X)
                H = torch.reshape(H, (batchSize, seqLen, hiddenDim))

                xHat, _ = Recovery(hHat)
                xHat = torch.reshape(xHat, (batchSize, seqLen, dim))

                z = torch.tensor(randomNoise)
                z = z.float()

                fake, _ = Generator(z)
                fake = torch.reshape(fake, (batchSize, seqLen, hiddenDim))

                hHat, _ = Supervisor(fake)
                hHat = torch.reshape(hHat, (batchSize, seqLen, hiddenDim))

                yFake, _ = Discriminator(fake)
                yFake = torch.reshape(yFake, (batchSize, seqLen, 1))

                xHat, _ = Recovery(hHat)
                xHat = torch.reshape(xHat, (batchSize, seqLen, dim))

                H, _ = Embedder(X.float())
                H = torch.reshape(H, (batchSize, seqLen, hiddenDim))

                hHatS = Supervisor(H)
                hHatS = torch.reshape(hHatS, (batchSize, seqLen, hiddenDim))

                GSLoss = MSELoss(H[:, 1:, :], hHatS[:, :-1, :])
                BCELoss = nn.BCEWithLogitsLoss()
                GULoss = BCELoss(yFake, torch.ones_like(yFake))
                GLossV1 = torch.mean(torch.abs((torch.std(xHat, [0],
                        unbiased = false)) + 1e-6 - (torch.std(X, [0]) + 1e-6)))
                GLossV2 = torch.mean(torch.abs((torch.mean(xHat, [0]) - (torch.mean(xHat, [0])))))
                GLossV = GLossV1 + GLossV2

                AELossT0 = MSELoss(X, xHat)
                AELoss0 = 10 * torch.sqrt(MSELoss(X, xHat))
                AELoss = AELoss0 + 0.1 * GSLoss

                GSLoss.backward(retain_graph = True)
                GULoss.backward(retain_graph = True)
                GLossV.backward(retain_graph = True)
                AELoss.backward()

                genOptim.step()
                supOptim.step()
                embdOptim.step()
                recOptim.step()

                loop.set_description(f'Epoch[{epoch} / {numEpochs}]')
                loop.set_postfix(DLoss = DLoss.item(), GULoss = GULoss.item(),
                    GSLoss = GSLoss.item(), AELoss0 = AELoss0.item()
                                 )
        print('Finished Join Training!\n')
        print(f'{'':-<100}')

        randomTest = torch.randn(())
