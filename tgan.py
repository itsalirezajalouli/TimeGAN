# Imports and device
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from copy import deepcopy
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
data = pd.read_csv('./AMZN.csv')
data = data[['Date', 'Close']]
data['Date'] = pd.to_datetime(data['Date'])

# Lookback function
def lookback(df, nsteps: int, label: str) -> pd.DataFrame:
    df = deepcopy(df)
    df.set_index('Date', inplace=True)
    for i in range(1, nsteps + 1):
        df[f'{label}(t-{i})'] = df[label].shift(i)
    # Drop NaNs
    df.dropna(inplace=True)
    return df

hist = 7
newdata = lookback(data, hist, 'Close')
npdata = newdata.to_numpy()

# Scale
scaler = MinMaxScaler(feature_range=(-1, 1))
npdata = scaler.fit_transform(npdata)

# Features & targets
x = npdata[:, 1:]
x = deepcopy(np.flip(x, axis=1))  # LSTM processes from oldest to latest summary
y = npdata[:, 0]  # Starting from -7 to -1

# Split
splitidx = int(len(x) * 0.95)
xtrain = x[:splitidx]
xtest = x[splitidx:]
ytrain = y[:splitidx]
ytest = y[splitidx:]

# PyTorch LSTM requires an extra dimension
xtrain = xtrain.reshape((-1, hist, 1))
xtest = xtest.reshape((-1, hist, 1))
ytrain = ytrain.reshape((-1, 1))
ytest = ytest.reshape((-1, 1))

# Move to torch tensors
xtrain = torch.tensor(xtrain).float()
xtest = torch.tensor(xtest).float()
ytrain = torch.tensor(ytrain).float()
ytest = torch.tensor(ytest).float()

# Dataset & loader
class TimeSeriesDataset(Dataset):
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

trainset = TimeSeriesDataset(xtrain, ytrain)
testset = TimeSeriesDataset(xtest, ytest)

class TimeGANModule(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers,
                 activation_func=torch.sigmoid, rnn_type='gru') -> None:
        super(TimeGANModule, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.act_func = activation_func
        self.rnn_type = rnn_type
        self.n_layers = n_layers

        # RNN layer
        if rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        if self.rnn_type == 'lstm':
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
            c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
            hidden = (h0, c0)
        else:
            hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)

        out, hidden = self.rnn(x, hidden)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        if self.act_func == nn.Identity():
            return out

        out = self.act_func(out)
        return out, hidden

def TimeGAN(data, params):
    # Params
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

    # Models
    Embedder = TimeGANModule(zDim, hiddenDim, hiddenDim, nLayer)
    Recovery = TimeGANModule(hiddenDim, dim, hiddenDim, nLayer)
    Generator = TimeGANModule(dim, hiddenDim, hiddenDim, nLayer)
    Supervisor = TimeGANModule(hiddenDim, hiddenDim, hiddenDim, nLayer)
    Discriminator = TimeGANModule(hiddenDim, 1, hiddenDim, nLayer, nn.Identity)

    # Optimizers
    embdOptim = optim.Adam(Embedder.parameters(), lr=0.001)
    recOptim = optim.Adam(Recovery.parameters(), lr=0.001)
    genOptim = optim.Adam(Generator.parameters(), lr=0.001)
    supOptim = optim.Adam(Supervisor.parameters(), lr=0.001)
    disOptim = optim.Adam(Discriminator.parameters(), lr=0.001)
    BCELoss = nn.BCEWithLogitsLoss()
    MSELoss = nn.MSELoss()
    #   If drop last is false embedding gets error
    trainLoader = DataLoader(data, params['batchSize'], shuffle=True, drop_last = True)

    # Embedding Network Training (AutoEncoder / Embedder + Recovery Training)
    print('Started Embedding Network Training!')
    for epoch in tqdm(range(numEpochs), leave=False):
        loop = tqdm(enumerate(trainLoader), leave=False)
        for _, X in loop:
            # Hidden encoded latent space
            H, _ = Embedder(X)
            H = torch.reshape(H, (batchSize, seqLen, hiddenDim))
            # Decoded to the real data
            XHat, _ = Recovery(H)
            XHat = torch.reshape(XHat, (batchSize, seqLen, dim))
            # AutoEncoder section's loss
            AELoss = 10 * torch.sqrt(MSELoss(X, XHat))

            Embedder.zero_grad()
            Recovery.zero_grad()

            AELoss.backward(retain_graph=True)

            embdOptim.step()
            recOptim.step()
            loop.set_description(f'Epoch[{epoch} / {numEpochs}]')
            loop.set_postfix(loss=AELoss.item())
    print('Finished Embedding Network Training!')
    print(f'{'':-<100}')

    # Training with Supervised Loss Only
    print('Starting Training with Supervised Loss!')
    for epoch in tqdm(range(numEpochs), leave=False):
        loop = tqdm(enumerate(trainLoader), leave=False)
        for _, X in loop:
            # Hidden encoded latent space
            H, _ = Embedder(X)
            H = torch.reshape(H, (batchSize, seqLen, hiddenDim))
            # Supervisor
            XHatS, _ = Supervisor(H)
            XHatS = torch.reshape(XHatS, (batchSize, seqLen, hiddenDim))
            # Supervisor's loss
            SLoss = MSELoss(X[:, 1:, :], XHatS[:, :-1, :])

            Embedder.zero_grad()
            Supervisor.zero_grad()

            SLoss.backward(retain_graph=True)

            embdOptim.step()
            recOptim.step()
            loop.set_description(f'Epoch[{epoch} / {numEpochs}]')
            loop.set_postfix(loss=SLoss.item())
    print('Finished Supervisor Training!')
    print(f'{'':-<100}')

    # Joint Training
    print('Starting Joint Training!')
    for epoch in tqdm(range(numEpochs), leave=False):
        loop = tqdm(enumerate(trainLoader), leave=False)
        for _, X in loop:
            # This inner loop is here because G, D and S are trained for an extra 2 step
            # Generator Training
            for _ in range(2):
                z = torch.randn(batchSize, seqLen, dim).to(device)
                # Fake stuff
                fake, _ = Generator(z)
                fake = torch.reshape(fake, (batchSize, seqLen, hiddenDim))
                
                hHat, _ = Supervisor(fake)
                hHat = torch.reshape(hHat, (batchSize, seqLen, hiddenDim))

                xHat, _ = Recovery(hHat)
                xHat = torch.reshape(xHat, (batchSize, seqLen, dim))

                yFake, _ = Discriminator(fake)
                yFake = torch.reshape(yFake, (batchSize, seqLen, 1))

                # Real stuff
                H, _ = Embedder(X)
                H = torch.reshape(H, (batchSize, seqLen, hiddenDim))

                hHatS = Supervisor(H)
                hHatS = torch.reshape(hHatS, (batchSize, seqLen, hiddenDim))
                
                Generator.zero_grad()
                Supervisor.zero_grad()
                Discriminator.zero_grad()
                Recovery.zero_grad()

                # Loss between real data & Supervisor(H)
                GSLoss = MSELoss(H[:, 1:, :], hHatS[:, :-1, :])

                GULoss = BCELoss(yFake, torch.ones_like(yFake))

                GLossV1 = torch.mean(torch.abs((torch.std(xHat, [0],
                        unbiased=False)) + 1e-6 - (torch.std(X, [0]) + 1e-6)))
                GLossV2 = torch.mean(torch.abs((torch.mean(xHat, [0]) - (torch.mean(xHat, [0])))))
                GLossV = GLossV1 + GLossV2

                GSLoss.backward(retain_graph=True)
                GULoss.backward(retain_graph=True)
                GLossV.backward(retain_graph=True)

                genOptim.step()
                supOptim.step()
                disOptim.step()

                # Training Embedder (how well AutoEncoder works?)
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

                GSLoss.backward(retain_graph=True)
                AELossT0.backward()

                Recovery.zero_grad()
                Embedder.zero_grad()
                Supervisor.zero_grad()

                embdOptim.step()
                recOptim.step()
                supOptim.step()

        # Test step
        with torch.no_grad():
            # Generate fake samples
            z = torch.randn(batchSize, seqLen, dim).to(device)
            fake, _ = Generator(z)
            fake = torch.reshape(fake, (batchSize, seqLen, dim))

            # Reconstruct real samples
            real = next(iter(trainLoader))
            H, _ = Embedder(real)
            H = torch.reshape(H, (batchSize, seqLen, hiddenDim))
            xHat, _ = Recovery(H)
            xHat = torch.reshape(xHat, (batchSize, seqLen, dim))

            # Calculate test losses
            testAELoss = MSELoss(real, xHat)
            testGLoss = MSELoss(real, fake)

        with torch.no_grad():
            z = torch.randn(xtrain.shape[0], xtrain.shape[1], xtrain.shape[2]).to(device)
            generated_data, _ = Generator(z)
            generated_data = generated_data.cpu().numpy()

            original_data = xtrain.cpu().numpy()  # Original training data for comparison

            # Plot original and generated data
            plt.figure(figsize=(14, 6))
            for i in range(min(len(original_data), 3)):  # Plot up to 3 examples for better visualization
                plt.subplot(3, 2, i*2 + 1)
                plt.plot(original_data[i].squeeze(), label='Original')
                plt.title(f'Original Data {i+1}')
                plt.xlabel('Time Steps')
                plt.ylabel('Values')

                plt.subplot(3, 2, i*2 + 2)
                plt.plot(generated_data[i].squeeze(), label='Generated', color='orange')
                plt.title(f'Generated Data {i+1}')
                plt.xlabel('Time Steps')
                plt.ylabel('Values')

                plt.tight_layout()
                plt.show()

            print(f"Epoch [{epoch}/{numEpochs}] - Test AE Loss: {testAELoss.item()}, Test G Loss: {testGLoss.item()}")

    print('Finished Join Training!')
    print(f'{'':-<100}')
# Generate and plot predictions next to original data at the end of training

# Parameters for the TimeGAN function
params = {
    'hiddenDim': 32,
    'nLayers': 2,
    'iters': 1000,
    'batchSize': 32,
    'module': 'gru',
    'numEpochs': 1
}

# Call the TimeGAN function to start training
TimeGAN(xtrain, params)
