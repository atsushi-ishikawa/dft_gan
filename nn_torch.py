import glob
import os
import pandas as pd
import torch
import torch.nn as nn
from tools import load_ase_json
from ase.db import connect
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from ase.build import fcc111
from ase.visualize import view

surf_json = "surf.json"
reac_json = "reaction_energy.json"
# argvs = sys.argv
# surface_data = str(argvs[1])
# reaction_data = str(argvs[2])
#
# load data and put it to DataFrame
#
df1 = load_ase_json(surf_json)
df2 = pd.read_json(reac_json)

df1 = df1.set_index("unique_id")
df2 = df2.set_index("unique_id")
df  = pd.concat([df1, df2], axis=1)
df  = df.sort_values("reaction_energy")
numdata = len(df)
#
# parameters
#
numuse     = int(numdata * 1.0)
nclass     = 10
log_dir    = "./log"
numepochs  = 200
printnum   = 50
batch_size = 30
z_dim      = 100
lr         = 1.0e-3
adam_b1    = 0.5
adam_b2    = 0.999
scaler     = StandardScaler()

if os.path.exists(log_dir):
    files = glob.glob(os.path.join(log_dir, "*.png"))
    for f in files:
        os.remove(f)
else:
    os.makedirs(log_dir)
#
# divide into groups according to adsorption energy
#
rank = pd.qcut(df.reaction_energy, nclass, labels=False)
df["rank"] = rank
print(df.head(numuse // 2 + 1))


def make_dataloader(x=None, y=None, batch_size=10):
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    global scaler

    x = [np.array(i) for i in x]  # convert to numpy
    for i, j in enumerate(x):
        x[i] = scaler.fit_transform(x[i].reshape(-1, 1))

    x = torch.tensor(x).float()
    y = torch.tensor(y).float()

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


# dataloader = make_dataloader(x=df["atomic_numbers"], y=df["reaction_energy"], batch_size=batch_size)
dataloader = make_dataloader(x=df["atomic_numbers"], y=df["rank"], batch_size=batch_size)

nchannel = 64
nstride = 2
natom = len(df.iloc[0]["atomic_numbers"])
nrun = df["run"].max()


class Discriminator(nn.Module):
    #
    # returns one (stable structure) or zero (unstable structure)
    #
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            ## CNN-like
            nn.Conv1d(1 + nclass, nchannel, kernel_size=4, stride=nstride, padding=1),
            nn.BatchNorm1d(nchannel),
            nn.LeakyReLU(0.2),
            nn.Conv1d(nchannel, nchannel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(nchannel),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear((1+nclass)*natom, 2*nchannel),
            nn.BatchNorm1d(2*nchannel),  # need
            nn.LeakyReLU(0.2),  # need

            nn.Linear(2*nchannel, 3*nchannel),
            nn.BatchNorm1d(3 * nchannel),  # need
            nn.LeakyReLU(0.2),  # need

            nn.Linear(3*nchannel, 1),

            #nn.Linear(natom // nstride * nchannel, 2 * nchannel),
            #nn.BatchNorm1d(2 * nchannel),  # seems unnecessary
            #nn.LeakyReLU(0.2),
            #nn.Linear(2 * nchannel, 1),

            nn.Sigmoid(),
        )

    def forward(self, input):
        # x = self.conv(input)
        x = input
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


class Generator(nn.Module):
    #
    # generate atomic number sequence that seems to be stable
    #
    def __init__(self):
        super().__init__()
        n_feature = z_dim * (nclass + 1)
        self.conv = nn.Sequential(
            nn.Linear(n_feature, 2*n_feature),
            nn.BatchNorm1d(2*n_feature),
            # nn.Linear(n_feature, 2*n_feature),
            # nn.BatchNorm1d(2*n_feature),
            # nn.ReLU(),
            # nn.Linear(2*n_feature,  2*n_feature),
            # nn.BatchNorm1d(2*n_feature),
            # nn.ReLU(),
            # nn.Linear(2*n_feature, natom),  # do not use activation after linear

            nn.Linear(2*n_feature, natom),  # do not use activation after linear
        )

    def forward(self, input):
        input = input.view(batch_size, -1)
        x = self.conv(input)
        x = x.view(batch_size, -1, 1)  # need to be 3D to include label information
        return x


def load_checkpoint(model, optimizer, filename):
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        print("no checkpoint found")


def onehot_encode(label, nclass):
    eye = torch.eye(nclass)
    return eye[label].view(-1, nclass, 1)


def concat_vector_label(vector, label, nclass):
    N, C, L = vector.shape
    vector = vector.view(N, L, C)
    oh_label = onehot_encode(label, nclass)
    oh_label = oh_label.expand(N, nclass, C)
    result = torch.cat((vector, oh_label), dim=1)
    return result


criterion = nn.MSELoss()
#
# define model and optimizer
#
D = Discriminator()
G = Generator()

D_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(adam_b1, adam_b2))
G_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(adam_b1, adam_b2))
#
# load state
#
D_file = os.path.join(log_dir, "D_last.pth")
G_file = os.path.join(log_dir, "G_last.pth")
load_checkpoint(D, D_opt, D_file)
load_checkpoint(G, G_opt, G_file)


def train(D, G, criterion, D_opt, G_opt, dataloader):
    D.train()
    G.train()

    y_real = torch.ones(batch_size, 1)
    y_fake = torch.zeros(batch_size, 1)

    D_running_loss = 0.0
    G_running_loss = 0.0

    for batch_idx, (real_system, label) in enumerate(dataloader):
        if real_system.size()[0] != batch_size: break
        z = torch.rand(batch_size, z_dim, 1)
        label = label.long()
        real_system_label = concat_vector_label(real_system, label, nclass)
        #
        # updating Discriminator
        #
        D_opt.zero_grad()
        D_real = D(real_system_label)
        D_real_loss = criterion(D_real, y_real)
        # D_real_loss = torch.sum((D_real - 1.0)**2) # LSGAN

        z_label = concat_vector_label(z, label, nclass)
        fake_system = G(z_label)
        fake_system_label = concat_vector_label(fake_system, label, nclass)
        D_fake = D(fake_system_label.detach())
        D_fake_loss = criterion(D_fake, y_fake)
        # D_fake_loss = torch.sum((D_fake - 0.0)**2) # LSGAN

        D_loss = D_real_loss + D_fake_loss
        # D_loss /= batch_size
        D_loss.backward()
        D_opt.step()
        D_running_loss += D_loss.item()
        #
        # updating Generator
        #
        z = torch.rand(batch_size, z_dim, 1)
        z_label = concat_vector_label(z, label, nclass)
        G_opt.zero_grad()
        fake_system = G(z_label)
        fake_system_label = concat_vector_label(fake_system, label, nclass)
        D_fake = D(fake_system_label)
        G_loss = criterion(D_fake, y_real)
        # G_loss = torch.sum((D_fake - 1.0)**2) # LSGAN
        # G_loss /= batch_size

        G_loss.backward()
        G_opt.step()
        G_running_loss += G_loss.item()

    return D_running_loss, G_running_loss


def generate(G, target=0):
    G.eval()
    z = torch.rand(batch_size, z_dim, 1)
    z_label = concat_vector_label(z, target, nclass)
    fake = G(z_label)
    fake = fake.detach().numpy()
    fake = scaler.inverse_transform(fake)
    return fake


def gan(numepochs=100):
    global D, G, criterion, D_opt, G_opt, dataloader

    history = {"D_loss": [], "G_loss": []}
    for epoch in range(numepochs):
        D_loss, G_loss = train(D, G, criterion, D_opt, G_opt, dataloader)
        history["D_loss"].append(D_loss)
        history["G_loss"].append(G_loss)

        if epoch != 0 and epoch % printnum == 0:
            print("epoch = %3d, D_loss = %8.5f, G_loss = %8.5f" % (epoch, D_loss, G_loss))

    plt.figure()
    plt.plot(range(numepochs), history["D_loss"], "r-", label="Discriminator loss")
    plt.plot(range(numepochs), history["G_loss"], "b-", label="Generator loss")
    plt.legend()
    plt.savefig(os.path.join(log_dir, "loss.png"))
    plt.close()


def make_atomic_numbers(inputlist, reflist):
    """
    :param inputlist:
    :param reflist:
    :return: newlist
    """
    AN = {"Pd": 46, "Pt": 78}  # atomic numbers
    # 3D --> 2D
    if len(inputlist.shape) == 3:
        inputlist = inputlist.reshape(batch_size, -1)

    tmplist = inputlist.astype(int).tolist()  # float --> int --> python list
    tmplist = [list(map(lambda x: AN["Pt"] if x > 70 else AN["Pd"], i)) for i in tmplist]

    reflist = reflist.values.tolist()
    #
    # make uniquelist
    #
    newlist = []
    for i in tmplist:
        i = list(i)
        #if i not in newlist:
        if i not in reflist:
            newlist.append(i)
            reflist.append(i)
            print("found new one: ", i)

    return newlist


gan(numepochs=numepochs)
#
# save state
#
torch.save({"state_dict": D.state_dict(), "optimizer": D_opt.state_dict()}, D_file)
torch.save({"state_dict": G.state_dict(), "optimizer": G_opt.state_dict()}, G_file)

fakesystem = []
for target in range(nclass):
    fakesystem.append(generate(G, target=target))

# print(fakesystem[0][0].astype(int).reshape(1, -1))
samples = make_atomic_numbers(fakesystem[0], df["atomic_numbers"])
#
# Make fake examples: need some template -- should be fixed
#
surf = fcc111(symbol="Pd", size=[4, 4, 4], a=4.0, vacuum=10.0)
check = False
write = True
db = connect(surf_json)  # add to existing file

for sample in samples:
    surf.set_atomic_numbers(sample)
    atomic_numbers = surf.get_atomic_numbers()
    formula = surf.get_chemical_formula()

    print("formula: ", surf.get_chemical_formula())
    if check: view(surf)
    data = {"chemical_formula": formula, "atomic_numbers": atomic_numbers, "run": nrun + 1}
    #
    # write candidate to file
    #
    if write:
        db.write(surf, data=data)
