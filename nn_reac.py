from tools import load_ase_json
from ase.db import connect
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from ase.build import fcc111, fcc211
from ase.visualize import view
import glob
import os
import pandas as pd
import torch
import torch.nn as nn
import h5py
import numpy as np
import argparse

debug = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device is %s" % device)

# set random number seed
seed = 0
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--surf_json", default="surf.json", help="json for surfaces")
parser.add_argument("--reac_json", default="reaction_energy.json", help="json for reaction energies")
parser.add_argument("--loss_file", default="loss.h5", help="file for generator and descriminator losses")

args = parser.parse_args()
surf_json = args.surf_json
reac_json = args.reac_json
loss_file = args.loss_file

if not os.path.exists(loss_file):
	# prepare loss h5 file if not exits
	h5file = h5py.File(loss_file, "w")
	h5file.create_dataset("epoch", (1,),  maxshape=(None, ), chunks=True, dtype="int")
	h5file.create_dataset("D_loss", (1,), maxshape=(None, ), chunks=True, dtype="float")
	h5file.create_dataset("G_loss", (1,), maxshape=(None, ), chunks=True, dtype="float")
	h5file.flush()
	h5file.close()

#
# load data and put it to DataFrame
#
df1 = load_ase_json(surf_json)
df2 = pd.read_json(reac_json)

df1 = df1.set_index("unique_id")
df2 = df2.set_index("unique_id")
df  = pd.concat([df1, df2], axis=1)
df  = df.sort_values("score", ascending=False)

if debug:
	print("best score:",  df.iloc[0]["score"])
	print("worst score:", df.iloc[-1]["score"])
#
# droping NaN in atomic numbers and score
#
df = df.dropna(subset=["atomic_numbers"])
df = df.dropna(subset=["score"])
numdata = len(df)
#
# parameters
#
numuse     = int(numdata * 1.0)
nclass     = 10  # 3 --- uniform distribution.  15,20 --- not good atomic numbers
num_epoch  = 1000 # 1000 seems better than 500
printnum   = 50
batch_size = numdata//5  # from experience
z_dim = 100
lr = 1.0e-3
b1 = 0.5
b2 = 0.999
dropoutrate = 0.3

scaler_selection = "minmax"
#scaler_selection = "standard"

if scaler_selection == "minmax":
	scaler  = MinMaxScaler()  # makes (0,1) for binary system
else:
	scaler  = StandardScaler()

#
# make logdir if not exists
#
log_dir  = "./log"
if not os.path.exists(log_dir):
	os.makedirs(log_dir)
#
# divide into groups according to score
# group with highest score = 0, lowest score = nclass-1
#
rank = pd.qcut(df["score"], nclass, labels=False)
df["rank"] = rank


def make_dataloader(x=None, y=None, batch_size=10):
	import numpy as np
	import torch
	from torch.utils.data import TensorDataset, DataLoader

	global scaler

	x = [np.array(i) for i in x]  # convert to numpy
	for i, j in enumerate(x):
		x[i] = scaler.fit_transform(x[i].reshape(-1, 1))

	x = torch.tensor(x, device=device).float()
	y = torch.tensor(y, device=device).float()

	dataset = TensorDataset(x, y)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	return dataloader


dataloader = make_dataloader(x=df["atomic_numbers"], y=df["rank"], batch_size=batch_size)

nchannel = 64
nstride  = 3
natom = len(df.iloc[0]["atomic_numbers"])
nrun  = df["run"].max()


class Discriminator(nn.Module):
	#
	# returns one (stable structure) or zero (unstable structure)
	#
	def __init__(self):
		super().__init__()

		self.fc = nn.Sequential(
			#nn.utils.spectral_norm(nn.Linear((1 + nclass) * natom, 2*nchannel)), # test
			nn.Linear((1 + nclass)*natom, 2*nchannel),
			nn.BatchNorm1d(2*nchannel),  # need
			nn.LeakyReLU(0.2),  # need
			nn.Dropout(dropoutrate), # test

			#nn.utils.spectral_norm(nn.Linear((1 + nclass) * natom, 2*nchannel)), # test
			nn.Linear(2*nchannel, 2*nchannel),
			nn.BatchNorm1d(2 * nchannel),  # need
			nn.LeakyReLU(0.2),  # need
			nn.Dropout(dropoutrate), # test

			nn.Linear(2*nchannel, 1),

			nn.Sigmoid(),
		)

	def forward(self, input):
		x = input  # when skipping conv
		x = x.view(batch_size, -1)
		x = self.fc(x)
		return x


class Generator(nn.Module):
	#
	# generate atomic number sequence that gives
	# desirable property (e.g. reaction energy)
	#
	# note: LeakyReLU is better (2021/04/14)
	#
	def __init__(self):
		super().__init__()

		n_feature = z_dim * (nclass + 1)
		self.conv = nn.Sequential(
			nn.Linear(n_feature, 2*n_feature),  # 2*n-->4*n is not good
			nn.BatchNorm1d(2*n_feature),
			nn.LeakyReLU(0.2),
			nn.Dropout(dropoutrate),

			nn.Linear(2*n_feature, 2*n_feature),
			nn.BatchNorm1d(2*n_feature),
			nn.LeakyReLU(0.2),
			nn.Dropout(dropoutrate),  # temporary killed (05/19)

			nn.Linear(2*n_feature, natom),
			nn.Sigmoid()  # output as (0,1)
		)

	def forward(self, input):
		input = input.view(batch_size, -1)
		x = self.conv(input)
		x = x.view(batch_size, -1, 1)  # need to be 3D to include label information
		return x


def load_checkpoint(model, optimizer, filename, device):
	if os.path.isfile(filename):
		print("=> loading checkpoint '{}'".format(filename))
		checkpoint = torch.load(filename, map_location=device)
		model.load_state_dict(checkpoint["state_dict"])
		optimizer.load_state_dict(checkpoint["optimizer"])
	else:
		print("no checkpoint found")


def onehot_encode(label, nclass, device):
	eye = torch.eye(nclass, device=device)
	return eye[label].view(-1, nclass, 1)


def concat_vector_label(vector, label, nclass, device):
	N, C, L = vector.shape
	vector = vector.view(N, L, C)
	oh_label = onehot_encode(label, nclass, device)
	oh_label = oh_label.expand(N, nclass, C)
	result = torch.cat((vector, oh_label), dim=1)
	return result


criterion = nn.MSELoss()
#
# define model and optimizer
#
D = Discriminator().to(device)
G = Generator().to(device)

D_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(b1, b2))
G_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(b1, b2))
#
# load state
#
D_file = os.path.join(log_dir, "D_last.pth")
G_file = os.path.join(log_dir, "G_last.pth")
load_checkpoint(D, D_opt, D_file, device)
load_checkpoint(G, G_opt, G_file, device)


def train(D, G, criterion, D_opt, G_opt, dataloader):
	D.train()
	G.train()

	y_real = torch.ones(batch_size, 1, device=device)
	y_fake = torch.zeros(batch_size, 1, device=device)

	D_running_loss = 0.0
	G_running_loss = 0.0

	for batch_idx, (real_system, label) in enumerate(dataloader):
		if real_system.size()[0] != batch_size: break
		z = torch.randn(batch_size, z_dim, 1, device=device)  # randn is better than rand
		label = label.long()
		real_system_label = concat_vector_label(real_system, label, nclass, device)
		#
		# updating Discriminator
		#
		D_opt.zero_grad()
		D_real = D(real_system_label)
		D_real_loss = criterion(D_real, y_real)

		z_label = concat_vector_label(z, label, nclass, device)
		fake_system = G(z_label)
		fake_system_label = concat_vector_label(fake_system, label, nclass, device)
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
		z = torch.randn(batch_size, z_dim, 1, device=device)
		z_label = concat_vector_label(z, label, nclass, device)
		G_opt.zero_grad()
		fake_system = G(z_label)
		fake_system_label = concat_vector_label(fake_system, label, nclass, device)
		D_fake = D(fake_system_label)
		G_loss = criterion(D_fake, y_real)
		# G_loss = torch.sum((D_fake - 1.0)**2) # LSGAN
		# G_loss /= batch_size

		G_loss.backward()
		G_opt.step()
		G_running_loss += G_loss.item()

	return D_running_loss, G_running_loss


def generate(G, target=0):
	global scaler, scaler_selection

	scaler2 = MinMaxScaler()

	G.eval()
	z = torch.randn(batch_size, z_dim, 1, device=device)
	z_label = concat_vector_label(z, target, nclass, device)
	fake = G(z_label)
	fake = fake.detach().cpu().numpy()

	if scaler_selection=="minmax":
		fake = np.array(list(map(scaler2.fit_transform, fake)))  # convert to (0,1)
	else:
		fake = scaler.inverse_transform(fake)

	return fake

def gan(num_epoch=100):
	global D, G, criterion, D_opt, G_opt, dataloader
	
	history = {"D_loss": [], "G_loss": []}
	for epoch in range(num_epoch):
		D_loss, G_loss = train(D, G, criterion, D_opt, G_opt, dataloader)
		history["D_loss"].append(D_loss)
		history["G_loss"].append(G_loss)

		if epoch != 0 and epoch % printnum == 0:
			print("epoch = %3d, D_loss = %8.5f, G_loss = %8.5f" % (epoch, D_loss, G_loss))

	with h5py.File(loss_file, "a") as f:
		size_resize = int(f["epoch"].shape[0] + num_epoch)
		f["epoch"].resize(size_resize,  axis=0)
		f["D_loss"].resize(size_resize, axis=0)
		f["G_loss"].resize(size_resize, axis=0)

		f["epoch"][:]  = list(range(size_resize))
		f["D_loss"][-num_epoch:] = history["D_loss"]
		f["G_loss"][-num_epoch:] = history["G_loss"]


def make_atomic_numbers(inputlist, oldlist):
	"""
	Assuming atomic number sequence is transformed to (0,1).
	:param inputlist
	:param oldlist
	:return: newlist
	"""
	global scaler_selection

	#elements = ["Ru", "Pt"]
	#elements = ["Ru", "Ni"]
	elements = ["Ru", "Pd"]

	AN = {"Ni": 28, "Ru": 44, "Rh": 45, "Pd": 46, "Pt": 78}
	elements = list(map(lambda x: AN[x], elements))
	elements = sorted(elements)

	# 3D --> 2D
	if len(inputlist.shape) == 3:
		inputlist = inputlist.reshape(batch_size, -1)

	lists = inputlist.astype(int).tolist()  # float --> int --> python list

	if debug:
		print("elements:", elements)
		print(lists[0])

	if scaler_selection == "minmax":
		lists = [list(map(lambda x: elements[0] if x < 0.5 else elements[-1], ilist)) for ilist in lists]
	else:
		lists = [list(map(lambda x: elements[0] if x < np.mean(list) else elements[-1], ilist)) for ilist in lists]

	if debug:
		print(lists[0])

	oldlist = oldlist.values.tolist()
	#
	# make uniquelist
	#
	newlist = []
	for ilist in lists:
		ilist = list(ilist)
		# if i not in newlist:
		if ilist not in oldlist:
			newlist.append(ilist)
			oldlist.append(ilist)

	return newlist


gan(num_epoch=num_epoch)
#
# save state
#
torch.save({"state_dict": D.state_dict(), "optimizer": D_opt.state_dict()}, D_file)
torch.save({"state_dict": G.state_dict(), "optimizer": G_opt.state_dict()}, G_file)
#
#target_class = nclass-1
target_class = 0
fakesystem = []
for target in range(nclass):
	fakesystem.append(generate(G, target=target))

samples = make_atomic_numbers(fakesystem[target_class], df["atomic_numbers"])
#
# Make fake examples: need some template -- should be fixed
#
vacuum = 9.0
a = 2.7*1.3
#surf = fcc111(symbol="Au", size=[2, 2, 5], a=a, vacuum=vacuum)
#surf = fcc111(symbol="Au", size=[3, 3, 4], a=a, vacuum=vacuum)
surf = fcc211(symbol="Au", size=[6, 3, 4], a=a, vacuum=vacuum)
surf.pbc = True
surf.translate([0, 0, -vacuum+1.0])

check = False
write = True
db = connect(surf_json, type="json")  # add to existing file

for sample in samples:
	surf.set_atomic_numbers(sample)
	atomic_numbers = surf.get_atomic_numbers()
	atomic_numbers = list(atomic_numbers)   # make non-numpy
	formula = surf.get_chemical_formula()

	print("formula: ", surf.get_chemical_formula())
	if check: view(surf)
	data = {"chemical_formula": formula, "atomic_numbers": atomic_numbers, "run": nrun + 1}
	#
	# write candidate to file
	#
	if write:
		db.write(surf, data=data)
