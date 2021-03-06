import Model2
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image

G = torch.load("Celeba_gen100")

latent_dim = 62
n_classes = 5
code_dim = 2
img_size = 32
channels = 3
batch_size = 128
lr = 0.0002
b1 = 0.5
b2 = 0.999
num_epochs = 12

cuda = True if torch.cuda.is_available() else False

if cuda:
    G.cuda()

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return Variable(FloatTensor(y_cat))


# Static generator inputs for sampling
static_z = Variable(FloatTensor(np.zeros((n_classes ** 2, latent_dim))))
static_label = to_categorical(
    np.array([num for _ in range(n_classes) for num in range(n_classes)]), num_columns=n_classes
)
static_code = Variable(FloatTensor(np.zeros((n_classes ** 2, code_dim))))



def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Static sample
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, latent_dim))))
    static_sample = G(z, static_label, static_code)
    save_image(static_sample.data, "static_x.png", nrow=n_row, normalize=True)

    # Get varied c1 and c2
    zeros = np.zeros((n_row ** 2, 1))
    c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
    c1 = Variable(FloatTensor(np.concatenate((c_varied, zeros), -1)))
    c2 = Variable(FloatTensor(np.concatenate((zeros, c_varied), -1)))
    sample1 = G(static_z, static_label, c1)
    sample2 = G(static_z, static_label, c2)
    save_image(sample1.data, "c1.png", nrow=n_row, normalize=True)
    save_image(sample2.data, "c2.png", nrow=n_row, normalize=True)


sample_image(n_row=5, batches_done=2)
