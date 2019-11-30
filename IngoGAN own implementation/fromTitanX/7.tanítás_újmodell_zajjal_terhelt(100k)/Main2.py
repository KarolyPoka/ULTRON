import torch.nn as nn
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import itertools
import matplotlib.pyplot as plt
import progressbar
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
import random
import training
import Model2

if __name__ == "__main__":

    cuda = True if torch.cuda.is_available() else False

    np.random.seed(42)
    torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    random.seed(42)

    print("Cuda:%r" % cuda)
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    IMAGE_PATH = "/home/karcsi/celeba"
    #IMAGE_PATH = "C:/Users/Karcsi/Documents/celeba"
    IMG = "tony.jpg"

    hyp = {
        "latent_dim": 62,
        "n_classes": 10,
        "code_dim": 2,
        "img_size": 64,
        "channels": 3,
        "batch_size": 128,
        "lr": 0.0002,
        "b1": 0.5,
        "b2": 0.999,
        "num_epochs": 17,

        # info loss weights
        "lambda_cat": 1,
        "lambda_con": 0.1
    }



    generator = Model2.generator(
        hyp['latent_dim'],
        hyp['n_classes'],
        hyp['code_dim'],
        hyp['img_size'],
        hyp['channels']
    )
    discriminator = Model2.discriminator(
        hyp['latent_dim'],
        hyp['n_classes'],
        hyp['code_dim'],
        hyp['img_size'],
        hyp['channels']
    )

    # Loss functions
    losses = {
        'classifier_loss': nn.BCELoss(),
        'categorical_loss': nn.CrossEntropyLoss(),
        'continuous_loss': nn.MSELoss(),
        'noise_loss': nn.MSELoss()
    }

    # convert everything to CUDA
    if cuda:
        generator.cuda()
        discriminator.cuda()
        losses['classifier_loss'].cuda()
        losses['categorical_loss'].cuda()
        losses['continuous_loss'].cuda()
        losses['noise_loss'].cuda()

    # Optimizers new see
    optimizers = {
        "G": torch.optim.Adam(generator.parameters(), lr=hyp["lr"], betas=(hyp["b1"], hyp["b2"])),
        "D": torch.optim.Adam(discriminator.parameters(), lr=hyp["lr"], betas=(hyp["b1"], hyp["b2"])),
        "Q": torch.optim.Adam(itertools.chain(generator.parameters(), discriminator.parameters()), lr=hyp["lr"], betas=(hyp["b1"], hyp["b2"]))
    }


    # Initialize weights

    Model2.weights_init(generator)
    Model2.weights_init(discriminator)

    transform = transforms.Compose([
        transforms.Resize(hyp["img_size"]),
        transforms.CenterCrop(hyp["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset_celeba = ImageFolder(IMAGE_PATH, transform)

    amount_of_used_images = int(len(dataset_celeba)*0.5)

    dataset_celeba, _ = torch.utils.data.random_split(dataset_celeba, [amount_of_used_images,
                                                                       len(dataset_celeba) - amount_of_used_images])
    print("Length of dataset:%d" % len(dataset_celeba))

    dataloader = DataLoader(dataset=dataset_celeba, batch_size=hyp["batch_size"], shuffle=True, num_workers=6, drop_last=True)

    D_loss = []
    G_loss = []
    I_loss = []
    discriminator_accuracy = []
    Z_loss = []
    C_cat_loss = []
    C_con_loss = []


    # TRAINING
    train_obj = training.Training(generator, discriminator, losses, optimizers, dataloader, hyp)

    # train_obj.train(num_epochs=20)
    # train_obj.plot(transform, path=IMG)
    #
    # train_obj.train(num_epochs=20)
    # train_obj.plot(transform, path=IMG)
    #
    # train_obj.train(num_epochs=20)
    # train_obj.plot(transform, path=IMG)
    #
    # train_obj.train(num_epochs=20)
    # train_obj.plot(transform, path=IMG)

    d_loss, g_loss, i_loss, disc_accuracy, z_loss, c_cat_loss, c_con_loss = train_obj.train(num_epochs=hyp['num_epochs'])
    train_obj.plot(transform, path=IMG)

    D_loss.extend(d_loss)
    G_loss.extend(g_loss)
    I_loss.extend(i_loss)
    discriminator_accuracy.extend(disc_accuracy)
    Z_loss.extend(z_loss)
    C_cat_loss.extend(c_cat_loss)
    C_con_loss.extend(c_con_loss)

    plt.figure(1)
    plt.title = "Losses after " + str(hyp['num_epochs']) + " epochs"
    plt.plot(range(hyp['num_epochs']), D_loss, "b-", label="D_loss")
    plt.plot(range(hyp['num_epochs']), G_loss, "y-", label="G_loss")
    plt.plot(range(hyp['num_epochs']), I_loss, "r-", label="I_loss")
    plt.plot(range(hyp['num_epochs']), discriminator_accuracy, "g-", label="D_accuracy")
    plt.plot(range(hyp['num_epochs']), Z_loss, "p-", label="Noise_loss(D)")
    plt.legend()
    #plt.show()
    plt.savefig("Losses_after " + str(hyp['num_epochs']) + " epochs"+".png")

    plt.figure(2)
    plt.title = "Losses after " + str(hyp['num_epochs']) + " epochs"
    plt.plot(range(hyp['num_epochs']), C_cat_loss, "b-", label="C_cat_loss")
    plt.plot(range(hyp['num_epochs']), C_con_loss, "y-", label="C_con_loss")
    plt.plot(range(hyp['num_epochs']), Z_loss, "p-", label="Z_loss")
    plt.legend()
    #plt.show()
    plt.savefig("Q_Losses_after " + str(hyp['num_epochs']) + " epochs" + ".png")
