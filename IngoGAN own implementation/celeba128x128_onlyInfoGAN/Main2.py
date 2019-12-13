import torch.nn as nn
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import itertools
import matplotlib
matplotlib.use('Agg')
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Cuda:%r" % cuda)
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    IMAGE_PATH = "/home/karcsi/celeba" #titanx
    #IMAGE_PATH = "/home/karcsi/Desktop/DAPGAN_PKARCSI/Celeba" #1060
    #IMAGE_PATH = "C:/Users/Karcsi/Documents/celeba"
    #IMAGE_PATH = "/home/karcsi/DAPGAN/Celeba" #960
    IMG = "tony.jpg"

    hyp = {
        "latent_dim": 256,
        "n_classes": 10,
        "code_dim": 2,
        "img_size": 128,
        "channels": 3,
        "batch_size": 128,
        "lr_G": 0.0002,
        "lr_D": 0.0002,
        "lr_N": 0.002,
        "b1": 0.5,
        "b2": 0.999,
        "num_epochs": 50,

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
    noise_predictor = Model2.noise_predictor(
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
        'noise_loss': nn.MSELoss(),
        'finetune_loss': nn.MSELoss()
    }

    # convert everything to CUDA
    if cuda:
        generator.cuda()
        discriminator.cuda()
        noise_predictor.cuda()
        losses['classifier_loss'].cuda()
        losses['categorical_loss'].cuda()
        losses['continuous_loss'].cuda()
        losses['noise_loss'].cuda()

    # Optimizers new see
    optimizers = {
        "G": torch.optim.Adam(generator.parameters(), lr=hyp["lr_G"], betas=(hyp["b1"], hyp["b2"])),
        "D": torch.optim.Adam(discriminator.parameters(), lr=hyp["lr_D"], betas=(hyp["b1"], hyp["b2"])),
        "Q": torch.optim.Adam(itertools.chain(generator.parameters(), discriminator.parameters()), lr=hyp["lr_D"], betas=(hyp["b1"], hyp["b2"])),
        "N": torch.optim.Adam(noise_predictor.parameters(), lr=hyp["lr_N"], betas=(hyp["b1"], hyp["b2"])),
    }


    # Initialize weights

    Model2.weights_init(generator)
    Model2.weights_init(discriminator)
    Model2.weights_init(noise_predictor)

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
    N_loss = []
    C_cat_loss = []
    C_con_loss = []
    Pred_Cat_loss = []
    Pred_Con_loss = []

    # pretrained_gen = torch.load("Celeba_gen100_64")
    # if cuda:
    #     pretrained_gen.cuda()

    # TRAINING
    train_obj = training.Training(generator, discriminator, noise_predictor, losses, optimizers, dataloader, hyp)

    train_obj.train(num_epochs=10)
    #train_obj.plot(transform, path=IMG)
    train_obj.plot(transform, path=None)

    train_obj.train(num_epochs=10)
    #train_obj.plot(transform, path=IMG)
    train_obj.plot(transform, path=None)

    train_obj.train(num_epochs=10)
    #train_obj.plot(transform, path=IMG)
    train_obj.plot(transform, path=None)

    train_obj.train(num_epochs=10)
    #train_obj.plot(transform, path=IMG)
    train_obj.plot(transform, path=None)

    train_obj.train(num_epochs=10)
    #train_obj.plot(transform, path=IMG)
    train_obj.plot(transform, path=None)

    train_obj.train(num_epochs=10)
    #train_obj.plot(transform, path=IMG)
    train_obj.plot(transform, path=None)

    train_obj.train(num_epochs=10)
    #train_obj.plot(transform, path=IMG)
    train_obj.plot(transform, path=None)

    train_obj.train(num_epochs=10)
    #train_obj.plot(transform, path=IMG)
    train_obj.plot(transform, path=None)

    train_obj.train(num_epochs=10)
    #train_obj.plot(transform, path=IMG)
    train_obj.plot(transform, path=None)

    d_loss, g_loss, i_loss, disc_accuracy, c_cat_loss, c_con_loss, n_loss, pred_cat, pred_con, epochs_done = train_obj.train(num_epochs=10)
    #train_obj.plot(transform, path=IMG)
    train_obj.plot(transform, path=None)


    D_loss.extend(d_loss)
    G_loss.extend(g_loss)
    I_loss.extend(i_loss)
    discriminator_accuracy.extend(disc_accuracy)
    C_cat_loss.extend(c_cat_loss)
    C_con_loss.extend(c_con_loss)
    N_loss.extend(n_loss)
    Pred_Cat_loss.extend(pred_cat)
    Pred_Con_loss.extend(pred_con)



    plt.figure(1)
    plt.title = "Losses after " + str(epochs_done) + " epochs"
    plt.plot(range(epochs_done), D_loss, "b-", label="D_loss")
    plt.plot(range(epochs_done), G_loss, "y-", label="G_loss")
    plt.plot(range(epochs_done), I_loss, "r-", label="I_loss")
    plt.plot(range(epochs_done), discriminator_accuracy, "g-", label="D_accuracy")
    plt.legend()
    #plt.show()
    plt.savefig("Losses_after " + str(epochs_done) + " epochs"+".png")

    # plt.figure(2)
    # plt.title = "Losses after " + str(hyp['num_epochs']) + " epochs"
    # plt.plot(range(hyp['num_epochs']), C_cat_loss, "b-", label="C_cat_loss")
    # plt.plot(range(hyp['num_epochs']), C_con_loss, "y-", label="C_con_loss")
    # plt.plot(range(hyp['num_epochs']), N_loss, "p-", label="N_loss")
    # plt.legend()
    # #plt.show()
    # plt.savefig("Q_Losses_after " + str(hyp['num_epochs']) + " epochs" + ".png")

    # plt.figure(3)
    # plt.title = "Losses after " + str(epochs_done2) + " epochs"
    # plt.plot(range(epochs_done2), Pred_Con_loss, "b-", label="P_Con_loss")
    # plt.plot(range(epochs_done2), Pred_Cat_loss, "y-", label="P_Cat_loss")
    # plt.plot(range(epochs_done2), N_loss, "p-", label="N_loss")
    # plt.legend()
    # #plt.show()
    # plt.savefig("Predictor_Losses_after " + str(epochs_done2) + " epochs" + ".png")
