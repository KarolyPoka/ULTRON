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
import os.path
from os import path
import Model2
import pytorch_msssim

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

    # Different paths for different trainings

    #IMAGE_PATH = "/home/karcsi/celeba" #titanx
    #IMAGE_PATH = "/home/labor/Desktop/DAPGAN_PKARCSI/Celeba" #1060
    IMAGE_PATH = "C:/Users/Karcsi/Documents/celeba"
    #IMAGE_PATH = "/home/karcsi/DAPGAN/Celeba" #960
    IMG = "tony.jpg"

    hyp = {
        "latent_dim": 64,
        "n_classes": 10,
        "code_dim": 2,
        "img_size": 64,
        "channels": 3,
        "batch_size": 16,
        "lr_G": 0.0001,
        "lr_D": 0.0001,
        "lr_N": 0.002,
        "b1": 0.5,
        "b2": 0.999,
        "num_epochs": 50,

        # info loss weights
        "lambda_cat": 1,
        "lambda_con": 0.1
    }

    # Instantiation of the models

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
    loss_funcs = {
        'classifier_loss': pytorch_msssim.MSSSIM(),
        'categorical_loss': nn.CrossEntropyLoss(),
        'continuous_loss': nn.SmoothL1Loss(),
        'noise_loss': nn.MSELoss(),
        'finetune_loss': nn.MSELoss()
    }

    epochs_done = 0
    loaded_losses = None
    state = None

    # If there is a saved model in the root folder we can load it

    if path.exists("DAPGAN_29.tar"):
        state = torch.load('DAPGAN_29.tar')

        discriminator.load_state_dict(state['discriminator'])
        generator.load_state_dict(state['generator'])
        #noise_predictor.load_state_dict(state['noise_predictor'])

        loaded_losses = {
            'D_loss': state['D_loss'],
            'G_loss': state['G_loss'],
            'I_loss': state['I_loss'],
            'discriminator_accuracy': state['discriminator_accuracy'],
            'C_cat_loss': state['C_cat_loss'],
            'C_con_loss': state['C_con_loss'],
            #'N_loss': state['N_loss'],
            'pred_C_cat_loss': state['pred_C_cat_loss'],
            'pred_C_con_loss': state['pred_C_con_loss']
        }
        epochs_done = state['epochs_done']
        print("Saved model and losses successfully loaded from epoch %d, Sir..." % epochs_done)


    # Convert everything to CUDA
    if cuda:
        generator.cuda()
        discriminator.cuda()
        noise_predictor.cuda()
        loss_funcs['classifier_loss'].cuda()
        loss_funcs['categorical_loss'].cuda()
        loss_funcs['continuous_loss'].cuda()
        loss_funcs['noise_loss'].cuda()
        print("Nets and losses converted to Cuda...")

    G = torch.optim.Adam(generator.parameters(), lr=hyp["lr_G"], betas=(hyp["b1"], hyp["b2"]))
    D = torch.optim.Adam(discriminator.parameters(), lr=hyp["lr_D"], betas=(hyp["b1"], hyp["b2"]))
    # Q = torch.optim.Adam(itertools.chain(generator.parameters(), discriminator.parameters()), lr=hyp["lr_D"], betas=(hyp["b1"], hyp["b2"]))
    N = torch.optim.Adam(noise_predictor.parameters(), lr=hyp["lr_N"], betas=(hyp["b1"], hyp["b2"]))

    if state is not None:

        G.load_state_dict(state['optimizer_G'])
        D.load_state_dict(state['optimizer_D'])
        #N.load_state_dict(state['optimizer_N'])
        # Q.load_state_dict(state['optimizer_Q'])
        print("Saved optimizers successfully loaded...")

    else:
        # Initialize weights
        generator.apply(Model2.weights_init)
        discriminator.apply(Model2.weights_init)
        noise_predictor.apply(Model2.weights_init)
        print("Weight init is done!")

    # Collect Optimizers
    optimizers = {
        "G": G,
        "D": D,
        # "Q": Q,
        "N": N
    }

    transform = transforms.Compose([
        transforms.Resize(hyp["img_size"]),
        transforms.CenterCrop(hyp["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Dataloader and make it splittabel. For time saving we trained our models only on the half of the CelebA

    dataset_celeba = ImageFolder(IMAGE_PATH, transform)

    amount_of_used_images = int(len(dataset_celeba)*0.0005)

    dataset_celeba, _ = torch.utils.data.random_split(dataset_celeba, [amount_of_used_images,
                                                                       len(dataset_celeba) - amount_of_used_images])
    print("Length of dataset:%d" % len(dataset_celeba))

    dataloader = DataLoader(dataset=dataset_celeba, batch_size=hyp["batch_size"], shuffle=True, num_workers=0, drop_last=True)


    D_loss = []
    G_loss = []
    I_loss = []
    discriminator_accuracy = []
    C_cat_loss = []
    C_con_loss = []
    N_loss = []
    Pred_Cat_loss = []
    Pred_Con_loss = []

    # We can train our model sequentially to check the results.

    # TRAINING
    train_obj = training.Training(generator, discriminator, noise_predictor, loss_funcs, optimizers, dataloader, hyp, epochs_done, loaded_losses)

    train_obj.train(num_epochs=1, train_encoder=False)
    # #train_obj.plot(transform, path=IMG)
    train_obj.plot(transform, path=None)
    #
    train_obj.train(num_epochs=3, train_encoder=False)
    # #train_obj.plot(transform, path=IMG)
    train_obj.plot(transform, path=None)
    #
    train_obj.train(num_epochs=5, train_encoder=False)
    # #train_obj.plot(transform, path=IMG)
    train_obj.plot(transform, path=None)
    #
    train_obj.train(num_epochs=5, train_encoder=False)
    # #train_obj.plot(transform, path=IMG)
    train_obj.plot(transform, path=None)
    #
    train_obj.train(num_epochs=5, train_encoder=False)
    # #train_obj.plot(transform, path=IMG)
    train_obj.plot(transform, path=None)
    #
    train_obj.train(num_epochs=5, train_encoder=False)
    # #train_obj.plot(transform, path=IMG)
    train_obj.plot(transform, path=None)
    #
    train_obj.train(num_epochs=5, train_encoder=False)
    # #train_obj.plot(transform, path=IMG)
    train_obj.plot(transform, path=None)
    #
    train_obj.train(num_epochs=5, train_encoder=False)
    # #train_obj.plot(transform, path=IMG)
    train_obj.plot(transform, path=None)
    #
    train_obj.train(num_epochs=5, train_encoder=False)
    # #train_obj.plot(transform, path=IMG)
    train_obj.plot(transform, path=None)

    d_loss, g_loss, i_loss, disc_accuracy, c_cat_loss, c_con_loss, n_loss, pred_cat, pred_con, epochs_done = train_obj.train(num_epochs=5, train_encoder=False)
    #train_obj.plot(transform, path=IMG)
    train_obj.plot(transform, path=None)

    # Storing accumulated losses (even extractiong it from a saved model if it is exist)

    D_loss.extend(d_loss)
    G_loss.extend(g_loss)
    I_loss.extend(i_loss)
    discriminator_accuracy.extend(disc_accuracy)
    C_cat_loss.extend(c_cat_loss)
    C_con_loss.extend(c_con_loss)
    N_loss.extend(n_loss)
    Pred_Cat_loss.extend(pred_cat)
    Pred_Con_loss.extend(pred_con)

    # ------------------------
    # Plot of Training losses
    # ------------------------
    plt.figure(1)
    plt.title = "Training_Losses_after_ " + str(epochs_done) + " epochs"
    plt.plot(range(epochs_done), D_loss, "b-", label="D_loss")
    plt.plot(range(epochs_done), G_loss, "y-", label="G_loss")
    plt.plot(range(epochs_done), I_loss, "r-", label="I_loss")
    plt.plot(range(epochs_done), discriminator_accuracy, "g-", label="D_accuracy")
    plt.plot(range(epochs_done), C_cat_loss, "c-", label="C_cat_loss")
    plt.plot(range(epochs_done), C_con_loss, "m-", label="C_con_loss")
    plt.legend()
    # plt.show()
    plt.savefig("Training_Losses_after_ " + str(epochs_done) + " epochs"+".png")

    # # ------------------------
    # # Plot of Predictor losses
    # # ------------------------
    # plt.figure(2)
    # plt.title = "Predictor_after_" + str(epochs_done) + " epochs"
    # plt.plot(range(epochs_done), N_loss, "r-", label="N_loss")
    # plt.legend()
    # # plt.show()
    # plt.savefig("Predictor_after_ " + str(epochs_done) + " epochs" + ".png")


    # plt.figure(3)
    # plt.title = "Losses after " + str(epochs_done2) + " epochs"
    # plt.plot(range(epochs_done2), Pred_Con_loss, "b-", label="P_Con_loss")
    # plt.plot(range(epochs_done2), Pred_Cat_loss, "y-", label="P_Cat_loss")
    # plt.plot(range(epochs_done2), N_loss, "p-", label="N_loss")
    # plt.legend()
    # #plt.show()
    # plt.savefig("Predictor_Losses_after " + str(epochs_done2) + " epochs" + ".png")
