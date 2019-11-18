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

import Model

if __name__ == "__main__":

    cuda = True if torch.cuda.is_available() else False
    print("Cuda:%r"%cuda)
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    latent_dim = 62
    n_classes = 10
    code_dim = 2
    img_size = 32
    channels = 1
    batch_size = 128
    lr = 0.0001
    b1 = 0.5
    b2 = 0.999
    num_epochs =100

    # Nem tudjuk hol kéne használni
    def to_categorical(y, num_columns):
        """Returns one-hot encoded Variable"""
        y_cat = np.zeros((y.shape[0], num_columns))
        y_cat[range(y.shape[0]), y] = 1.0

        return FloatTensor(y_cat)

    #Ezt sum tudjuk hol kéne használni
    def generate_latent_points(latent_dim, n_classes):
      static_z = FloatTensor(np.zeros((n_classes ** 2, latent_dim)))
      static_class = to_categorical(np.array([num for _ in range(n_classes) for num in range(n_classes)]), num_columns=n_classes)
      static_code = FloatTensor(np.zeros((n_classes ** 2, code_dim)))

      return torch.cat([static_z, static_class, static_code])



    generator = Model.generator(latent_dim, n_classes, code_dim, img_size, channels)
    discriminator = Model.discriminator(latent_dim, n_classes, code_dim, img_size, channels)

    #Loss functions
    classifier_loss = nn.BCELoss()
    categorical_loss = nn.CrossEntropyLoss()
    continuous_loss = nn.MSELoss()

    #convert everything to CUDA
    if cuda:
        generator.cuda()
        discriminator.cuda()
        classifier_loss.cuda()
        categorical_loss.cuda()
        continuous_loss.cuda()


    #Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_info = torch.optim.Adam(itertools.chain(generator.parameters(), discriminator.parameters()), lr=lr, betas=(b1, b2))

    #Loss weight
    lambda_cat = 1
    lambda_con = 0.1


    #Initialize weights

    Model.weights_init(generator)
    Model.weights_init(discriminator)


    dataset = dset.MNIST('dataset', transform=transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(),
                                                                  transforms.Normalize([0.5], [0.5])]), download=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6)



    discriminator_loss = []
    generator_loss = []
    InfoGAN_loss = []
    discriminator_accuracy = []

    bar = progressbar.ProgressBar(0, len(dataloader), redirect_stdout=True)

    #TRAINING
    for epoch in range(num_epochs):

        batch_discriminator_loss = 0.0
        batch_generator_loss = 0.0
        batch_InfoGAN_loss = 0.0
        batch_discriminator_accuracy = 0.0

        for i, (imgs, _) in enumerate(dataloader):

            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = FloatTensor(batch_size, 1).fill_(1.0)
            fake = FloatTensor(batch_size, 1).fill_(0.0)

            # Configure input
            real_imgs = imgs.type(FloatTensor)
            #making one-hot vector
            #labels = to_categorical(labels.numpy(), num_columns=n_classes)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            static_z = FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim)))
            static_class = to_categorical(np.random.randint(0, n_classes, batch_size), num_columns=n_classes)
            static_code = FloatTensor(np.random.uniform(-1, 1, (batch_size, code_dim)))

            # Generate a batch of images
            gen_imgs = generator(static_z, static_class, static_code)

            # Loss measures generator's ability to fool the discriminator
            validity, _, _ = discriminator(gen_imgs)
            g_loss = classifier_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            real_pred, _, _ = discriminator(real_imgs)
            d_real_loss = classifier_loss(real_pred, valid)

            # Loss for fake images
            fake_pred, _, _ = discriminator(gen_imgs.detach())
            d_fake_loss = classifier_loss(fake_pred, fake)

            # Accuracy of discriminator
            #print(real_pred)
            num_good_preds = len([item for item in real_pred if item >= 0.5])
            d_accuracy = num_good_preds/len(real_pred)
            #print(num_good_preds)
            #print(len(real_pred))

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # ------------------
            # Information Loss
            # ------------------

            optimizer_info.zero_grad()

            # Sample labels
            sampled_labels = np.random.randint(0, n_classes, batch_size)

            # Ground truth labels
            gt_labels = LongTensor(sampled_labels)

            # Sample noise, labels and code as generator input
            static_z = FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim)))
            static_class = to_categorical(sampled_labels, num_columns=n_classes)
            static_code = FloatTensor(np.random.uniform(-1, 1, (batch_size, code_dim)))

            gen_imgs = generator(static_z, static_class, static_code)
            _, pred_label, pred_code = discriminator(gen_imgs)

            info_loss = lambda_cat * categorical_loss(pred_label, gt_labels) + lambda_con * continuous_loss(
                pred_code, static_code
            )

            info_loss.backward()
            optimizer_info.step()

            # --------------
            # Log Progress and calculate losses and accuracy
            # --------------

            batch_discriminator_loss += d_loss.item()
            batch_generator_loss += g_loss.item()
            batch_InfoGAN_loss += info_loss.item()
            batch_discriminator_accuracy += d_accuracy

            bar.update(i)

        discriminator_loss.append(batch_discriminator_loss / len(dataloader))
        generator_loss.append(batch_generator_loss / len(dataloader))
        InfoGAN_loss.append(batch_InfoGAN_loss / len(dataloader))
        discriminator_accuracy.append(batch_discriminator_accuracy / len(dataloader))

        bar.finish()
        print("[Epoch %d/%d] [D loss: %f] [D accur: %f] [G loss: %f] [info loss: %f]" % (epoch+1,
                                                                                         num_epochs,
                                                                                         (batch_discriminator_loss / len(dataloader)),
                                                                                         (batch_discriminator_accuracy / len(dataloader)),
                                                                                         (batch_generator_loss / len(dataloader)),
                                                                                         (batch_InfoGAN_loss / len(dataloader))))



    plt.figure()
    plt.plot(range(num_epochs), discriminator_loss, "b-", label="D_loss")
    plt.plot(range(num_epochs), generator_loss, "y-", label="G_loss")
    plt.plot(range(num_epochs), InfoGAN_loss, "r-", label="InfoGAN_loss")
    plt.plot(range(num_epochs), discriminator_accuracy, "g-", label="D_accuracy")
    plt.legend()
    plt.show()



    torch.save(generator, "Mnist_gen")
    torch.save(discriminator, "Mnist_disc")