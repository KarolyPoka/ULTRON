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

from PIL import Image

#import Model

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


class Training:
    def __init__(self, G, D, losses, optimizers, dataloader, hyperparams):
        self.optimizer_G = optimizers['G']
        self.optimizer_D = optimizers['D']
        self.optimizer_Q = optimizers['Q']
        self.classifier_loss = losses['classifier_loss']
        self.categorical_loss = losses['categorical_loss']
        self.continuous_loss = losses['continuous_loss']
        self.noise_loss = losses['noise_loss']
        self.dataloader = dataloader
        self.generator = G
        self.discriminator = D

        self.D_loss = []
        self.G_loss = []
        self.I_loss = []
        self.discriminator_accuracy = []
        self.C_cat_loss = []
        self.C_con_loss = []
        self.Z_loss = []
        self.params = hyperparams
        self.bar = progressbar.ProgressBar(0, len(dataloader), redirect_stdout=False)

        self.epochs_done = 0

    def to_categorical(self, y, num_columns):
        """Returns one-hot encoded Variable"""
        y_cat = np.zeros((y.shape[0], num_columns))
        y_cat[range(y.shape[0]), y] = 1.0
        return FloatTensor(y_cat)

    def train(self, train_gen=True, train_disc=True, train_info=True, num_epochs=20):

        for epoch in range(num_epochs):

            batch_discriminator_loss = 0.0
            batch_generator_loss = 0.0
            batch_info_loss = 0.0
            batch_discriminator_accuracy = 0.0
            batch_C_cat_loss = 0.0
            batch_C_con_loss = 0.0
            batch_Z_loss = 0.0

            for i, (imgs, _) in enumerate(self.dataloader):
                batch_size = imgs.shape[0]

                # Adversarial ground truths "labels"
                valid = FloatTensor(batch_size, 1).fill_(1.0)
                fake = FloatTensor(batch_size, 1).fill_(0.0)

                # Configure input
                real_imgs = imgs.type(FloatTensor)
                #gen_imgs = None
                # making one-hot vector
                # labels = to_categorical(labels.numpy(), num_columns=n_classes)

                # -----------------
                #  Train Generator
                # -----------------
                #if train_gen:

                self.optimizer_G.zero_grad()


                # Sample noise and labels as generator input
                static_z = FloatTensor(np.random.normal(0, 1, (batch_size, self.params['latent_dim'])))
                static_class = self.to_categorical(np.random.randint(0, self.params["n_classes"], batch_size), num_columns=self.params["n_classes"])
                static_code = FloatTensor(np.random.uniform(-1, 1, (batch_size, self.params["code_dim"])))

                # Generate a batch of images
                gen_imgs = self.generator(static_z, static_class, static_code)

                # Loss measures generator's ability to fool the discriminator

                validity, _, _, _ = self.discriminator(gen_imgs)
                g_loss = self.classifier_loss(validity, valid)

                g_loss.backward()
                self.optimizer_G.step()
                batch_generator_loss += g_loss.item()

                # ---------------------
                #  Train Discriminator
                # ---------------------
                #if train_disc:

                self.optimizer_D.zero_grad()

                # Loss for real images
                noise_for_convergence1 = FloatTensor(np.random.normal(0, 1, imgs.shape))
                disc_real_input = real_imgs + noise_for_convergence1/15

                real_pred, _, _, _ = self.discriminator(disc_real_input)
                d_real_loss = self.classifier_loss(real_pred, valid)

                # if gen_imgs is None:
                #     # Sample noise and labels as generator input
                #     static_z = FloatTensor(np.random.normal(0, 1, (batch_size, self.params['latent_dim'])))
                #     static_class = self.to_categorical(np.random.randint(0, self.params["n_classes"], batch_size),
                #                                        num_columns=self.params["n_classes"])
                #     static_code = FloatTensor(np.random.uniform(-1, 1, (batch_size, self.params["code_dim"])))
                #
                #     # Generate a batch of images
                #     print("fuck")
                #     gen_imgs = self.generator(static_z, static_class, static_code)

                # Loss for fake images
                noise_for_convergence2 = FloatTensor(np.random.normal(0, 1, imgs.shape))
                disc_fake_input = gen_imgs.detach() + noise_for_convergence2/15

                fake_pred, _, _, _ = self.discriminator(disc_fake_input)
                d_fake_loss = self.classifier_loss(fake_pred, fake)

                # Accuracy of discriminator Acc=(TN+TP)/(TN+TP+FN+FP)
                num_good_preds = len([item for item in real_pred if item >= 0.5])  # TP
                num_good_preds += len([item for item in fake_pred if item < 0.5])  # TN
                d_accuracy = num_good_preds / (len(real_pred) + len(fake_pred))  # Acc=(TN+TP)/(TN+TP+FN+FP)

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss.backward()
                self.optimizer_D.step()
                batch_discriminator_loss += d_loss.item()
                batch_discriminator_accuracy += d_accuracy

                # ------------------
                # Information Loss
                # ------------------
                #if train_info:
                self.optimizer_Q.zero_grad()

                # Sample labels
                sampled_labels = np.random.randint(0, self.params["n_classes"], batch_size)

                # Ground truth labels
                gt_labels = LongTensor(sampled_labels)

                # Sample noise, labels and code as generator input
                static_z = FloatTensor(np.random.normal(0, 1, (batch_size, self.params["latent_dim"])))
                static_class = self.to_categorical(sampled_labels, num_columns=self.params["n_classes"])
                static_code = FloatTensor(np.random.uniform(-1, 1, (batch_size, self.params["code_dim"])))

                gen_imgs = self.generator(static_z, static_class, static_code)
                _, pred_label, pred_code, pred_noise = self.discriminator(gen_imgs)

                info_loss = self.params["lambda_cat"] * self.categorical_loss(pred_label, gt_labels) + \
                            self.params["lambda_con"] * self.continuous_loss(pred_code, static_code) + \
                            self.params["lambda_con"] * self.noise_loss(pred_noise, static_z)

                batch_C_cat_loss += self.categorical_loss(pred_label, gt_labels).item()
                batch_C_con_loss += self.continuous_loss(pred_code, static_code).item()
                batch_Z_loss += self.noise_loss(pred_noise, static_z).item()

                info_loss.backward()
                self.optimizer_Q.step()
                batch_info_loss += info_loss.item()

                self.bar.update(i)


            # --------------
            # Log Progress & fütyikeh
            # --------------
            self.D_loss.append(batch_discriminator_loss / len(self.dataloader))
            self.G_loss.append(batch_generator_loss / len(self.dataloader))
            self.I_loss.append(batch_info_loss / len(self.dataloader))
            self.discriminator_accuracy.append(batch_discriminator_accuracy / len(self.dataloader))
            self.C_cat_loss.append(batch_C_cat_loss / len(self.dataloader))
            self.C_con_loss.append(batch_C_con_loss / len(self.dataloader))
            self.Z_loss.append(batch_Z_loss / len(self.dataloader))

            self.bar.finish()
            self.epochs_done += 1
            print("[Epoch %d/%d] [D loss: %f] [D accur: %f] [G loss: %f] "
                  "[info loss: %f] [c_cat_loss: %f] [c_con_loss: %f] [z_loss: %f]"
                  % (epoch + 1,
                     num_epochs,
                     (batch_discriminator_loss / len(self.dataloader)),
                     (batch_discriminator_accuracy / len(self.dataloader)),
                     (batch_generator_loss / len(self.dataloader)),
                     (batch_info_loss / len(self.dataloader)),
                     (batch_C_cat_loss / len(self.dataloader)),
                     (batch_C_con_loss / len(self.dataloader)),
                     (batch_Z_loss / len(self.dataloader))))

        torch.save(self.generator, "Celeba_gen" + str(self.epochs_done))
        torch.save(self.discriminator, "Celeba_disc" + str(self.epochs_done))

        #Plot will happen in the main function

        # plt.figure()
        # plt.title = "Losses after " + str(self.epochs_done) + " epochs"
        # plt.plot(range(num_epochs), self.D_loss, "b-", label="D_loss")
        # plt.plot(range(num_epochs), self.G_loss, "y-", label="G_loss")
        # plt.plot(range(num_epochs), self.I_loss, "r-", label="I_loss")
        # plt.plot(range(num_epochs), self.discriminator_accuracy, "g-", label="D_accuracy")
        # plt.plot(range(num_epochs), self.Z_loss, "p-", label="Noise_loss(D)")
        # plt.legend()
        # plt.show()
        return self.D_loss, self.G_loss, self.I_loss, self.discriminator_accuracy, self.Z_loss, self.C_cat_loss, self.C_con_loss

    def plot(self, transform, path=None, n_row=10):
        # self.D_loss = []
        # self.G_loss = []
        # self.I_loss = []
        # self.discriminator_accuracy = []
        # self.C_cat_loss = []
        # self.C_con_loss = []
        # self.Z_loss = []

        n_classes = self.params["n_classes"]
        static_label = self.to_categorical(np.array([num for _ in range(n_classes) for num in range(n_classes)]), num_columns=n_classes)
        static_code = FloatTensor(np.zeros((n_classes ** 2, self.params["code_dim"])))

        if path is None:
            # Static generator inputs for sampling
            static_z = FloatTensor(np.zeros((n_classes ** 2, self.params["latent_dim"])))

            z = FloatTensor(np.random.normal(0, 1, (n_row ** 2, self.params["latent_dim"])))
            static_sample = self.generator(z, static_label, static_code)
            save_image(static_sample.data, "static_" + str(self.epochs_done) + ".png", nrow=n_row, normalize=True)

            # Get varied c1 and c2
            zeros = np.zeros((n_row ** 2, 1))
            c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
            c1 = FloatTensor(np.concatenate((c_varied, zeros), -1))
            c2 = FloatTensor(np.concatenate((zeros, c_varied), -1))
            sample1 = self.generator(static_z, static_label, c1)
            sample2 = self.generator(static_z, static_label, c2)
            save_image(sample1.data, "varied_c1_" + str(self.epochs_done) + ".png", nrow=n_row, normalize=True)
            save_image(sample2.data, "varied_c2_" + str(self.epochs_done) + ".png", nrow=n_row, normalize=True)

        else:
            img = Image.open(path)
            img = transform(img)
            img = img.unsqueeze_(0).cuda()


            # Static sample from image noise generated by the Discriminator
            _, _, _, img_noise = self.discriminator(img)
            img_z = torch.cat([img_noise, ] * n_row ** 2)
            static_sample = self.generator(img_z, static_label, static_code)
            save_image(static_sample.data, "img_static_" + str(self.epochs_done) + ".png", nrow=n_row, normalize=True)

            # Get varied c1 and c2 on with image noise given by the Discriminator
            zeros = np.zeros((n_row ** 2, 1))
            c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
            c1 = FloatTensor(np.concatenate((c_varied, zeros), -1))
            c2 = FloatTensor(np.concatenate((zeros, c_varied), -1))
            sample1 = self.generator(img_z, static_label, c1)
            sample2 = self.generator(img_z, static_label, c2)
            save_image(sample1.data, "img_varied_c1_" + str(self.epochs_done) + ".png", nrow=n_row, normalize=True)
            save_image(sample2.data, "img_varied_c2_" + str(self.epochs_done) + ".png", nrow=n_row, normalize=True)
            print("Probe image is used.")

