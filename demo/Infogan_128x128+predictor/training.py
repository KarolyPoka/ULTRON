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
import torch.nn.functional as F
import pytorch_msssim

from PIL import Image

#import Model

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


class Training:
    def __init__(self, G, D, N, losses, optimizers, dataloader, hyperparams, epochs_done=0, loaded_losses=None):
        self.optimizer_G = optimizers['G']
        self.optimizer_D = optimizers['D']
        self.optimizer_Q = optimizers['Q']
        self.optimizer_N = optimizers['N']
        self.classifier_loss = losses['classifier_loss']
        self.categorical_loss = losses['categorical_loss']
        self.continuous_loss = losses['continuous_loss']
        self.noise_loss = losses['noise_loss']
        self.finetune_loss = losses['finetune_loss']
        self.dataloader = 2#dataloader
        self.generator = G
        self.discriminator = D
        self.noise_predictor = N

        self.D_loss = []
        self.G_loss = []
        self.I_loss = []
        self.discriminator_accuracy = []
        self.C_cat_loss = []
        self.C_con_loss = []
        self.N_loss = []
        self.pred_C_cat_loss = []
        self.pred_C_con_loss = []
        self.loaded_epochs = epochs_done

        if loaded_losses is not None:
            self.D_loss = loaded_losses['D_loss']
            self.G_loss = loaded_losses['G_loss']
            self.I_loss = loaded_losses['I_loss']
            self.discriminator_accuracy = loaded_losses['discriminator_accuracy']
            self.C_cat_loss = loaded_losses['C_cat_loss']
            self.C_con_loss = loaded_losses['C_con_loss']
            self.N_loss = loaded_losses['N_loss']
            self.pred_C_cat_loss = loaded_losses['pred_C_cat_loss']
            self.pred_C_con_loss = loaded_losses['pred_C_con_loss']

        self.params = hyperparams
        self.bar = progressbar.ProgressBar(0, 2)

        self.epochs_done = epochs_done

    def to_categorical(self, y, num_columns):
        """Returns one-hot encoded Variable"""
        y_cat = np.zeros((y.shape[0], num_columns))
        y_cat[range(y.shape[0]), y] = 1.0
        return FloatTensor(y_cat)

    def train(self, train_gen=True, train_disc=True, train_info=True, num_epochs=20, train_encoder=False):
        self.generator.train()
        self.discriminator.train()
        self.noise_predictor.train()
        self.loaded_epochs +=num_epochs

        ms_ssim_loss = pytorch_msssim.MSSSIM()

        for epoch in range(self.epochs_done, self.loaded_epochs):

            batch_discriminator_loss = 0.0
            batch_generator_loss = 0.0
            batch_info_loss = 0.0
            batch_discriminator_accuracy = 0.0
            batch_C_cat_loss = 0.0
            batch_C_con_loss = 0.0
            batch_N_loss = 0.0
            batch_pred_C_cat_loss = 0.0
            batch_pred_C_con_loss = 0.0
            if (train_encoder == False):
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
                    _, pred_label, pred_code, _ = self.discriminator(gen_imgs)

                    info_loss = self.params["lambda_cat"] * self.categorical_loss(pred_label, gt_labels) + \
                                self.params["lambda_con"] * self.continuous_loss(pred_code, static_code)

                    batch_C_cat_loss += self.categorical_loss(pred_label, gt_labels).item()
                    batch_C_con_loss += self.continuous_loss(pred_code, static_code).item()

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

                self.bar.finish()

                print("[Epoch %d/%d] [D loss: %f] [D accur: %f] [G loss: %f] "
                      "[info loss: %f] [c_cat_loss: %f] [c_con_loss: %f]"
                      % (epoch + 1,
                         self.loaded_epochs,
                         (batch_discriminator_loss / len(self.dataloader)),
                         (batch_discriminator_accuracy / len(self.dataloader)),
                         (batch_generator_loss / len(self.dataloader)),
                         (batch_info_loss / len(self.dataloader)),
                         (batch_C_cat_loss / len(self.dataloader)),
                         (batch_C_con_loss / len(self.dataloader))))
                self.epochs_done += 1
            # ------------------
            # Train the noise_predictor
            # ------------------
            if(train_encoder == True):
                for i, (imgs, _) in enumerate(self.dataloader):
                    #self.discriminator.eval()
                    self.generator.eval()

                    real_imgs = imgs.type(FloatTensor)

                    self.optimizer_N.zero_grad()

                    # Sample labels
                    #sampled_labels = np.random.randint(0, self.params["n_classes"], self.params["batch_size"])

                    # Ground truth labels
                    #gt_labels = LongTensor(sampled_labels)

                    # Sample noise, labels and code as generator input
                    #static_z = FloatTensor(np.random.normal(0, 1, (self.params["batch_size"], self.params["latent_dim"])))
                    # static_class = self.to_categorical(sampled_labels, num_columns=self.params["n_classes"])
                    # static_code = FloatTensor(np.random.uniform(-1, 1, (self.params["batch_size"], self.params["code_dim"])))

                    #_, cat_code, cont_code, _ = self.discriminator(real_imgs)

                    pred_noise, cat_code, cont_code = self.noise_predictor(real_imgs)

                    gen_imgs = self.generator(pred_noise, cat_code, cont_code)

                    n_loss = 1 - ms_ssim_loss(real_imgs, gen_imgs)

                    #n_loss = self.noise_loss(gen_imgs, real_imgs)
                    #cat_c_loss = self.categorical_loss(pred_cat_c, gt_labels)
                    #con_c_loss = self.continuous_loss(pred_con_c, static_code)

                    #noise_pred_loss = n_loss + cat_c_loss + con_c_loss

                    n_loss.backward()
                    self.optimizer_N.step()

                    batch_N_loss += n_loss.item()
                    #batch_pred_C_cat_loss += cat_c_loss.item()
                    #batch_pred_C_con_loss += con_c_loss.item()

                    self.bar.update(i)

                self.N_loss.append(batch_N_loss / len(self.dataloader))
                self.pred_C_cat_loss.append(batch_pred_C_cat_loss / len(self.dataloader))
                self.pred_C_con_loss.append(batch_pred_C_con_loss / len(self.dataloader))

                self.bar.finish()

                print("[Epoch %d/%d] [N loss: %f] [c_cat_loss: %f] [c_con_loss: %f]"
                      % (epoch+1, self.loaded_epochs, (batch_N_loss / len(self.dataloader)),
                         (batch_pred_C_cat_loss / len(self.dataloader)),
                         (batch_pred_C_con_loss / len(self.dataloader))))
                self.epochs_done += 1
            # ------------------
            # Fine tuning the generator
            # ------------------

            # self.optimizer_G.zero_grad()
            #
            # _, cat_code, cont_code = self.discriminator(real_imgs)
            # img_noise = self.noise_predictor(real_imgs)
            # actual_sample = self.generator(img_noise, cat_code, cont_code)
            # f_loss = self.finetune_loss(actual_sample, real_imgs)
            # f_loss.backward()
            # self.optimizer_G.step()

        #torch.save(self.generator, "Celeba_gen" + str(self.epochs_done))
        #torch.save(self.discriminator, "Celeba_disc" + str(self.epochs_done))
        #torch.save(self.noise_predictor, "Celebe_noise" + str(self.epochs_done))

        state = {
            'epochs_done': self.epochs_done,
            'generator': self.generator,
            'discriminator': self.discriminator,
            'noise_predictor': self.noise_predictor,
            'optimizer_G': self.optimizer_G,
            'optimizer_D': self.optimizer_D,
            'optimizer_Q': self.optimizer_Q,
            'optimizer_N': self.optimizer_N,
            'D_loss': self.D_loss,
            'G_loss': self.G_loss,
            'I_loss': self.I_loss,
            'discriminator_accuracy': self.discriminator_accuracy,
            'C_cat_loss': self.C_cat_loss,
            'C_con_loss': self.C_con_loss,
            'N_loss': self.N_loss,
            'pred_C_cat_loss': self.pred_C_cat_loss,
            'pred_C_con_loss': self.pred_C_con_loss,

        }

        torch.save(state, "DAPGAN")

        return self.D_loss,\
               self.G_loss, \
               self.I_loss, \
               self.discriminator_accuracy, \
               self.C_cat_loss, \
               self.C_con_loss, \
               self.N_loss, \
               self.pred_C_cat_loss, \
               self.pred_C_con_loss, \
               self.epochs_done


    def plot(self, transform, path=None):
        # self.D_loss = []
        # self.G_loss = []
        # self.I_loss = []
        # self.discriminator_accuracy = []
        # self.C_cat_loss = []
        # self.C_con_loss = []
        # self.Z_loss = []
        self.generator.eval()
        self.discriminator.eval()
        self.noise_predictor.eval()
        n_classes = self.params["n_classes"]
        n_row = n_classes
        static_label = self.to_categorical(np.array([num for _ in range(n_classes) for num in range(n_classes)]), num_columns=n_classes)
        static_code = FloatTensor(np.zeros((n_classes ** 2, self.params["code_dim"])))

        if path is None:
            # Static generator inputs for sampling
            static_z = FloatTensor(np.zeros((n_classes ** 2, self.params["latent_dim"])))

            z = FloatTensor(np.random.normal(0, 1, (n_row ** 2, self.params["latent_dim"])))
            static_sample = self.generator(z, static_label, static_code)
            save_image(static_sample.data, "static_noise_" + str(self.epochs_done) + ".png", nrow=n_row, normalize=True)

            # Get varied c1 and c2
            zeros = np.zeros((n_row ** 2, 1))
            c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
            c1 = FloatTensor(np.concatenate((c_varied, zeros), -1))
            c2 = FloatTensor(np.concatenate((zeros, c_varied), -1))
            sample1 = self.generator(static_z, static_label, c1)
            sample2 = self.generator(static_z, static_label, c2)
            save_image(sample1.data, "varied_c1_noise_" + str(self.epochs_done) + ".png", nrow=n_row, normalize=True)
            save_image(sample2.data, "varied_c2_noise_" + str(self.epochs_done) + ".png", nrow=n_row, normalize=True)

        else:
            img = Image.open(path)
            img_t = transform(img)
            #print("image_shape:")
            #print(img.shape)
            img = img_t.unsqueeze_(0).cuda()
            #print("image_shape_after:")
            #print(img.shape)

            # Static sample from image noise generated by the Discriminator

            #_, cat_code, cont_code, _ = self.discriminator(img)
            img_noise, cat_code, cont_code = self.noise_predictor(img)
            img_z = torch.cat([img_noise, ] * n_row ** 2)
            cat_code = torch.cat([cat_code, ] * n_row ** 2)
            cont_code = torch.cat([cont_code, ] * n_row ** 2)
            static_sample = self.generator(img_z, cat_code, cont_code)
            save_image(static_sample.data, "blond_static_" + str(self.epochs_done) + ".png", nrow=n_row, normalize=True)

            # Get varied c1 and c2 on with image noise given by the Discriminator
            zeros = np.zeros((n_row ** 2, 1))
            c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
            c1 = FloatTensor(np.concatenate((c_varied, zeros), -1))
            c2 = FloatTensor(np.concatenate((zeros, c_varied), -1))
            sample1 = self.generator(img_z, cat_code, c1+cont_code)
            sample2 = self.generator(img_z, cat_code, c2+cont_code)
            save_image(sample1.data, "blond_varied_c1_" + str(self.epochs_done) + ".png", nrow=n_row, normalize=True)
            save_image(sample2.data, "blond_varied_c2_" + str(self.epochs_done) + ".png", nrow=n_row, normalize=True)
            print("Probe image is used.")

