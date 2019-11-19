import torch.nn as nn
import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class generator(nn.Module):
    def __init__(self, latent_dim, n_classes, code_dim, img_size, channels):
        super(generator, self).__init__()
        input_dim = latent_dim + n_classes + code_dim

        self.init_size = img_size // 4 # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))

        self.gen_layers = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels, code):
        # print(noise.shape, labels.shape)
        gen_input = torch.cat((noise, labels, code), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.gen_layers(out)
        return img



class discriminator(nn.Module):
    def __init__(self, latent_dim, n_classes, code_dim, img_size, channels):
        super(discriminator, self).__init__()

        self.disc_layers = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(32, 0.8),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(64, 0.8),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(128, 0.8),
        )
        discriminator_size = img_size // 2 ** 4
        #output layers (talán az osztályozó kaphatna egy Sigmoid aktivációt???)
        self.classifier_layer = nn.Sequential(nn.Linear(128 * discriminator_size ** 2, 1), nn.Sigmoid())
        self.categorical_layer = nn.Sequential(nn.Linear(128 * discriminator_size ** 2, n_classes), nn.Softmax())
        self.continuous_layer = nn.Sequential(nn.Linear(128 * discriminator_size ** 2, code_dim))
        self.noise_layer = nn.Sequential(nn.Linear(128 * discriminator_size ** 2, latent_dim))

    def forward(self, img):
        out = self.disc_layers(img)
        out = out.view(out.shape[0], -1)
        judgement = self.classifier_layer(out)
        cat_code = self.categorical_layer(out)
        cont_code = self.continuous_layer(out)
        noise = self.noise_layer(out)

        return judgement, cat_code, cont_code, noise
