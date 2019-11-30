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

        self.init_size = img_size // 32  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(input_dim, 228 * self.init_size ** 2))



        self.gen_layers = nn.Sequential(
            nn.ConvTranspose2d(228, 448, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(448),
            nn.ReLU(),
            nn.ConvTranspose2d(448, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(64, channels, 4, 2, padding=1, bias=False),
            nn.Tanh(),
        )


    def forward(self, noise, labels, code):
        # print(noise.shape, labels.shape)
        gen_input = torch.cat((noise, labels, code), -1)
        #print("generator_bemenete:")
        #print(gen_input.shape)
        out = self.l1(gen_input)
        #print("linear réteg után:")
        #print(out.shape)
        out = out.view(out.shape[0], 228, self.init_size, self.init_size)
        #print("view művelet után:")
        #print(out.shape)
        img = self.gen_layers(out)

        #print("generator_kimenetén:")
        #print(img.shape)
        return img



class discriminator(nn.Module):
    def __init__(self, latent_dim, n_classes, code_dim, img_size, channels):
        super(discriminator, self).__init__()

        self.disc_layers = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(128),


            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(256),

        )
        discriminator_size = img_size // 2**3

        self.classifier_layer = nn.Sequential(nn.Linear(256 * discriminator_size ** 2, 1), nn.Sigmoid())
        self.categorical_layer = nn.Sequential(nn.Linear(256 * discriminator_size ** 2, n_classes), nn.Softmax())
        self.continuous_layer = nn.Sequential(nn.Linear(256 * discriminator_size ** 2, code_dim))
        self.noise_layer = nn.Sequential(nn.Linear(256 * discriminator_size ** 2, latent_dim))

    def forward(self, img):
        out = self.disc_layers(img)
        #print("disc_forwad:")
        #print(out.shape)
        out = out.view(out.shape[0], -1)
        #print("disc_forwad:")
        #print(out.shape)
        judgement = self.classifier_layer(out)
        cat_code = self.categorical_layer(out)
        cont_code = self.continuous_layer(out)
        noise = self.noise_layer(out)

        return judgement, cat_code, cont_code, noise
