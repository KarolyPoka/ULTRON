import torch.nn as nn
import torch
import math

class block(nn.Module):
    def __init__(self, n, nout=None):
        super().__init__()
        if not nout:
            nout = n
        self.conv = nn.Conv2d(n, nout, (3,3), padding=1)#, bias=False)
    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.selu(x)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        print("conv layer initialized")
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
        print("batchnorm layer initialized")


class Decoder(nn.Module):
    def __init__(self, latent_dim, code_dim):
        super(Decoder, self).__init__()
        input_dim = latent_dim + code_dim

        #self.init_size = img_size // 2**5  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * 8 * 8))



        self.gen_layers = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.ELU(),
            #nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.ELU(),
            #nn.BatchNorm2d(128),

            nn.UpsamplingNearest2d(scale_factor=2),

            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.ELU(),
            #nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.ELU(),
            #nn.BatchNorm2d(128),

            nn.UpsamplingNearest2d(scale_factor=2),

            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.ELU(),
            #nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.ELU(),
            #nn.BatchNorm2d(128),

            nn.UpsamplingNearest2d(scale_factor=2),

            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.ELU(),
            #nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.ELU(),
            #nn.BatchNorm2d(128),

            nn.Conv2d(128, 3, 3, padding=1, bias=False),
            nn.Tanh()


        )

    def forward(self, x):
        # print(noise.shape, labels.shape)
        #gen_input = torch.cat((noise, code), -1)
        # print("linear réteg előtt:")
        # print(x.shape)
        # print("a réteg maga:")
        # print(self.l1)
        out = self.l1(x)
        # print("linear réteg után:")
        # print(out.shape)
        out = out.view(out.shape[0], 128, 8, 8)
        # print("view művelet után:")
        # print(out.shape)
        # print(self.gen_layers)
        img = self.gen_layers(out)
        #print(img.shape)

        #print("generator_kimenetén:")
        #print(img.shape)
        return img



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.latent_dim = latent_dim
        # self.n_classes = n_classes
        # self.code_dim = code_dim
        # self.img_size = img_size
        # self.channels = channels


        self.disc_layers = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ELU(),
            #nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ELU(),
            #nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ELU(),
            #nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=1, padding=0),
            nn.AvgPool2d(2, 2),


            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ELU(),
            #nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ELU(),
            #nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=1, padding=0),
            #nn.BatchNorm2d(256),
            nn.AvgPool2d(2, 2),


            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ELU(),
            #nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ELU(),
            #nn.BatchNorm2d(256),
            nn.Conv2d(256, 384, kernel_size=1, padding=0),
            #nn.BatchNorm2d(384),
            nn.AvgPool2d(2, 2),


            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ELU(),
            #nn.BatchNorm2d(384),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ELU(),
            #nn.BatchNorm2d(384),



        )
        self.linear = nn.Sequential(nn.Linear(8*8*3*128, 64))

    def forward(self, img):

        out = self.disc_layers(img)


        # print("enc_forwad:")
        # print(out.shape)
        out = out.view(out.shape[0], 8*8*3*128)
        # print("enc_forwad_linear_input:")
        # print(out.shape)
        code = self.linear(out)

        return code

class generator(nn.Module):
    def __init__(self, latent_dim, n_classes, code_dim, img_size, channels):
        super(generator, self).__init__()

        self.decoder = Decoder(latent_dim, code_dim)

    def forward(self, noise, code):
        gen_input = torch.cat((noise, code), -1)
        img = self.decoder(gen_input)
        return img

class discriminator(nn.Module):
    def __init__(self, latent_dim, n_classes, code_dim, img_size, channels):
        super(discriminator, self).__init__()

        self.decoder = Decoder(latent_dim, code_dim)
        self.encoder = Encoder()

        self.continuous_layer = nn.Sequential(nn.Linear(64, code_dim), nn.Tanh())

    def forward(self, img):
        z = self.encoder(img)
        code = self.continuous_layer(z)
        dec_input = torch.cat((z, code), -1)
        rec_img = self.decoder(dec_input)
        return code, rec_img


class noise_predictor(nn.Module):
    def __init__(self, latent_dim, n_classes, code_dim, img_size, channels):
        super(noise_predictor, self).__init__()

        self.n_pred_layers = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            #nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            #nn.Dropout2d(0.25),
            nn.BatchNorm2d(256),


            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            #nn.Dropout2d(0.25),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            #nn.Dropout2d(0.25),
            nn.BatchNorm2d(1024),

            nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            #nn.Dropout2d(0.25),
            nn.BatchNorm2d(2048)

        )
        predictor_size = img_size // 2**5
        self.categorical_layer = nn.Sequential(nn.Linear(2048 * predictor_size ** 2, n_classes), nn.Softmax())
        self.continuous_layer = nn.Sequential(nn.Linear(2048 * predictor_size ** 2, code_dim))
        self.final_layer = nn.Sequential(nn.Linear(2048 * predictor_size ** 2, 4 * latent_dim),
                                         nn.LeakyReLU(0.1, inplace=True),
                                         nn.BatchNorm1d(4 * latent_dim),
                                         nn.Linear(4 * latent_dim, latent_dim))

    def forward(self, img):
        out = self.n_pred_layers(img)
        #print("disc_forwad:")
        #print(out.shape)
        out = out.view(out.shape[0], -1)
        #print("disc_forwad:")
        #print(out.shape)
        cat_c = self.categorical_layer(out)
        cont_c = self.continuous_layer(out)
        noise = self.final_layer(out)

        return noise, cat_c, cont_c
