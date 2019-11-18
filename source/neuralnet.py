import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNet(object):

    def __init__(self, height, width, channel, device, ngpu, ksize, z_dim, learning_rate=1e-3):

        self.height, self.width, self.channel = height, width, channel
        self.device, self.ngpu = device, ngpu
        self.ksize, self.z_dim, self.learning_rate = ksize, z_dim, learning_rate

        self.encoder = \
            Encoder(height=self.height, width=self.width, channel=self.channel, \
            ngpu=self.ngpu, ksize=self.ksize, z_dim=self.z_dim).to(self.device)
        self.decoder = \
            Decoder(height=self.height, width=self.width, channel=self.channel, \
            ngpu=self.ngpu, ksize=self.ksize, z_dim=self.z_dim).to(self.device)
        self.discriminator = \
            Discriminator(height=self.height, width=self.width, channel=self.channel, \
            ngpu=self.ngpu, ksize=self.ksize, z_dim=self.z_dim).to(self.device)

        self.models = [self.encoder, self.decoder, self.discriminator]

        for idx_m, model in enumerate(self.models):
            if(self.device.type == 'cuda') and (self.models[idx_m].ngpu > 0):
                self.models[idx_m] = nn.DataParallel(self.models[idx_m], list(range(self.models[idx_m].ngpu)))

        self.num_params = 0
        for idx_m, model in enumerate(self.models):
            for p in model.parameters():
                self.num_params += p.numel()
            print(model)
        print("The number of parameters: %d" %(self.num_params))

        self.params = \
            list(self.encoder.parameters()) + \
            list(self.decoder.parameters()) + \
            list(self.discriminator.parameters())
        self.optimizer = optim.Adam(self.params, lr=self.learning_rate)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Encoder(nn.Module):

    def __init__(self, height, width, channel, ngpu, ksize, z_dim):
        super(Encoder, self).__init__()

        self.height, self.width, self.channel = height, width, channel
        self.ngpu, self.ksize, self.z_dim = ngpu, ksize, z_dim

        self.en_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.channel, out_channels=16, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),
        )

        self.en_dense = nn.Sequential(
            Flatten(),
            nn.Linear((self.height//(2**2))*(self.width//(2**2))*self.channel*64, 512),
            nn.ELU(),
            nn.Linear(512, self.z_dim),
        )

    def forward(self, input):

        convout = self.en_conv(input)
        z_code = self.en_dense(convout)

        return z_code

class Decoder(nn.Module):

    def __init__(self, height, width, channel, ngpu, ksize, z_dim):
        super(Decoder, self).__init__()

        self.height, self.width, self.channel = height, width, channel
        self.ngpu, self.ksize, self.z_dim = ngpu, ksize, z_dim

        self.de_dense = nn.Sequential(
            nn.Linear(self.z_dim, 512),
            nn.ELU(),
            nn.Linear(512, (self.height//(2**2))*(self.width//(2**2))*self.channel*64),
            nn.ELU(),
        )

        self.de_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.ksize+1, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),

            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=self.ksize+1, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=self.channel, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.Sigmoid(),
        )

    def forward(self, input):

        denseout = self.de_dense(input)
        denseout_res = denseout.view(denseout.size(0), 64, (self.height//(2**2)), (self.height//(2**2)))
        x_hat = self.de_conv(denseout_res)

        return x_hat

class Discriminator(nn.Module):

    def __init__(self, height, width, channel, ngpu, ksize, z_dim):
        super(Discriminator, self).__init__()

        self.height, self.width, self.channel = height, width, channel
        self.ngpu, self.ksize, self.z_dim = ngpu, ksize, z_dim

        self.dis_conv = nn.ModuleList([
            nn.Conv2d(in_channels=self.channel, out_channels=16, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),
        ])

        self.dis_dense = nn.ModuleList([
            Flatten(),
            nn.Linear((self.height//(2**2))*(self.width//(2**2))*self.channel*64, 512),
            nn.ELU(),
            nn.Linear(512, 1),
        ])

    def forward(self, input):

        featurebank = []

        for idx_l, layer in enumerate(self.dis_conv):
            input = layer(input)
            if("torch.nn.modules.activation" in str(type(layer))):
                featurebank.append(input)
        convout = input

        for idx_l, layer in enumerate(self.dis_dense):
            input = layer(input)
            if("torch.nn.modules.activation" in str(type(layer))):
                featurebank.append(input)
        disc_score = input

        return disc_score, featurebank
