import torch
import torch.nn as nn
from layers import LinearNorm, ConvNorm, UNetDown, UNetUp

class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.embedding_note = nn.Embedding(config.num_note, config.note_embed_size)
        self.embedding_text = nn.Embedding(config.num_char, config.text_embed_size)

    def forward(self, x):
        embedded_note = self.embedding_note(x[:,:,0].long())
        embedded_text = self.embedding_text(x[:,:,1].long())

        embedded = torch.cat((embedded_note, embedded_text), dim=2)

        return embedded

class Prenet(nn.Module):
    def __init__(self, config):
        super(Prenet, self).__init__()

        input_size = config.note_embed_size + config.text_embed_size
        output_size = config.fft_size//2 + 1

        conv_layers_prev = []
        for i in range(2):
            conv_layers_prev += [   ConvNorm(output_size, output_size, kernel_size=5, stride=1, padding=2, w_init_gain='relu'),
                                    nn.BatchNorm1d(output_size),
                                    nn.ReLU(),
                                    nn.Dropout(0.5)]

        linear_layers = [   LinearNorm(input_size, output_size),
                            nn.ReLU(),
                            nn.Dropout(0.5)] 

        conv_layers = []
        for i in range(2):
            conv_layers += [ConvNorm(output_size, output_size, kernel_size=5, stride=1, padding=2, w_init_gain='relu'),
                            nn.BatchNorm1d(output_size),
                            nn.ReLU(),
                            nn.Dropout(0.5)]

        self.convolutions_prev = nn.Sequential(*conv_layers_prev)
        self.linear = nn.Sequential(*linear_layers) 
        self.convolutions = nn.Sequential(*conv_layers)
        self.prev_length = config.prev_length

    def forward(self, x, y_prev):
        x = self.linear(x)
        x = x.transpose(1, 2)
        x = self.convolutions(x)
        x = x.transpose(1, 2)

        if self.prev_length > 0:
            y_prev = y_prev.transpose(1, 2)
            y_prev = self.convolutions_prev(y_prev)
            y_prev = y_prev.transpose(1, 2)

            x = torch.cat((y_prev, x), dim=1)

        return x.unsqueeze(1)

class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()
        self.down1 = UNetDown(1, config.size_factor, stride=(4, 4)) # 33 x 129
        self.down2 = UNetDown(config.size_factor, 2*config.size_factor, stride=(1, 2)) # 33 x 65
        self.down3 = UNetDown(2*config.size_factor, 4*config.size_factor, stride=(1, 2)) # 33 x 33
        self.down4 = UNetDown(4*config.size_factor, 8*config.size_factor) # 17 x 17
        self.down5 = UNetDown(8*config.size_factor, 8*config.size_factor) # 9 x 9
        self.down6 = UNetDown(8*config.size_factor, 8*config.size_factor, dropout=0.5) # 5 x 5
        self.down7 = UNetDown(8*config.size_factor, 8*config.size_factor, dropout=0.5) # 3 x 3

        self.up1 = UNetUp(8*config.size_factor, 8*config.size_factor, dropout=0.5) # 5 x 5
        self.up2 = UNetUp(16*config.size_factor, 8*config.size_factor, dropout=0.5) # 9 x 9
        self.up3 = UNetUp(16*config.size_factor, 8*config.size_factor, dropout=0.5) # 17 x 17
        self.up4 = UNetUp(16*config.size_factor, 4*config.size_factor) # 33 x 33 
        self.up5 = UNetUp(8*config.size_factor, 2*config.size_factor, stride=(1, 2)) # 33 x 65
        self.up6 = UNetUp(4*config.size_factor, 1*config.size_factor, stride=(1, 2)) # 33 x 129

        self.final = nn.Sequential(
            nn.ConvTranspose2d(2*config.size_factor, 1, 5, stride=(3, 4), padding=2, bias=False), # 97 x 513
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)

        return self.final(u6)

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.embedding = Embedding(config)
        self.prenet = Prenet(config)
        self.unet = UNet(config)

    def forward(self, x, y_prev):
        x_embedded = self.embedding(x)
        prenet_output = self.prenet(x_embedded, y_prev)
        output = self.unet(prenet_output)

        return output

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        channels = 1
        if config.feature is 'magphs': channels = 2

        def down_block(input_size, output_size):
            layers = [  nn.Conv2d(input_size, input_size, kernel_size=3, stride=1, padding=1),
                        nn.ELU(True),
                        nn.Conv2d(input_size, input_size, kernel_size=3, stride=1, padding=1),
                        nn.ELU(True),
                        nn.Conv2d(input_size, output_size, kernel_size=3, stride=1, padding=1),
                        nn.AvgPool2d(kernel_size=2, stride=2)]

            return layers

        def up_block(input_size, output_size):
            layers = [  nn.Conv2d(input_size, output_size, kernel_size=3, stride=1, padding=1),
                        nn.ELU(True),
                        nn.Conv2d(output_size, output_size, kernel_size=3, stride=1, padding=1),
                        nn.ELU(True),
                        nn.UpsamplingNearest2d(scale_factor=2)]

            return layers

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, config.size_factor, kernel_size=3, stride=1, padding=1),
            nn.ELU(True),
            *down_block(config.size_factor, config.size_factor),
            *down_block(config.size_factor, 2*config.size_factor),
            *down_block(2*config.size_factor, 3*config.size_factor),
            nn.Conv2d(3*config.size_factor, config.size_factor, kernel_size=1, stride=1, padding=0))

        self.decoder = nn.Sequential(
            nn.Conv2d(config.size_factor, config.size_factor, kernel_size=1, stride=1, padding=0),
            *up_block(config.size_factor, config.size_factor),
            *up_block(config.size_factor, config.size_factor),
            *up_block(config.size_factor, config.size_factor),
            nn.Conv2d(config.size_factor, config.size_factor, kernel_size=4, stride=1, padding=2),
            nn.ELU(True),
            nn.Conv2d(config.size_factor, config.size_factor, kernel_size=3, stride=1, padding=1),
            nn.ELU(True),
            nn.Conv2d(config.size_factor, channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        encoder_output = self.encoder(x)
        return self.decoder(encoder_output)
