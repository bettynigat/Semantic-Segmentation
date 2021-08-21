import torch
import torch.nn as nn
import torch.nn.functional as F

class segnet(nn.Module):
    def __init__(self, out_channel=10):
        super(segnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(16, out_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.conv4(out)
        return out


class UNetEnc(nn.Module):

    def __init__(self, in_channels, features, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(features,momentum=0.99, affine=True, track_running_stats=False),
            nn.Conv2d(features, features, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features, out_channels, 2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class UNetDec(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels,momentum=0.99, affine=True, track_running_stats=False), 
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers += [nn.Dropout(.5)]
        layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        return self.down(x)



class UNet(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.dec1 = UNetDec(3, 64)
        self.dec2 = UNetDec(64, 128)
        self.dec3 = UNetDec(128, 256, dropout=True)
        self.center = nn.Sequential(
            nn.Conv2d(256, 512, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.enc3 = UNetEnc(512, 256, 128)
        self.enc2 = UNetEnc(256, 128, 64)
        self.enc1 = nn.Sequential(
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        dec1 = self.dec1(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        center = self.center(dec3)
        enc3 = self.enc3(torch.cat([
            center, F.upsample_bilinear(dec3, center.size()[2:])], 1))
        enc2 = self.enc2(torch.cat([
            enc3, F.upsample_bilinear(dec2, enc3.size()[2:])], 1))
        enc1 = self.enc1(torch.cat([
            enc2, F.upsample_bilinear(dec1, enc2.size()[2:])], 1))

        return F.upsample_bilinear(self.final(enc1), x.size()[2:])


if __name__ == "__main__":
    batch = torch.zeros(64, 3, 256, 256)
    model = segnet()
    output = model(batch)
    print(output.size())