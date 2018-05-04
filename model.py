import torch.nn as nn
import torch.nn.functional as F


def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(
        nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False)
    )
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(
        nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False)
    )
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


class G12(nn.Module):

    """Generator for transfering from MNIST to SVHN."""

    def __init__(self, conv_dim=64):
        """Initialize a new instance.

        Parameters
        ----------
        conv_dim : int, optional
            Base for the number of convolutional channels. The sequence of
            channels is ::

                1 -> conv_dim -> 2 * conv_dim -> ... -> 2 *conv_dim ->
                conv_dim -> 3
        """
        super(G12, self).__init__()
        # Encoding blocks
        self.conv1 = conv(c_in=1, c_out=conv_dim,
                          k_size=4, stride=2, pad=1, bn=True)
        self.conv2 = conv(c_in=conv_dim, c_out=conv_dim * 2,
                          k_size=4, stride=2, pad=1, bn=True)

        # Residual blocks
        self.conv3 = conv(c_in=conv_dim * 2, c_out=conv_dim * 2,
                          k_size=3, stride=1, pad=1, bn=True)
        self.conv4 = conv(c_in=conv_dim * 2, c_out=conv_dim * 2,
                          k_size=3, stride=1, pad=1, bn=True)

        # Decoding blocks
        self.deconv1 = deconv(c_in=conv_dim * 2, c_out=conv_dim,
                              k_size=4, stride=2, pad=1, bn=True)
        self.deconv2 = deconv(c_in=conv_dim, c_out=3,
                              k_size=4, stride=2, pad=1, bn=False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)      # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)    # (?, 128, 8, 8)

        out = F.leaky_relu(self.conv3(out), 0.05)    # ( " )
        out = F.leaky_relu(self.conv4(out), 0.05)    # ( " )

        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 64, 16, 16)
        out = F.tanh(self.deconv2(out))              # (?, 3, 32, 32)
        return out


class G21(nn.Module):
    """Generator for transfering from SVHN to MNIST."""
    def __init__(self, conv_dim=64):
        """Initialize a new instance.

        Parameters
        ----------
        conv_dim : int, optional
            Base for the number of convolutional channels. The sequence of
            channels is ::

                3 -> conv_dim -> 2 * conv_dim -> ... -> 2 *conv_dim ->
                conv_dim -> 1
        """
        super(G21, self).__init__()
        # Encoding blocks
        self.conv1 = conv(c_in=3, c_out=conv_dim,
                          k_size=4, stride=2, pad=1, bn=True)
        self.conv2 = conv(c_in=conv_dim, c_out=conv_dim * 2,
                          k_size=4, stride=2, pad=1, bn=True)

        # Residual blocks
        self.conv3 = conv(c_in=conv_dim * 2, c_out=conv_dim * 2,
                          k_size=3, stride=1, pad=1, bn=True)
        self.conv4 = conv(c_in=conv_dim * 2, c_out=conv_dim * 2,
                          k_size=3, stride=1, pad=1, bn=True)

        # Decoding blocks
        self.deconv1 = deconv(c_in=conv_dim * 2, c_out=conv_dim,
                              k_size=4, stride=2, pad=1, bn=True)
        self.deconv2 = deconv(c_in=conv_dim, c_out=1,
                              k_size=4, stride=2, pad=1, bn=False)

    def forward(self, x):
        out = x.reshape((-1, 3, 32, 32))
        out = F.leaky_relu(self.conv1(out), 0.05)    # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)    # (?, 128, 8, 8)

        out = F.leaky_relu(self.conv3(out), 0.05)    # ( " )
        out = F.leaky_relu(self.conv4(out), 0.05)    # ( " )

        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 64, 16, 16)
        out = F.tanh(self.deconv2(out))              # (?, 1, 32, 32)
        return out


class D1(nn.Module):

    """Discriminator for MNIST."""

    def __init__(self, conv_dim=64, use_labels=False):
        """Initialize a new instance.

        Parameters
        ----------
        conv_dim : int, optional
            Base for the number of convolutional channels. The sequence of
            channels is ::

                1 -> conv_dim -> 2 * conv_dim -> 4 *conv_dim -> n_out

            Here, ``n_out`` is 1 for ``use_labels=False`` and 11 otherwise.

        use_labels : bool, optional
            Whether or not to use the image labels. For ``True``, the
            discriminator performs classification with one extra "fake"
            class. For ``False``, the discriminator labels as "true" vs.
            "fake".
        """
        super(D1, self).__init__()
        n_out = 11 if use_labels else 1
        self.conv1 = conv(c_in=1, c_out=conv_dim,
                          k_size=4, stride=2, pad=1, bn=False)
        self.conv2 = conv(c_in=conv_dim, c_out=conv_dim * 2,
                          k_size=4, stride=2, pad=1, bn=True)
        self.conv3 = conv(c_in=conv_dim * 2, c_out=conv_dim * 4,
                          k_size=4, stride=2, pad=1, bn=True)
        self.fc = conv(c_in=conv_dim * 4, c_out=n_out,
                       k_size=4, stride=1, pad=0, bn=False)

    def forward(self, x):
        out = x.reshape((-1, 1, 28, 28))
        out = F.leaky_relu(self.conv1(out), 0.05)  # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        out = self.fc(out).squeeze()
        return out


class D2(nn.Module):

    """Discriminator for SVHN."""

    def __init__(self, conv_dim=64, use_labels=False):
        """Initialize a new instance.

        Parameters
        ----------
        conv_dim : int, optional
            Base for the number of convolutional channels. The sequence of
            channels is ::

                3 -> conv_dim -> 2 * conv_dim -> 4 *conv_dim -> n_out

            Here, ``n_out`` is 1 for ``use_labels=False`` and 11 otherwise.

        use_labels : bool, optional
            Whether or not to use the image labels. For ``True``, the
            discriminator performs classification with one extra "fake"
            class. For ``False``, the discriminator labels as "true" vs.
            "fake".
        """
        super(D2, self).__init__()
        n_out = 11 if use_labels else 1
        self.conv1 = conv(c_in=3, c_out=conv_dim,
                          k_size=4, stride=2, pad=1, bn=False)
        self.conv2 = conv(c_in=conv_dim, c_out=conv_dim * 2,
                          k_size=4, stride=2, pad=1, bn=True)
        self.conv3 = conv(c_in=conv_dim * 2, c_out=conv_dim * 4,
                          k_size=4, stride=2, pad=1, bn=True)
        self.fc = conv(c_in=conv_dim * 4, c_out=n_out,
                       k_size=4, stride=1, pad=0, bn=False)

    def forward(self, x):
        out = x.reshape((-1, 3, 32, 32))
        out = F.leaky_relu(self.conv1(out), 0.05)  # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        out = self.fc(out).squeeze()
        return out
