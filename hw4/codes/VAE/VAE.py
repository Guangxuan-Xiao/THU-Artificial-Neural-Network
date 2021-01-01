import torch.nn as nn
import torch
import os


class VAE(nn.Module):
    def __init__(self, num_channals, latent_dim, no_cnn=False):
        super(VAE, self).__init__()
        self.num_channals = num_channals
        self.latent_dim = latent_dim
        # Define the architecture of VAE here
        # TODO START
        self.no_cnn = no_cnn
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=16,
                      kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.encoder_mlp = nn.Sequential(
            nn.Linear(16 * 8 * 8, 2 * latent_dim),
            nn.ReLU(),
            nn.Linear(2 * latent_dim, 2 * latent_dim)
        ) if not self.no_cnn else nn.Sequential(
            nn.Linear(1 * 32 * 32, 2 * latent_dim),
            nn.ReLU(),
            nn.Linear(2 * latent_dim, 2 * latent_dim)
        )
        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 16 * 8 * 8),
        ) if not self.no_cnn else nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1 * 32 * 32),
            nn.Sigmoid(),
        )

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,
                               padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=1,
                               padding=1, output_padding=0),
            nn.Sigmoid(),
        )
        # TODO END

    def reparameterize(self, mu, log_var):
        '''
        *   Arguments:
            *   mu (torch.FloatTensor): [batch_size, latent_dim], parameters of the diagnoal Gaussian posterior q(z|x)
            *   log_var (torch.FloatTensor): [batch_size, latent_dim], parameters of the diagnoal Gaussian posterior q(z|x)
        *   Returns:
            *   reparameterized samples (torch.FloatTensor): [batch_size, latent_dim]
        '''
        # TODO START
        sampled_z = mu + torch.exp(log_var / 2) * torch.randn_like(log_var)
        return sampled_z
        # TODO END

    def encode(self, x):
        if not self.no_cnn:
            x = self.encoder_cnn(x)
        x = nn.Flatten()(x)
        x = self.encoder_mlp(x)
        mu, log_var = x.chunk(2, 1)
        return mu, log_var

    def decode(self, z):
        z = self.decoder_mlp(z)
        if not self.no_cnn:
            z = z.view(z.size(0), 16, 8, 8)
            gen_x = self.decoder_cnn(z)
        else:
            gen_x = z.view(z.size(0), 1, 32, 32)
        return gen_x

    def forward(self, x=None, z=None):
        '''
        *   Arguments:
            *   x (torch.FloatTensor): [batch_size, 1, 32, 32]
            *   z (torch.FloatTensor): [batch_size, latent_dim]
        *   Returns:
            *   if `x` is not `None`, return a list:
                *   Reconstruction of `x` (torch.FloatTensor)
                *   mu (torch.FloatTensor): [batch_size, latent_dim], parameters of the diagnoal Gaussian posterior q(z|x)
                *   log_var (torch.FloatTensor): [batch_size, latent_dim], parameters of the diagnoal Gaussian posterior q(z|x)
            *  if `x` is `None`, return samples generated from the given `z` (torch.FloatTensor): [num_samples, 1, 32, 32]
        '''
        if x is not None:
            # TODO START
            mu, log_var = self.encode(x)
            sampled_z = self.reparameterize(mu, log_var)
            recon_x = self.decode(sampled_z)
            return recon_x, mu, log_var
            # TODO END
        else:
            assert z is not None
            # TODO START
            return self.decode(z)
            # TODO END

    def restore(self, ckpt_dir):
        try:
            if os.path.exists(os.path.join(ckpt_dir, 'pytorch_model.bin')):
                path = os.path.join(ckpt_dir, 'pytorch_model.bin')
            else:
                path = os.path.join(ckpt_dir, str(
                    max(int(name) for name in os.listdir(ckpt_dir))), 'pytorch_model.bin')
        except:
            return None
        self.load_state_dict(torch.load(path))
        return os.path.split(path)[0]

    def save(self, ckpt_dir, global_step):
        os.makedirs(os.path.join(ckpt_dir, str(global_step)), exist_ok=True)
        path = os.path.join(ckpt_dir, str(global_step), 'pytorch_model.bin')
        torch.save(self.state_dict(), path)
        return os.path.split(path)[0]
