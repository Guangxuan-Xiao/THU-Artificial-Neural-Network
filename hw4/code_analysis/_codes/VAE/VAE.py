import torch.nn as nn
import torch
import os

class VAE(nn.Module):
    def __init__(self, num_channals, latent_dim):
        super(VAE, self).__init__()
        self.num_channals = num_channals
        self.latent_dim = latent_dim
        # Define the architecture of VAE here
        # TODO START

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
		
		return sampled_z
        # TODO END

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
			
			return recon_x, mu, log_var
            # TODO END
        else:
            assert z is not None
            # TODO START
			
			return gen_x
            # TODO END

    def restore(self, ckpt_dir):
        try:
            if os.path.exists(os.path.join(ckpt_dir, 'pytorch_model.bin')):
                path = os.path.join(ckpt_dir, 'pytorch_model.bin')
            else:
                path = os.path.join(ckpt_dir, str(max(int(name) for name in os.listdir(ckpt_dir))), 'pytorch_model.bin')
        except:
            return None
        self.load_state_dict(torch.load(path))
        return os.path.split(path)[0]

    def save(self, ckpt_dir, global_step):
        os.makedirs(os.path.join(ckpt_dir, str(global_step)), exist_ok=True)
        path = os.path.join(ckpt_dir, str(global_step), 'pytorch_model.bin')
        torch.save(self.state_dict(), path)
        return os.path.split(path)[0]
