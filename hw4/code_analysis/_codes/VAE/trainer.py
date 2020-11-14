import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class Trainer(object):
    def __init__(self, device, model, optimizer, dataset, ckpt_dir, tb_writer):
        self._device = device
        self._model = model.to(self._device)
        self._optimizer = optimizer
        self._dataset = dataset
        self._ckpt_dir = ckpt_dir
        self._tb_writer = tb_writer
        os.makedirs(ckpt_dir, exist_ok=True)
        self._model.restore(ckpt_dir)

    def loss_fn(self, target, recon, mu, log_var):
        '''
        *   Arguments:
            *   target (torch.FloatTensor): [batch_size, 1, 32, 32], the original image
            *   recon (torch.FloatTensor): [batch_size, 1, 32, 32], reconstructions
            *   mu (torch.FloatTensor): [batch_size, latent_dim], parameters of the diagnoal Gaussian posterior q(z|x)
            *   log_var (torch.FloatTensor): [batch_size, latent_dim], parameters of the diagnoal Gaussian posterior q(z|x)
        *   Returns:
            *   Reconstruction loss (scalar): **average** reconstruction loss within a batch
            *   kl divergence loss (scalar): **average** kl divergence loss within a batch
        '''
        # TODO START
		
		return recon_loss, kl_loss
        # TODO END

    def train(self, num_training_updates, logging_steps, saving_steps):
        iterator = iter(cycle(self._dataset.training_loader))
        fixed_noise = torch.randn(32, self._model.latent_dim, device=self._device)
        for i in tqdm(range(num_training_updates), desc='Training'):
            (inp, _) = next(iterator)
            inp = inp.to(self._device)
            self._model.train()
            self._model.zero_grad()
            out, mu, log_var = self._model(inp)
            recon_loss, kl_div = self.loss_fn(inp, out, mu, log_var)
            loss = recon_loss + kl_div
            loss.backward()
            self._optimizer.step()

            if (i + 1) % logging_steps == 0:
                self._tb_writer.add_scalar("reconstruction_loss", recon_loss, global_step=i)
                self._tb_writer.add_scalar("KL_divergence", kl_div, global_step=i)
                self._tb_writer.add_scalar("loss", loss, global_step=i)
            if (i + 1) % saving_steps == 0:
                dirname = self._model.save(self._ckpt_dir, i)
                dev_imgs, recons, samples, eval_recon_loss, eval_kl_div = self.evaluate(fixed_noise)
                self._tb_writer.add_scalar('dev/reconstruction_loss', eval_recon_loss, global_step=i)
                self._tb_writer.add_scalar('dev/KL_divergence', eval_kl_div, global_step=i)
                self._tb_writer.add_scalar('dev/loss', eval_recon_loss + eval_kl_div, global_step=i)
                for imgs, name in zip([dev_imgs, recons, samples], ['dev_imgs', 'reconstructions', 'samples']):
                    self._tb_writer.add_image(name, imgs, global_step=i)
                    save_image(imgs, os.path.join(dirname, "{}.png".format(name)))

    def evaluate(self, fixed_noise):
        self._model.eval()
        with torch.no_grad():
            dev_imgs, _ = next(iter(self._dataset.validation_loader))
            dev_imgs = dev_imgs.to(self._device)
            recons, _, _ = self._model(dev_imgs)
            samples = self._model(z=fixed_noise)
            dev_imgs = make_grid(dev_imgs)
            recons = make_grid(recons)
            samples = make_grid(samples)

            num_inst = 0
            recon_loss = 0.0
            kl_div_loss = 0.0
            for inp, _ in tqdm(self._dataset.validation_loader, desc='Evaluating'):
                inp = inp.to(self._device)
                out, mu, log_var = self._model(inp)
                recon, kl_div = self.loss_fn(inp, out, mu, log_var)

                num_inst += inp.size(0)
                recon_loss += recon * inp.size(0)
                kl_div_loss += kl_div * inp.size(0)
        return dev_imgs, recons, samples, recon_loss / num_inst, kl_div_loss / num_inst
