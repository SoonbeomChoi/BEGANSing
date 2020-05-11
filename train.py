import sys
sys.path.append('utils')
sys.path.append('model')

import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config_parser import Config
from file_utils import create_path
from torch_utils import set_device, save_checkpoint

import dataprocess
from models import Generator, Discriminator

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.num = 0
        self.avg = 0.0
        self.steps = 0

    def step(self, val, n=1):
        self.val = val
        self.sum += n*val
        self.num += n
        self.steps += 1
        self.avg = self.sum/self.num

def criterionGAN(D, x):
    loss_gan = torch.mean(torch.abs(D(x) - x))
    return loss_gan

def main():
    config = Config()

    config.basename = os.path.basename(config.file)
    checkpoint_path = create_path(config.checkpoint_path)
    config.save(os.path.join(create_path, config))
    writer = SummaryWriter(checkpoint_path)

    dataloader = dataprocess.load(config)
    step_size = config.step_epoch*len(dataloader.train)

    G = Generator(config)
    D = Discriminator(config)
    criterionL1 = nn.L1Loss()
    G, D = set_device((G, D), config.device, config.use_cpu)

    optimizerG = torch.optim.Adam(G.parameters(), lr=config.learn_rate, betas=config.betas, weight_decay=config.weight_decay)
    optimizerD = torch.optim.Adam(D.parameters(), lr=config.learn_rate, betas=config.betas, weight_decay=config.weight_decay)
    schedulerG = StepLR(optimizerG, step_size=step_size, gamma=config.decay_factor)
    schedulerD = StepLR(optimizerD, step_size=step_size, gamma=config.decay_factor)
    M = AverageMeter()

    lossG_train = AverageMeter()
    lossG_valid = AverageMeter()
    lossD_train = AverageMeter()
    for epoch in range(config.stop_epoch + 1):
        G.train()
        D.train()
        for batch in tqdm(dataloader.train, leave=False, ascii=True):
            x, y_prev, y = set_device(batch, config.device, config.use_cpu)

            optimizerG.zero_grad()
            y_gen = G(x, y_prev)
            lossL1 = criterionL1(y_gen, y)
            lossGAN = criterionGAN(D, y_gen)
            lossG = lossL1 + lossGAN
            lossG.backward()
            optimizerG.step()
            schedulerG.step()

            optimizerD.zero_grad()
            loss_real = criterionGAN(D, y)
            loss_fake = criterionGAN(D, y_gen.detach())
            lossD = loss_real - k*loss_fake
            lossD.backward()
            optimizerD.step()
            schedulerD.step()

            diff = torch.mean(config.gamma*loss_real - loss_fake)
            k = k + config.lambda_k*diff.item()
            k = min(max(k, 0), 1)

            measure = (loss_real + torch.abs(diff)).data
            M.step(measure, y.size(0))

            lossG_train.step(lossG.item(), y.size(0))
            lossD_train.step(lossD.item(), y.size(0))

        G.eval()
        D.eval()
        for batch in tqdm(dataloader.valid, leave=False, ascii=True):
            x, y_prev, y = set_device(batch, config.device, config.use_cpu)

            y_gen = G(x, y_prev)
            lossL1 = criterionL1(y_gen, y)
            lossGAN = criterionGAN(D, y_gen)
            lossG = lossL1 + lossGAN

            lossG_valid.step(lossG.item(), y.size(0))

    for param_group in optimizerG.param_groups:
        learn_rate = param_group['lr']

    print("[Epoch %d/%d] [loss G train: %.5f] [loss G valid: %.5f] [loss D train: %.5f] [M: %.5f] [k: %.5f] [lr: %.6f]" %
          (epoch, config.stop_epoch, lossG_train.avg, lossG_valid.avg, lossD_train, M.avg, k, learn_rate))
    
    lossG_train.reset()
    lossG_valid.reset()
    lossD_train.reset()

    savename = os.path.join(checkpoint_path, 'latest')
    save_checkpoint(savename + 'G.pt', G, optimizerG, learn_rate, lossG_train.steps)
    save_checkpoint(savename + 'D.pt', D, optimizerD, learn_rate, lossD_train.steps)
    if epoch%config.save_epoch == 0:
        savename = os.path.join(checkpoint_path, 'epoch' + str(epoch))
        save_checkpoint(savename + 'G.pt', G, optimizerG, learn_rate, lossG_train.steps)
        save_checkpoint(savename + 'D.pt', D, optimizerD, learn_rate, lossD_train.steps)

if __name__ == "__main__":
    main()