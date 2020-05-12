from torch.utils.tensorboard import SummaryWriter

class Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Logger, self).__init__(logdir)

    def log_train(self, lossL1, loss_advG, lossG, loss_real, loss_fake, loss_advD, M, k, steps):
        self.add_scalar('train/gen/lossL1', lossL1.item(), steps)
        self.add_scalar('train/gen/loss_advG', loss_advG.item(), steps)
        self.add_scalar('train/gen/lossG', lossG.item(), steps)

        self.add_scalar('train/dis/loss_real', loss_real.item(), steps)
        self.add_scalar('train/dis/loss_fake', loss_fake.item(), steps)
        self.add_scalar('train/dis/loss_advD', loss_advD.item(), steps)

        self.add_scalar('train/dis/measure_M', M, steps)
        self.add_scalar('train/dis/measure_k', k, steps)

    def log_valid(self, lossL1, loss_advG, lossG, steps):
        self.add_scalar('valid/gen/lossL1', lossL1.item(), steps)
        self.add_scalar('valid/gen/loss_advG', loss_advG.item(), steps)
        self.add_scalar('valid/gen/lossG', lossG.item(), steps)