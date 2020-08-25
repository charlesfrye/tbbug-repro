import argparse
import random
import sys

from torch.utils.tensorboard import SummaryWriter

import wandb


class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument("--new_dir", action="store_const", const=True)
        self.cli_args = parser.parse_args(sys_argv)

        self.trn_writer = None
        self.val_writer = None

    def initTensorboardWriters(self, new_dir=False):
        if self.trn_writer is None:
            if new_dir:
                tb_dir = "tb"
            else:
                tb_dir = wandb.run.dir

            self.trn_writer = SummaryWriter(
                log_dir=tb_dir)
            # self.val_writer = SummaryWriter(
            #    log_dir=tb_dir)

    def main(self):
        wandb.init(config=self.cli_args, sync_tensorboard=True)

        for epoch_ndx in range(2):

            self.logMetrics(epoch_ndx, 'trn')
            # self.logMetrics(epoch_ndx, 'val')

        if self.trn_writer is not None:
            self.trn_writer.close()

        if self.val_writer is not None:
            self.val_writer.close()

        wandb.join()

    def logMetrics(self, epoch_ndx, mode_str):

        self.initTensorboardWriters(new_dir=self.cli_args.new_dir)

        writer = getattr(self, mode_str + '_writer')

        metrics_dict = {"foo": random.random(),
                        "bar": random.random()}

        for key, value in metrics_dict.items():
            writer.add_scalar(
                mode_str + "/" + key, value, epoch_ndx)


if __name__ == '__main__':
    LunaTrainingApp().main()
