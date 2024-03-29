#!/usr/bin/env python3
import statistics
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
import numpy as np
from sklearn.metrics import roc_auc_score
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse
from pathlib import Path

from dataset import MagnaTagATune
from evaluation import evaluate

import pandas as pd

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a simple CNN on END-TO-END LEARNING FOR MUSIC AUDIO",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
default_dataset_dir = r"/mnt/storage/scratch/ey20699/MagnaTagATune"
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=1e-2, type=float, help="Learning rate")
parser.add_argument(
    "--batch-size",
    default=10,
    type=int,
    help="Number of examples within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=5,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)

# Add a string argument with allowed values
parser.add_argument(
    '--optimizer',
    type=str,
    choices=['sgd', 'adam', 'adamW'],
    default='sgd',
    help='Specify an optimizer (sgd, adam, adamW).')

parser.add_argument(
    "--sgd-momentum",
    default=0,
    type=float,
    help="Value of SGD momentum, only works when optimizer specify to 'sgd'")

parser.add_argument(
    "--stride-conv-length",
    default=256,
    type=int,
    help="Value of stride convolution kernel length")

parser.add_argument(
    "--stride-conv-stride",
    default=256,
    type=int,
    help="Value of stride convolution kernel stride")

# Add a string argument with allowed values
parser.add_argument(
    '--model',
    type=str,
    choices=['base', 'more', 'super'],
    default='base',
    help='Specify the model version (base, more, super).')

parser.add_argument(
    "--dropout",
    default=0.3,
    type=float,
    help="Value of dropout rate")


class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def initialize_optimizer(model, args):
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.sgd_momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'adamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    return optimizer


def main(args):
    trainMagnaTagATune = MagnaTagATune(args.dataset_root + "/annotations/train_labels.pkl",
                                       args.dataset_root + "/samples")
    gts_pkl_path = args.dataset_root + "/annotations/val_labels.pkl"

    validateMagnaTagATune = MagnaTagATune(gts_pkl_path,
                                          args.dataset_root + "/samples")
    train_loader = torch.utils.data.DataLoader(
        trainMagnaTagATune,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    test_loader = torch.utils.data.DataLoader(
        validateMagnaTagATune,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    if args.model == 'more':
        model = CNNMore(channels=1, num_samples=34950, sub_clips=10, class_count=10,
                        stride_conv_size=args.stride_conv_length, stride_conv_stride=args.stride_conv_stride)
    elif args.model == 'super':
        model = CNNSuper(channels=1, num_samples=34950, sub_clips=10, class_count=10,
                         stride_conv_size=args.stride_conv_length, stride_conv_stride=args.stride_conv_stride,
                         dropout_ratio=args.dropout)
    else:
        model = CNN(channels=1, num_samples=34950, sub_clips=10, class_count=10,
                    stride_conv_size=args.stride_conv_length, stride_conv_stride=args.stride_conv_stride)

    # TASK 8: Redefine the criterion to be softmax cross entropy
    criterion = nn.BCELoss()

    # TASK 11: Define the optimizer

    optimizer = initialize_optimizer(model, args)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
        str(log_dir),
        flush_secs=5
    )
    trainer = Trainer(
        model, train_loader, test_loader, criterion, optimizer, summary_writer, DEVICE, gts_pkl_path
    )

    trainer.train(
        args.epochs,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    # save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        # Add any other information you want to save
    }, 'model.pth')

    summary_writer.close()


class CNNMore(nn.Module):
    def __init__(self, sub_clips: int, channels: int,
                 num_samples: int, class_count: int,
                 stride_conv_size: int, stride_conv_stride: int):
        super().__init__()

        self.class_count = class_count

        # TODO:could this layer have more numbers of filter
        self.sConv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels * 32,
            kernel_size=stride_conv_size,
            stride=stride_conv_stride
        )
        self.initialise_layer(self.sConv)

        self.conv1d1 = nn.Conv1d(
            in_channels=channels * 32,
            out_channels=channels * 32,
            kernel_size=8,
            padding='same'
        )
        self.initialise_layer(self.conv1d1)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.batchNorm1d1 = nn.BatchNorm1d(self.conv1d1.out_channels)

        self.conv1d2 = nn.Conv1d(
            in_channels=self.conv1d1.out_channels,
            out_channels=self.conv1d1.out_channels * 32,
            kernel_size=8,
            padding='same'
        )
        self.initialise_layer(self.conv1d2)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.batchNorm1d2 = nn.BatchNorm1d(self.conv1d2.out_channels)

        self.fc1 = nn.Linear(8704, 100)
        self.initialise_layer(self.fc1)
        self.batchNorm1d3 = nn.BatchNorm1d(self.fc1.out_features)

        self.fc2 = nn.Linear(100, 50)
        self.initialise_layer(self.fc2)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = audio
        x = F.relu(self.sConv(torch.reshape(x.flatten(start_dim=0),
                                            (audio.shape[0], 1, audio.shape[1] * audio.shape[3]))))

        x = F.relu(self.batchNorm1d1(self.conv1d1(x)))
        x = self.pool1(x)

        x = F.relu(self.batchNorm1d2(self.conv1d2(x)))
        x = self.pool2(x)

        x = torch.reshape(x, (audio.size(0), -1))

        # Check if the size of the last dimension is not a multiple of 10
        if x.size(1) % 10 != 0:
            # Calculate the padding needed to make the size a multiple of 10

            padding_size = (10 - x.size(1) % 10) % 10

            # Pad the last dimension
            x = F.pad(x, (0, padding_size))

        x = torch.reshape(x,
                          (audio.size(0), 10, -1))

        x = x.view(-1, x.shape[2])
        fc_input_size = x.size(1)

        # Update fc layer sizes if necessary
        if self.fc1.in_features != fc_input_size:
            self.fc1 = nn.Linear(fc_input_size, 100).to(x.device)
            self.initialise_layer(self.fc1)
        x = F.relu(self.batchNorm1d3(self.fc1(x)))

        x = torch.sigmoid(self.fc2(x).reshape(audio.shape[0], 10, 50).mean(dim=1))

        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)


class CNN(nn.Module):
    def __init__(self, sub_clips: int, channels: int,
                 num_samples: int, class_count: int,
                 stride_conv_size: int, stride_conv_stride: int,
                 second_kernel_number: int = 1):
        super().__init__()

        self.class_count = class_count

        # TODO:could this layer have more numbers of filter
        self.sConv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels * 32,
            kernel_size=stride_conv_size,
            stride=stride_conv_stride
        )
        self.initialise_layer(self.sConv)

        self.poolsC = nn.AdaptiveAvgPool1d(output_size=1365)

        self.conv1d1 = nn.Conv1d(
            in_channels=channels * 32,
            out_channels=channels * 32,
            kernel_size=8,
            padding='same'
        )
        self.initialise_layer(self.conv1d1)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.batchNorm1d1 = nn.BatchNorm1d(self.conv1d1.out_channels)

        self.conv1d2 = nn.Conv1d(
            in_channels=self.conv1d1.out_channels,
            out_channels=self.conv1d1.out_channels * self.class_count,
            kernel_size=8,
            padding='same'
        )
        self.initialise_layer(self.conv1d2)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.batchNorm1d2 = nn.BatchNorm1d(self.conv1d2.out_channels)

        self.fc1 = nn.Linear(1, 100)
        self.batchNorm1d3 = nn.BatchNorm1d(self.fc1.out_features)

        self.fc2 = nn.Linear(100, 50)
        self.initialise_layer(self.fc2)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = audio
        x = F.relu(self.sConv(torch.reshape(x.flatten(start_dim=0),
                                            (audio.shape[0], 1, audio.shape[1] * audio.shape[3]))))

        x = F.relu(self.batchNorm1d1(self.conv1d1(x)))
        x = self.pool1(x)
        x = F.relu(self.batchNorm1d2(self.conv1d2(x)))
        x = self.pool2(x)
        x = torch.reshape(x, (audio.size(0), -1))

        # Check if the size of the last dimension is not a multiple of 10
        if x.size(1) % 10 != 0:
            # Calculate the padding needed to make the size a multiple of 10
            padding_size = (10 - x.size(1) % 10) % 10

            # Pad the last dimension
            x = F.pad(x, (0, padding_size))

        x = torch.reshape(x,
                          (audio.size(0), 10, -1))

        x = x.view(-1, x.shape[2])
        fc_input_size = x.size(1)

        # Update fc layer sizes if necessary
        if self.fc1.in_features != fc_input_size:
            self.fc1 = nn.Linear(fc_input_size, 100).to(x.device)
            self.initialise_layer(self.fc1)
            print(self.fc1.in_features)
        x = F.relu(self.batchNorm1d3(self.fc1(x)))

        x = torch.sigmoid(self.fc2(x).reshape(audio.shape[0], 10, 50).mean(dim=1))

        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)


class CNNSuper(CNN):
    def __init__(self, sub_clips: int, channels: int,
                 num_samples: int, class_count: int,
                 stride_conv_size: int, stride_conv_stride: int,
                 dropout_ratio: float,
                 second_kernel_number: int = 1,
                 ):
        super(CNNSuper, self).__init__(sub_clips, channels, num_samples, class_count,
                                       stride_conv_size, stride_conv_stride, second_kernel_number)

        # TODO:could this layer have more numbers of filter
        self.sConv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels * 32,
            kernel_size=stride_conv_size,
            stride=stride_conv_stride
        )
        self.initialise_layer(self.sConv)

        self.conv1d1 = nn.Conv1d(
            in_channels=channels * 32,
            out_channels=channels * 32,
            kernel_size=8,
            padding='same'
        )

        self.initialise_layer(self.conv1d1)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.batchNorm1d1 = nn.BatchNorm1d(self.conv1d1.out_channels)

        self.conv1d2 = nn.Conv1d(
            in_channels=self.conv1d1.out_channels,
            out_channels=self.conv1d1.out_channels * 32,
            kernel_size=8,
            padding='same'
        )
        self.initialise_layer(self.conv1d2)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.batchNorm1d2 = nn.BatchNorm1d(self.conv1d2.out_channels)

        self.conv1d3 = nn.Conv1d(
            in_channels=self.conv1d2.out_channels,
            out_channels=self.conv1d2.out_channels,
            kernel_size=8,
            padding='same'
        )
        self.initialise_layer(self.conv1d3)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.batchNorm1d3 = nn.BatchNorm1d(self.conv1d3.out_channels)

        self.conv1d4 = nn.Conv1d(
            in_channels=self.conv1d3.out_channels,
            out_channels=self.conv1d3.out_channels,
            kernel_size=8,
            padding='same'
        )
        self.initialise_layer(self.conv1d4)
        self.batchNorm1d4 = nn.BatchNorm1d(self.conv1d4.out_channels)

        self.dropout1 = nn.Dropout1d(p=dropout_ratio)

        self.conv1d5 = nn.Conv1d(
            in_channels=self.conv1d4.out_channels,
            out_channels=self.conv1d4.out_channels,
            kernel_size=8,
            padding='same'
        )
        self.initialise_layer(self.conv1d5)
        self.batchNorm1d5 = nn.BatchNorm1d(self.conv1d5.out_channels)

        self.poolFinal = nn.MaxPool1d(kernel_size=4, stride=4)

        self.fc1 = nn.Linear(8704, 100)
        self.initialise_layer(self.fc1)
        self.batchNorm1dfc1 = nn.BatchNorm1d(self.fc1.out_features)

        self.fc2 = nn.Linear(100, 50)
        self.initialise_layer(self.fc2)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = audio
        x = F.relu(self.sConv(torch.reshape(x.flatten(start_dim=0),
                                            (audio.shape[0], 1, audio.shape[1] * audio.shape[3]))))

        x = self.pool1(F.relu(self.batchNorm1d1(self.conv1d1(x))))
        x = self.pool2(F.relu(self.batchNorm1d2(self.conv1d2(x))))
        x = self.pool3(F.relu(self.batchNorm1d3(self.conv1d3(x))))

        residual = x.clone()

        x = self.dropout1(F.relu(self.batchNorm1d4(self.conv1d4(x))))
        x = self.poolFinal(F.relu(self.batchNorm1d5(self.conv1d5(x)) + residual))

        x = torch.reshape(x, (audio.size(0), -1))
        # Check if the size of the last dimension is not a multiple of 10
        if x.size(1) % 10 != 0:
            # Calculate the padding needed to make the size a multiple of 10
            padding_size = (10 - x.size(1) % 10) % 10

            # Pad the last dimension
            x = F.pad(x, (0, padding_size))

        x = torch.reshape(x,
                          (audio.size(0), 10, -1))

        x = x.view(-1, x.shape[2])
        fc_input_size = x.size(1)

        # Update fc layer sizes if necessary
        if self.fc1.in_features != fc_input_size:
            self.fc1 = nn.Linear(fc_input_size, 100).to(x.device)
            self.initialise_layer(self.fc1)
        x = F.relu(self.batchNorm1dfc1(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x).reshape(audio.shape[0], 10, 50).mean(dim=1))

        return x


def find_per_class_accucy(preds, gts_path):
    # gts = torch.load(gts_path, map_location='cpu') # Ground truth labels, pass path to val.pkl
    gts = pd.read_pickle(gts_path)

    labels = []
    model_outs = []
    for i in range(len(preds)):
        labels.append(np.array(gts.iloc[i]['label']).astype(float))  # A 50D Ground Truth binary vector
        model_outs.append(preds[i].cpu().numpy())  # A 50D vector that assigns probability to each class

    labels = np.array(labels).astype(float)
    model_outs = np.array(model_outs)

    auc_score = roc_auc_score(y_true=labels, y_score=model_outs, average=None)

    print(auc_score)

    return


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module,
            optimizer: Optimizer,
            summary_writer: SummaryWriter,
            device: torch.device,
            path_to_pkl: str,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0
        self.path_to_pkl = path_to_pkl

    def train(
            self,
            epochs: int,
            print_frequency: int = 20,
            log_frequency: int = 5,
            start_epoch: int = 0
    ):
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()

            all_labels = []
            model_outs = []
            total_loss = 0
            for _, batch, labels in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()

                # TASK 1: Compute the forward pass of the model, print the output shape
                #         and quit the program

                # TASK 7: Rename `output` to `logits`, remove the output shape printing
                #         and get rid of the `import sys; sys.exit(1)`
                logits = self.model.forward(batch)

                # TASK 9: Compute the loss using self.criterion and
                #         store it in a variable called `loss`
                loss = self.criterion(logits, labels)

                # Compute accuracy
                all_labels.append(labels)
                model_outs.append(logits.detach())

                total_loss += loss.item()

                # TASK 10: Compute the backward pass
                loss.backward()
                # TASK 12: Step the optimizer and then zero out the gradient buffers.
                self.optimizer.step()
                self.optimizer.zero_grad()

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time

                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, 4396, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            epoch_loss = total_loss / len(self.train_loader)
            all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
            model_outs = torch.cat(model_outs, dim=0).cpu().numpy().astype(float)
            self.log_train_metrics(epoch, roc_auc_score(y_true=all_labels, y_score=model_outs), epoch_loss)
            if True:
                self.validate()
                self.model.train()

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
            f"epoch: [{epoch}], "
            f"step: [{epoch_step}/{len(self.train_loader)}], "
            f"batch loss: {loss:.5f}, "
            f"batch accuracy: {accuracy * 100:2.2f}, "
            f"data load time: "
            f"{data_load_time:.5f}, "
            f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
            "accuracy",
            {"train": accuracy},
            self.step
        )
        self.summary_writer.add_scalars(
            "loss",
            {"train": float(loss.item())},
            self.step
        )
        self.summary_writer.add_scalar(
            "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
            "time/data", step_time, self.step
        )

    def log_train_metrics(self, epoch, accuracy, loss):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
            "accuracy",
            {"train": accuracy},
            self.step
        )
        self.summary_writer.add_scalars(
            "loss",
            {"train": float(loss)},
            self.step
        )

    def validate(self):
        # results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()
        tensor_list = []
        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for _, batch, labels in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits

                tensor_list.append(preds)

        accuracy = evaluate(torch.cat(tensor_list, dim=0), self.path_to_pkl)
        average_loss = total_loss / len(self.val_loader)

        self.summary_writer.add_scalars(
            "accuracy",
            {"test": accuracy},
            self.step
        )
        self.summary_writer.add_scalars(
            "loss",
            {"test": average_loss},
            self.step
        )
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")


def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = f'CNN_MIR_bs={args.batch_size}_lr={args.learning_rate}_momentum={args.sgd_momentum}_run_'
    tb_log_dir_prefix += f'strde_conv_size,stride({args.stride_conv_length}, {args.stride_conv_stride})_'
    tb_log_dir_prefix += f'optimizer={args.optimizer}_'
    tb_log_dir_prefix += f'model={args.model}_'

    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


if __name__ == "__main__":
    main(parser.parse_args())
