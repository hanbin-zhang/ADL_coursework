#!/usr/bin/env python3
import statistics
import sys
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import argparse
from pathlib import Path

from dataset import MagnaTagATune
from evaluation import evaluate

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
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
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

parser.add_argument(
    "--sgd-momentum",
    default=0,
    type=float,
    help="Value of SGD momentum")

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


class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


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

    model = CNN(channels=1, num_samples=34950, sub_clips=10, class_count=10,
                stride_conv_size=args.stride_conv_length, stride_conv_stride=args.stride_conv_stride)

    # TASK 8: Redefine the criterion to be softmax cross entropy
    criterion = nn.BCELoss()

    # TASK 11: Define the optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.sgd_momentum)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
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
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    summary_writer.close()


class CNN(nn.Module):
    def __init__(self, sub_clips: int, channels: int,
                 num_samples: int, class_count: int,
                 stride_conv_size: int, stride_conv_stride: int):
        super().__init__()

        self.class_count = class_count

        # TODO:could this layer have more numbers of filter
        self.sConv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels*32,
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

        # self.fc1 = None
        # self.batchNorm1d3 = None
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

        x = torch.reshape(x.flatten(start_dim=0),
                          (-1, 10, int(x.shape[1] * x.shape[2] / 10)))
        # if self.fc1 is None:
        #     self.fc1 = nn.Linear(x.shape[2], 100)
        #     self.initialise_layer(self.fc1)
        #     self.batchNorm1d3 = nn.BatchNorm1d(self.fc1.out_features)

        x = x.view(-1, x.shape[2])
        x = F.relu(self.batchNorm1d3(self.fc1(x)))

        x = torch.sigmoid(self.fc2(x).reshape(audio.shape[0], 10, 50).mean(dim=1))

        # print(x.shape)
        # sys.exit()

        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)


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
            val_frequency: int,
            print_frequency: int = 20,
            log_frequency: int = 5,
            start_epoch: int = 0
    ):
        # self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for _, batch, labels in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()

                # TASK 1: Compute the forward pass of the model, print the output shape
                #         and quit the program
                # output = self.model.forward(batch)
                # print(output.shape)
                # import sys
                # sys.exit(1)

                # TASK 7: Rename `output` to `logits`, remove the output shape printing
                #         and get rid of the `import sys; sys.exit(1)`
                logits = self.model.forward(batch)

                # TASK 9: Compute the loss using self.criterion and
                #         store it in a variable called `loss`
                loss = self.criterion(logits, labels)

                # TASK 10: Compute the backward pass
                loss.backward()
                # TASK 12: Step the optimizer and then zero out the gradient buffers.
                self.optimizer.step()
                self.optimizer.zero_grad()

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, 4396, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, 4396, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if True:
                self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
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
                # preds = logits.argmax(dim=-1).cpu().numpy()
                preds = logits

                tensor_list.append(preds)
                # results["preds"].extend(list(preds))
                # results["labels"].extend(list(labels.cpu().numpy()))

        accuracy = evaluate(torch.cat(tensor_list, dim=0).cuda(), self.path_to_pkl)
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


def compute_accuracy(
        labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    # Get the number of items in the batch (j) and the number of tags (k)
    j, k = labels.shape

    # Initialize a list to store individual AUROC scores
    auroc_per_item = []

    for i in range(j):
        # Extract true labels and predicted probabilities for the i-th item
        y_true_item = labels[i]
        y_pred_item = preds[i]

        # Sort predictions and true labels based on predicted probabilities
        sorted_indices = torch.argsort(y_pred_item, descending=True)
        y_true_sorted = y_true_item[sorted_indices]

        # Count the number of true positives and false positives at each threshold
        num_true_positives = torch.cumsum(y_true_sorted, dim=0)
        num_false_positives = torch.cumsum(1 - y_true_sorted, dim=0)

        # Compute true positive rate (sensitivity) and false positive rate
        true_positive_rate = num_true_positives / torch.sum(y_true_item)
        false_positive_rate = num_false_positives / torch.sum(1 - y_true_item)

        # Compute AUROC using the trapezoidal rule
        auroc = torch.trapz(true_positive_rate, false_positive_rate)
        auroc_per_item.append(auroc.item())

    return statistics.mean(auroc_per_item)


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
    tb_log_dir_prefix = f'CNN_bn_bs={args.batch_size}_lr={args.learning_rate}_momentum={args.sgd_momentum}_run_'
    tb_log_dir_prefix += f'strde_conv_size,stride({args.stride_conv_length}, {args.stride_conv_stride})_'
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


if __name__ == "__main__":
    main(parser.parse_args())
