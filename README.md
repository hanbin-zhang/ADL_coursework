# CNN Model Training Readme

This README provides information on how to use the provided Python script for training a Convolutional Neural Network (CNN) on the MagnaTagATune dataset.

## Usage

### Script Execution
1. Open a terminal or command prompt.
2. Navigate to the directory containing the script (`train.py`).
3. Run the script with the desired command line arguments. Example:

   ```bash
   python train.py --dataset-root /path/to/dataset --learning-rate 0.001 --optimizer adamW --model super --dropout 0.2
   ```

### Command Line Arguments

- `--dataset-root`: Path to the root directory of the MagnaTagATune dataset.
- `--log-dir`: Path to the directory where logs will be stored.
- `--learning-rate`: Learning rate for the optimizer.
- `--batch-size`: Number of examples within each mini-batch.
- `--epochs`: Number of epochs (passes through the entire dataset) to train for.
- `--log-frequency`: How frequently to save logs to Tensorboard in the number of steps.
- `--print-frequency`: How frequently to print progress to the command line in the number of steps.
- `-j` or `--worker-count`: Number of worker processes used to load data.
- `--optimizer`: Specify an optimizer (sgd, adam, adamW).
- `--sgd-momentum`: Value of SGD momentum, only works when the optimizer is set to 'sgd'.
- `--stride-conv-length`: Value of stride convolution kernel length.
- `--stride-conv-stride`: Value of stride convolution kernel stride.
- `--model`: Specify the model version (base, more, super).
- `--dropout`: Value of dropout rate.

### Model Versions
The script supports three model versions:
- `base`: Base version of the CNN.
- `more`: A modified version of the CNN with more kernels.
- `super`: A super version with more convolutional layers, kernels, dropout and ResNet.

### Tensorboard Logs
Logs for Tensorboard are stored in the specified `--log-dir`. To visualize the logs, use the following command:

```bash
tensorboard --logdir /path/to/log-directory
```

Visit [http://localhost:6006](http://localhost:6006) in your browser to view Tensorboard.

## Saving the Model
The trained model is saved to a file named `model.pth` in the script's directory.

## Additional Notes
- The provided script is designed to train a CNN on the MagnaTagATune dataset. Ensure that you have the dataset available at the specified `--dataset-root`.
- Make sure to install the required Python packages before running the script.

For any additional help or inquiries, please refer to the script's author or documentation.