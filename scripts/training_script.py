# Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import os
import timm
import pandas as pd

from classifier_utils import ImageDataset, Model

def convert_to_grayscale(model: nn.Module) -> nn.Module:
    """
    Converts the first Conv2d layer of a model from 3-channel (RGB) 
    to 1-channel (Grayscale) using Rec. 601 luminance weights.
    """
    first_layer = None
    layer_name = ""

    # Shape: (1, 3, 1, 1) to align with (out_channels, in_channels, kernel, kernel)
    luma_weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if module.in_channels == 3:
                first_layer = module
                layer_name = name
                break
            else:
                # Model is already not RGB or logic found a non-input conv
                return model

    if first_layer is None:
        raise ValueError("No input Conv2d layer with 3 channels found.")

    # Create the new grayscale layer
    new_first = nn.Conv2d(
        in_channels=1,
        out_channels=first_layer.out_channels,
        kernel_size=first_layer.kernel_size,
        stride=first_layer.stride,
        padding=first_layer.padding,
        dilation=first_layer.dilation,
        groups=first_layer.groups,
        bias=(first_layer.bias is not None)
    )

    # Move luma_weights to the same device as the model parameters
    device = first_layer.weight.device
    luma_weights = luma_weights.to(device)

    with torch.no_grad():
        # Weighted sum across the channel dimension (dim=1)
        # Original weight shape: [Out, 3, K, K]
        # New weight shape: [Out, 1, K, K]
        weighted_weights = (first_layer.weight * luma_weights).sum(dim=1, keepdim=True)
        new_first.weight.copy_(weighted_weights)
        
        if first_layer.bias is not None:
            new_first.bias.copy_(first_layer.bias)

    # Replace the layer in the model hierarchy
    hierarchy = layer_name.split('.')
    target = model
    for attr in hierarchy[:-1]:
        target = getattr(target, attr)
    setattr(target, hierarchy[-1], new_first)

    return model

# Consts
LEARNING_RATE = 1e-4
BATCH_SIZE = 64

# Args
parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="Which model to use")
parser.add_argument("train_dir", type=Path, help="Path to directory of training images")
parser.add_argument("train_csv", type=Path, help="Path to csv of training images")
parser.add_argument("save_dir", type=Path, help="Path to directory to save model checkpoints")
parser.add_argument("epochs", type=int, help="Number of epochs to train the model")
parser.add_argument("--load_model", default=None, type=str, help="Model checkpoint to begin at")
parser.add_argument("--save_freq", default=10, type=int, help="Number of epochs between model saves")
parser.add_argument("--log", default=None, type=str, help="Output log files to where?")
parser.add_argument("--freeze", action="store_true", help="Freeze all but the final layer?")
parser.add_argument("--mix", action="store_true", help="Mix with another data set?")
parser.add_argument("--mix_dir", default=None, type=str, help="Path to directory of training images (for mixing)")
parser.add_argument("--mix_csv", default=None, type=str, help="Path to csv of training images (for mixing)")
parser.add_argument("--mix_ratio", default=0.5, type=float, help="Ratio of data mixing to achieve")
parser.add_argument("--path_lim", default=2, type=float, help="Maximum # of pathologies in an image")

args = parser.parse_args()

# Validate inputs
if not args.train_dir.is_dir():
    raise ValueError(f"Could not find directory {args.train_dir}")
if not args.save_dir.is_dir():
    os.makedirs(args.save_dir, exist_ok=True)

# Create datasets and dataloaders
train_df = pd.read_csv(args.train_csv)

subset = train_df.iloc[:, 3:]
    
# Vectorized count of occurrences of 1 per row
# This represents the L0 norm of the row for the value 1
ones_count = (subset == 1).sum(axis=1)

# Boolean indexing to retain only rows within the limit
filtered_df = train_df[ones_count <= args.path_lim].copy()

dropped_count = len(train_df) - len(filtered_df)
print(f"Dropped {dropped_count} rows exceeding path_limit of {args.path_lim}.")

test_df = pd.read_csv("./datasets/test.csv")
train_data = ImageDataset(args.train_dir, filtered_df)
test_data = ImageDataset(Path("./datasets/test"), test_df)

if train_data.num_classes() != test_data.num_classes():
    raise ValueError(F"Different number of classes! Train has {train_data.num_classes()} while test has {test_data.num_classes()}")

n_classes = train_data.num_classes()

if args.mix:
    other_df = pd.read_csv(args.mix_csv)
    other_data = ImageDataset(args.mix_dir, other_df)
    train_data = ImageDataset.merge(train_data, other_data, ratio=args.mix_ratio)

train_dl = DataLoader(
    train_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=os.cpu_count(), # Use available CPU cores
    pin_memory=True
)
test_dl = DataLoader(
    test_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=os.cpu_count(),
    pin_memory=True 
)

# Force model to be pre-downloaded
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Create and modify classifier
try:
    net =  timm.create_model(args.model, pretrained=True, num_classes=n_classes, cache_dir="./base_models")
    print(f"Found and loaded {args.model} from timm")
except Exception as e:
    print(e)
    raise ValueError(f"Could not parse model {args.model}, maybe check your spelling?")

net = convert_to_grayscale(net)

if args.freeze:
    for param in net.parameters():
        param.requires_grad = False

    for param in net.get_classifier().parameters():
        param.requires_grad = True

# Define loss fn and optim
loss_fn = torch.nn.BCEWithLogitsLoss()
optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=LEARNING_RATE)

model = Model(net, loss_fn, optim, n_classes, model_name=args.model)
if args.load_model:
    model.load(args.load_model)

# Training
total_epochs = args.epochs
save_freq = args.save_freq

for start_epoch in range(0, total_epochs, save_freq):
    current_run = min(save_freq, total_epochs - start_epoch)
    model.train(train_loader=train_dl, num_epochs=current_run)
    model.test(test_dl, test_file=args.log)
    model.save(args.save_dir)