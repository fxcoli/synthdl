import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import pandas as pd
from torchmetrics.classification import (
    MultilabelAccuracy, 
    MultilabelF1Score, 
    MultilabelPrecision, 
    MultilabelRecall
)
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, folder: Path|None, pathology_df: pd.DataFrame, size=(224, 224)):
        """
        Create an image dataset based on a directory of images

        Arguments:
            folder -- Single directory containing all images
            pathology_df -- DataFrame containing pathology information for 
            images, the first column should be the image name and following 
            columns should contain pathology info

        Keyword Arguments:
            size -- _description_ (default: {(256, 256)})
        """
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.Grayscale(),
            transforms.RandomRotation(180),
            transforms.RandomVerticalFlip(),
            transforms.RandomAutocontrast(),
            transforms.RandomResizedCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.449], [0.226])  # Mean values from torch docs
        ])

        # Ensure that the directory exists
        if folder is not None:
            folder = Path(folder)
            assert folder.is_dir(), f"Could not find directory: {folder}"
            
            # Create the dataframe and filter it
            files = list(Path(".").rglob("*.jpg"))
            assert len(files) > 0, f"Could not find any files in {folder}"
            
            # Get existing files
            id_col = pathology_df.columns[0]
            self.df = pathology_df.copy() # Now it should only contain relevant information
            self.df[id_col] = self.df[id_col].apply(lambda x: folder / x)
            assert len(self.df) > 0, f"No filenames in the DataFrame matched files found on disk"
        else:   # Otherwise we are merging
            self.df = pathology_df


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(row.iloc[0]).convert("L")
        return self.transform(image), torch.tensor(row[2:].to_list())
    
    def num_classes(self):
        """Returns the number of classes in the dataset."""
        return len(self.df.columns[2:])

    @classmethod
    def merge(cls, dataset1, dataset2, limit=None, ratio=0.5, random_state=42):
        """
        Merges the incoming datasets according to a given ratio

        Arguments:
            dataset1 -- First dataset (real)
            dataset2 -- Second dataset (synthetic)
            limit -- Size limit on final dataset

        Keyword Arguments:
            ratio -- How much should the second dataset comprise of the final dataset? (default: {0.5})
        """
        # Basic checks to ensure safe merge
        if not isinstance(dataset1, ImageDataset) or not isinstance(dataset2, ImageDataset):
            raise TypeError("Both arguments must be of type ImageDataset")
        assert dataset1.size == dataset2.size, "Both datasets must have equal transformed image sizes"

        if ratio == 0.0:
            return dataset1
        if ratio == 1.0:
            return dataset2

        # Get limit if not given
        if limit is None:
            limit = min(
                int(len(dataset1) / (1-ratio)),
                int(len(dataset2) / ratio)
            )

        # Get split sizes
        n1 = int(limit * (1-ratio))
        n2 = limit - n1

        # Random sample from each
        sample1 = dataset1.df.sample(n1, random_state=random_state) # arbitrary ass number for reproducibility later
        sample2 = dataset2.df.sample(n2, random_state=random_state)

        # Create new dataset
        return ImageDataset(folder=None, pathology_df=pd.concat((sample1, sample2), ignore_index=True), size=dataset1.size)


class Model:
    def __init__(self, model: torch.nn.Module, loss_fn, optimizer: torch.optim.Optimizer, 
                    num_classes: int, device: torch.device = None, model_name: str = None):
            self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            self.model = model.to(self.device)
            self.loss_fn = loss_fn
            self.optimizer = optimizer
            self.epochs = 0
            self.model_name = model_name or "model"

            self.use_amp = self.device.type == 'cuda'
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

            metrics_kwargs = {"num_labels": num_classes, "average": "macro"}
            
            # Initialize Training Metrics
            self.train_acc = MultilabelAccuracy(**metrics_kwargs).to(self.device)
            self.train_f1 = MultilabelF1Score(**metrics_kwargs).to(self.device)
            self.train_prec = MultilabelPrecision(**metrics_kwargs).to(self.device)
            self.train_recall = MultilabelRecall(**metrics_kwargs).to(self.device)
            
            # Initialize Testing Metrics
            self.test_acc = MultilabelAccuracy(**metrics_kwargs).to(self.device)
            self.test_f1 = MultilabelF1Score(**metrics_kwargs).to(self.device)
            self.test_prec = MultilabelPrecision(**metrics_kwargs).to(self.device)
            self.test_recall = MultilabelRecall(**metrics_kwargs).to(self.device)

    def train(self, train_loader: DataLoader, num_epochs: int):
        self.model.train()

        for _ in range(num_epochs):
            total_loss = 0.0
            # Reset all metrics at start of epoch
            self.train_acc.reset()
            self.train_f1.reset()
            self.train_prec.reset()
            self.train_recall.reset()

            pbar = tqdm(train_loader, desc=f"Epoch {self.epochs + 1}")
            
            for images, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).float()

                self.optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    preds = self.model(images)
                    loss = self.loss_fn(preds, labels)
                
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                with torch.no_grad():
                    probs = torch.sigmoid(preds)
                    labels_int = labels.long()
                    # Update all metrics
                    self.train_acc.update(probs, labels_int)
                    self.train_f1.update(probs, labels_int)
                    self.train_prec.update(probs, labels_int)
                    self.train_recall.update(probs, labels_int)
                    total_loss += loss.item()

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            self.epochs += 1
            epoch_loss = total_loss / len(train_loader)
            
            # Compute and print all metrics
            print(f"Epoch {self.epochs} | Loss: {epoch_loss:.4f} | "
                  f"Acc: {self.train_acc.compute():.4f} | Prec: {self.train_prec.compute():.4f} | "
                  f"Rec: {self.train_recall.compute():.4f} | F1: {self.train_f1.compute():.4f}")

    @torch.inference_mode()
    def test(self, test_loader: DataLoader, test_file: Path | None = None):
        self.model.eval()
        total_loss = 0.0
        self.test_acc.reset()
        self.test_f1.reset()
        self.test_prec.reset()
        self.test_recall.reset()

        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            for images, labels in tqdm(test_loader, desc="Testing"):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).float()

                preds = self.model(images)
                total_loss += self.loss_fn(preds, labels).item()

                probs = torch.sigmoid(preds)
                labels_int = labels.long()
                self.test_acc.update(probs, labels_int)
                self.test_f1.update(probs, labels_int)
                self.test_prec.update(probs, labels_int)
                self.test_recall.update(probs, labels_int)

        test_loss = total_loss / len(test_loader)
        metrics_str = (f"Epoch: {self.epochs}, Test Loss: {test_loss:.4f}, "
                       f"Acc: {self.test_acc.compute():.4f}, Prec: {self.test_prec.compute():.4f}, "
                       f"Rec: {self.test_recall.compute():.4f}, F1: {self.test_f1.compute():.4f}")
        print(metrics_str)

        if test_file:
            Path(test_file).parent.mkdir(parents=True, exist_ok=True)
            with open(test_file, 'a') as f:
                f.write(metrics_str + "\n")

    def save(self, location: str):
        save_path = Path(location) / f"{self.model_name}_epoch_{self.epochs}.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Include scaler state if resuming training is required
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'epochs': self.epochs
        }, save_path)

    def load(self, file: str):
        checkpoint = torch.load(file, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.use_amp and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.epochs = checkpoint.get('epochs', 0)

if __name__ == "__main__":

    def get_grayscale_conv(conv_layer: torch.nn.Conv2d):
        new_first = torch.nn.Conv2d(
            in_channels=1,  # Grayscale
            out_channels=conv_layer.out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            bias=(conv_layer.bias is not None)
        )

        with torch.no_grad():
            new_first.weight[:] = conv_layer.weight.mean(dim=1, keepdim=True)
            if conv_layer.bias is not None:
                new_first.bias[:] = conv_layer.bias

        return new_first

    train_df = pd.read_csv("datasets/train.csv")
    train = ImageDataset(
        folder=Path("datasets/train"),
        pathology_df=train_df
    )
    train_loader = DataLoader(train, batch_size=16, shuffle=True)

    test_df = pd.read_csv("datasets/test.csv")
    test = ImageDataset(
        folder=Path("./datasets/test"),
        pathology_df=test_df
    )
    test_loader = DataLoader(test, batch_size=16, shuffle=True)
    
    mixed = ImageDataset.merge(train, test, ratio=0.5)

    from torchvision.models import resnet18
    net = resnet18(num_classes=train.num_classes())
    net.conv1 = get_grayscale_conv(net.conv1)
    model = Model(
        model=net,
        loss_fn=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.Adam(params=net.parameters()),
        num_classes=train.num_classes()
    )

    model.train(train_loader, 1)
    model.test(test_loader)
