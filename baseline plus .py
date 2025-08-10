# baseline_enhanced.py

import numpy as np
import pandas as pd
import pathlib, sys, os, random, time
import numba, cv2, gc
import tqdm

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

import albumentations as A

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D

import torchvision
from torchvision import transforms as T


# --- Helper Functions (RLE, etc.) ---
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(512, 512)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    if not isinstance(mask_rle, str):
        return np.zeros(shape, dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


# --- NEW: Visualization and Metric Helper Functions ---
def dice_score(y_pred, y_true, smooth=1e-7):
    """Calculates the Dice score for a batch."""
    y_pred = y_pred.sigmoid()  # Apply sigmoid to get probabilities
    y_pred = (y_pred > 0.5).float()  # Threshold to get binary mask

    tp = (y_true * y_pred).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()

    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    return dice.item()


def get_boundary(mask, kernel_size=(3, 3)):
    """Extracts the boundary from a binary mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    return gradient


def unnormalize(tensor, mean, std):
    """Un-normalizes a tensor image for visualization."""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def plot_history(history):
    """Plots training and validation loss and Dice score."""
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(18, 6))

    # Plot Training and Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Validation Dice Score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_dice'], label='Validation Dice Score', color='orange')
    plt.title('Validation Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)

    plt.show()


def visualize_predictions(model, loader, device, num_examples=4):
    """Visualizes model predictions against ground truth."""
    model.eval()

    # Get a batch of data
    images, gt_masks = next(iter(loader))
    images = images.to(device)

    with torch.no_grad():
        pred_logits = model(images)['out']
        pred_masks = (pred_logits.sigmoid() > 0.5).cpu().numpy().astype(np.uint8)

    # Move data to CPU and un-normalize for plotting
    images = images.cpu()
    gt_masks = gt_masks.cpu().numpy().astype(np.uint8)

    # Define normalization constants for un-normalization
    mean = [0.625, 0.448, 0.688]
    std = [0.131, 0.177, 0.101]

    plt.figure(figsize=(20, 5 * num_examples))

    for i in range(num_examples):
        # Un-normalize and prepare image
        img_display = unnormalize(images[i].clone(), mean, std)
        img_display = img_display.permute(1, 2, 0).numpy()
        img_display = np.clip(img_display, 0, 1)  # Clip values to be in [0, 1] range

        # Get masks and boundaries
        gt_mask_single = gt_masks[i, 0]
        pred_mask_single = pred_masks[i, 0]

        gt_boundary = get_boundary(gt_mask_single)
        pred_boundary = get_boundary(pred_mask_single)

        # Calculate Dice score for this specific image
        # Note: we need to re-create tensors for the dice_score function
        dice = dice_score(pred_logits[i].unsqueeze(0).cpu(), torch.from_numpy(gt_masks[i]).unsqueeze(0).cpu())

        # Plotting
        # Row for Ground Truth
        plt.subplot(num_examples, 4, i * 4 + 1)
        plt.imshow(img_display)
        plt.title(f"Original Image {i + 1}")
        plt.axis('off')

        plt.subplot(num_examples, 4, i * 4 + 2)
        plt.imshow(gt_mask_single, cmap='gray')
        plt.title("GT Mask")
        plt.axis('off')

        plt.subplot(num_examples, 4, i * 4 + 3)
        plt.imshow(gt_boundary, cmap='gray')
        plt.title("GT Boundary")
        plt.axis('off')

        # Empty plot for alignment
        plt.subplot(num_examples, 4, i * 4 + 4)
        plt.axis('off')

        # Row for Prediction (we'll just plot them below the originals for a 2x4 layout if num_examples=2)
        # But a long strip is easier to code. I'll modify title to show it's a prediction.

        # We can create a second row for predictions. Let's adjust the subplot grid.
        # Let's create an 8-plot grid for each example instead.
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f"Example {i + 1}", fontsize=16)

        # Ground Truth Row
        axes[0, 0].imshow(img_display)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(gt_mask_single, cmap='gray')
        axes[0, 1].set_title("GT Mask")
        axes[0, 1].axis('off')

        axes[0, 2].imshow(gt_boundary, cmap='gray')
        axes[0, 2].set_title("GT Boundary")
        axes[0, 2].axis('off')

        axes[0, 3].axis('off')  # Placeholder

        # Prediction Row
        axes[1, 0].imshow(img_display)
        axes[1, 0].set_title("Original Image")
        axes[1, 0].axis('off')

        axes[1, 1].imshow(pred_mask_single, cmap='gray')
        axes[1, 1].set_title(f"Pred Mask (Dice: {dice:.4f})")
        axes[1, 1].axis('off')

        axes[1, 2].imshow(pred_boundary, cmap='gray')
        axes[1, 2].set_title("Pred Boundary")
        axes[1, 2].axis('off')

        axes[1, 3].axis('off')  # Placeholder

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


# --- Dataset and Model Classes (largely unchanged) ---
class TianChiDataset(D.Dataset):
    def __init__(self, paths, rles, transform, test_mode=False):
        self.paths = paths
        self.rles = rles
        self.transform = transform
        self.test_mode = test_mode
        self.len = len(paths)

        # The normalization constants
        self.norm_mean = [0.625, 0.448, 0.688]
        self.norm_std = [0.131, 0.177, 0.101]

        self.as_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(self.norm_mean, self.norm_std),
        ])

    def __getitem__(self, index):
        img = cv2.imread(self.paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        if not self.test_mode:
            mask = rle_decode(self.rles[index])
            # Albumentations expects 'image' and 'mask' keys
            augments = self.transform(image=img, mask=mask)
            # The output of as_tensor is already resized by the transform
            return self.as_tensor(augments['image']), torch.from_numpy(augments['mask'][None]).float()
        else:
            # For test mode, we still need to resize
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            return self.as_tensor(img), ''

    def __len__(self):
        return self.len


def get_model():
    model = torchvision.models.segmentation.fcn_resnet50(True)
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
    return model


# --- Loss Function ---
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return 1 - dc


def loss_fn(y_pred, y_true):
    bce_fn = nn.BCEWithLogitsLoss()
    dice_fn = SoftDiceLoss()
    bce = bce_fn(y_pred, y_true)
    dice = dice_fn(y_pred.sigmoid(), y_true)
    return 0.8 * bce + 0.2 * dice


# --- MODIFIED: Validation Function ---
@torch.no_grad()
def validation(model, loader, device):
    model.eval()
    losses = []
    dice_scores = []

    for image, target in loader:
        image, target = image.to(device), target.to(device)
        output = model(image)['out']

        # Calculate loss
        loss = loss_fn(output, target)
        losses.append(loss.item())

        # Calculate Dice score
        score = dice_score(output, target)
        dice_scores.append(score)

    return np.array(losses).mean(), np.array(dice_scores).mean()


# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Configuration ---
    EPOCHES = 20
    BATCH_SIZE = 16  # Reduced batch size to prevent OOM with larger models/images
    IMAGE_SIZE = 256
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Data Preparation ---
    trfm = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(),
    ])

    train_mask = pd.read_csv('./data/train_mask.csv', sep='\t', names=['name', 'mask'])
    train_mask['name'] = train_mask['name'].apply(lambda x: './data/train/' + x)

    # Use a more robust train/validation split
    from sklearn.model_selection import train_test_split

    ids = train_mask.index.values
    train_ids, valid_ids = train_test_split(ids, test_size=0.15, random_state=42)

    dataset = TianChiDataset(
        train_mask['name'].values,
        train_mask['mask'].fillna('').values,
        trfm, False
    )

    train_ds = D.Subset(dataset, train_ids)
    valid_ds = D.Subset(dataset, valid_ids)

    loader = D.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    vloader = D.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Train samples: {len(train_ds)}, Validation samples: {len(valid_ds)}")

    # --- Model and Optimizer ---
    model = get_model()
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    # --- MODIFIED: Training Loop ---
    header = r'''
        Train |         Validation
Epoch |  Loss |    Loss    | Dice Score | Time, m
'''
    raw_line = '{:6d}' + '\u2502{:7.3f}' * 3 + '\u2502{:6.2f}'
    print(header)

    best_loss = 1e9
    best_dice = 0

    # History for plotting
    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}

    for epoch in range(1, EPOCHES + 1):
        model.train()
        epoch_losses = []
        start_time = time.time()

        for image, target in tqdm.tqdm(loader, desc=f"Epoch {epoch}/{EPOCHES}"):
            image, target = image.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(image)['out']
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # Validation
        vloss, vdice = validation(model, vloader, DEVICE)

        # Store history
        history['train_loss'].append(np.mean(epoch_losses))
        history['val_loss'].append(vloss)
        history['val_dice'].append(vdice)

        print(raw_line.format(epoch, np.mean(epoch_losses), vloss, vdice, (time.time() - start_time) / 60))

        # Save model based on best validation Dice score
        if vdice > best_dice:
            best_dice = vdice
            print(f"🎉 New best Dice score: {best_dice:.4f}. Saving model...")
            torch.save(model.state_dict(), 'model_best_dice.pth')

    # --- Post-Training Visualization ---
    print("\nTraining finished. Plotting history...")
    plot_history(history)

    print("\nVisualizing predictions on validation set...")
    # Load the best model for visualization
    model.load_state_dict(torch.load('model_best_dice.pth'))
    # Visualize 2 examples from the validation loader
    visualize_predictions(model, vloader, DEVICE, num_examples=2)

    # --- Submission Generation (unchanged logic) ---
    print("\nGenerating submission file...")
    subm = []
    model.eval()

    test_mask = pd.read_csv('./data/test_a_samplesubmit.csv', sep='\t', names=['name', 'mask'])
    test_mask['name'] = test_mask['name'].apply(lambda x: './data/test_a/' + x)

    # Define test transform (only resize and normalize)
    test_trfm = A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE)])
    as_tensor_test = T.Compose([
        T.ToTensor(),
        T.Normalize([0.625, 0.448, 0.688], [0.131, 0.177, 0.101]),
    ])

    for idx, name in enumerate(tqdm.tqdm(test_mask['name'].iloc[:])):
        image = cv2.imread(name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = test_trfm(image=image)['image']

        with torch.no_grad():
            image_tensor = as_tensor_test(image).to(DEVICE)[None]
            score = model(image_tensor)['out'][0][0]
            score_sigmoid = score.sigmoid().cpu().numpy()
            # Thresholding and resizing
            binary_mask = (score_sigmoid > 0.5).astype(np.uint8)
            final_mask = cv2.resize(binary_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        subm.append([name.split('/')[-1], rle_encode(final_mask)])

    subm_df = pd.DataFrame(subm, columns=['name', 'mask'])
    subm_df.to_csv('./submission.csv', index=None, header=None, sep='\t')
    print("Submission file created: submission.csv")