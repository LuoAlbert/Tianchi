import numpy as np
import pandas as pd
import pathlib, sys, os, random, time
import numba, cv2, gc
import tqdm

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import ttach as tta

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D

import torchvision
from torchvision import transforms as T

# 新增：用于生成边界和接触点图
from scipy.ndimage import sobel, distance_transform_edt
from skimage.morphology import disk, dilation

# --- 可配置的超参数 ---
DATA_DIR = './data/'
EPOCHES = 60  # 增加总训练轮数以适应更复杂的任务和学习率重启
BATCH_SIZE = 8 # 增大了模型和数据，需要减小Batch Size
IMAGE_SIZE = 512  # 策略2：增大图像尺寸以保留更多细节
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# =====================================================================================
# 函数和类的定义部分
# =====================================================================================
def rle_encode(im):
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(512, 512)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


# =======================
# 策略1: 后处理函数
# =======================
def remove_small_objects(mask, min_size=100):
    """移除二值掩码中的小面积连通域"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    # stats的第一列是x，第二列是y，...，最后一列是面积
    # 第0个标签是背景，所以从1开始
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            mask[labels == i] = 0
    return mask


def get_transforms(data_type='train'):
    mean = [0.625, 0.448, 0.688]
    std = [0.131, 0.177, 0.101]
    if data_type == 'train':
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.ShiftScaleRotate(p=0.3, shift_limit=0.1, scale_limit=0.1, rotate_limit=20),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    elif data_type == 'valid' or data_type == 'test':
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


# ==============================================================
# 策略3 & 4: 改造Dataset以支持多任务学习 (建筑物/边界/接触点)
# ==============================================================
class TianChiDataset(D.Dataset):
    def __init__(self, paths, rles, transform, test_mode=False):
        self.paths = paths
        self.rles = rles
        self.transform = transform
        self.test_mode = test_mode
        self.mean = np.array([0.625, 0.448, 0.688])
        self.std = np.array([0.131, 0.177, 0.101])

    def __getitem__(self, index):
        img = cv2.imread(self.paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if not self.test_mode:
            mask = rle_decode(self.rles[index])
            # 1. 边界图 (Boundary map)
            boundary = sobel(mask)
            boundary = dilation(boundary, disk(1)).astype(np.float32)
            boundary = np.clip(boundary, 0, 1)
            # 2. 接触点图 (Contact map)
            # 使用距离变换找到物体间的接触点
            dist_map = distance_transform_edt(mask)
            dist_map = (dist_map > 0) & (dist_map <= 2)  # 距离边缘1-2个像素
            contact = dist_map.astype(np.float32)

            # 将三个mask堆叠起来
            masks = np.stack([mask, boundary, contact], axis=-1)

            augmented = self.transform(image=img, mask=masks)
            img = augmented['image']
            # permute从 (H,W,C) -> (C,H,W)
            masks = augmented['mask'].permute(2, 0, 1)
            return img, masks
        else:
            augmented = self.transform(image=img)
            img = augmented['image']
            return img, ''

    def __len__(self):
        return len(self.paths)

    def denormalize(self, tensor_image):
        img = tensor_image.permute(1, 2, 0).cpu().numpy()
        img = (img * self.std + self.mean)
        return np.clip(img, 0, 1)


def get_model():
    # 模型输出3个通道，分别预测 建筑物/边界/接触点
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3,  # <--- 核心修改
    )
    return model


# 损失函数类保持不变
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)): super(SoftDiceLoss,
                                                        self).__init__(); self.smooth = smooth; self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims);
        fp = (x * (1 - y)).sum(self.dims);
        fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth);
        dc = dc.mean();
        return 1 - dc


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'): super(FocalLoss,
                                                                     self).__init__(); self.alpha = alpha; self.gamma = gamma; self.reduction = reduction; self.bce_loss = nn.BCEWithLogitsLoss(
        reduction='none')

    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets);
        probas = torch.sigmoid(logits)
        loss = self.alpha * (1 - probas) ** self.gamma * bce if self.alpha >= 0 else (probas) ** self.gamma * bce
        if self.reduction == 'mean': return loss.mean()
        return loss



def visualize_multitask_predictions(model, dataset, device, best_threshold, num_samples=2):
    """
    可视化多任务模型的预测结果，生成类似示例图的 2x4 布局。
    Args:
        model (nn.Module): 训练好的模型。
        dataset (Dataset): 用于采样的验证集 (valid_ds)。
        device (str): 'cuda' or 'cpu'。
        best_threshold (float): 用于建筑物掩码的最佳阈值。
        num_samples (int): 要可视化的随机样本数量。
    """
    model.eval()

    for i in range(num_samples):
        # 从数据集中随机选择一个样本
        sample_idx = random.randint(0, len(dataset) - 1)
        img_tensor, mask_tensors = dataset[sample_idx]

        with torch.no_grad():
            # 为模型输入增加一个 batch 维度，然后移除输出的 batch 维度
            pred_tensors = model(img_tensor.to(device).unsqueeze(0)).squeeze(0).cpu()

        # 反归一化图像以正确显示
        img_display = dataset.denormalize(img_tensor)

        # --- 准备真实标签 (Ground Truth) ---
        true_mask = mask_tensors[0].numpy()
        true_boundary = mask_tensors[1].numpy()
        true_contact = mask_tensors[2].numpy()

        # --- 准备预测结果 ---
        # 对建筑物掩码应用最佳阈值
        pred_mask_prob = pred_tensors[0].sigmoid()
        pred_mask = (pred_mask_prob > best_threshold).numpy()

        # 对边界和接触点使用默认的0.5阈值
        pred_boundary = (pred_tensors[1].sigmoid() > 0.5).numpy()
        pred_contact = (pred_tensors[2].sigmoid() > 0.5).numpy()

        # 为当前样本计算 Dice 分数，用于显示在标题中
        dice_val = np_dice_score(pred_mask_prob.numpy(), true_mask, best_threshold)

        # --- 开始绘图 ---
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        fig.suptitle(f"Example {i + 1} (Sample ID: {sample_idx})", fontsize=20)

        # --- 第一行: 真实标签 (Ground Truth) ---
        axes[0, 0].imshow(img_display)
        axes[0, 0].set_title("Original Image")

        axes[0, 1].imshow(true_mask, cmap='gray')
        axes[0, 1].set_title("GT Mask")

        axes[0, 2].imshow(true_boundary, cmap='gray')
        axes[0, 2].set_title("GT Boundary")

        axes[0, 3].imshow(true_contact, cmap='gray')
        axes[0, 3].set_title("GT Contact")

        # --- 第二行: 预测结果 (Prediction) ---
        axes[1, 0].imshow(img_display)
        axes[1, 0].set_title("Original Image")

        axes[1, 1].imshow(pred_mask, cmap='gray')
        axes[1, 1].set_title(f"Pred Mask (Dice: {dice_val:.4f})")

        axes[1, 2].imshow(pred_boundary, cmap='gray')
        axes[1, 2].set_title("Pred Boundary")

        axes[1, 3].imshow(pred_contact, cmap='gray')
        axes[1, 3].set_title("Pred Contact")

        # 统一关闭所有子图的坐标轴
        for ax in axes.flat:
            ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局以适应主标题
        plt.show()


# ==================================
# 改造损失函数以适应多任务
# ==================================
def loss_fn(y_pred, y_true):
    # y_pred and y_true are (N, 3, H, W)

    # 任务1: 建筑物掩码 (Focal + Dice)
    pred_mask, true_mask = y_pred[:, 0, ...], y_true[:, 0, ...].float()
    loss_mask = 0.7 * FocalLoss()(pred_mask, true_mask) + 0.3 * SoftDiceLoss()(pred_mask.sigmoid(), true_mask)

    # 任务2: 边界掩码 (BCE or Focal)
    pred_boundary, true_boundary = y_pred[:, 1, ...], y_true[:, 1, ...].float()
    loss_boundary = nn.BCEWithLogitsLoss()(pred_boundary, true_boundary)

    # 任务3: 接触点掩码 (BCE or Focal)
    pred_contact, true_contact = y_pred[:, 2, ...], y_true[:, 2, ...].float()
    loss_contact = nn.BCEWithLogitsLoss()(pred_contact, true_contact)

    # 加权组合总损失
    return loss_mask + 0.5 * loss_boundary + 0.5 * loss_contact


def np_dice_score(probability, mask, threshold=0.5):
    p = probability.reshape(-1) > threshold
    t = mask.reshape(-1) > 0.5
    uion = p.sum() + t.sum()
    overlap = (p * t).sum()
    return 2 * overlap / (uion + 1e-6)


@torch.no_grad()
def validation(model, loader, loss_fn):
    losses, dice_scores = [], []
    model.eval()
    for image, target in loader:
        image, target = image.to(DEVICE), target.to(DEVICE)
        output = model(image)
        loss = loss_fn(output, target)
        losses.append(loss.item())

        # Dice分数只基于第一个通道（建筑物掩码）计算
        output_sigmoid_mask = output[:, 0, ...].sigmoid().cpu().numpy()
        target_np_mask = target[:, 0, ...].cpu().numpy()
        dice_score = np_dice_score(output_sigmoid_mask, target_np_mask, threshold=0.5)
        dice_scores.append(dice_score)

    return np.array(losses).mean(), np.array(dice_scores).mean()


# =====================================================================================
# 主执行代码块
# =====================================================================================
if __name__ == '__main__':
    train_mask = pd.read_csv(os.path.join(DATA_DIR, 'train_mask.csv'), sep='\t', names=['name', 'mask'])
    train_mask['name'] = train_mask['name'].apply(lambda x: os.path.join(DATA_DIR, 'train', x))
    train_names, valid_names, train_rles, valid_rles = train_test_split(
        train_mask['name'].values, train_mask['mask'].fillna('').values, test_size=0.2, random_state=42)

    train_ds = TianChiDataset(train_names, train_rles, get_transforms('train'), False)
    valid_ds = TianChiDataset(valid_names, valid_rles, get_transforms('valid'), False)
    num_workers = 4 if sys.platform != 'win32' else 0
    loader = D.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    vloader = D.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Train samples: {len(train_ds)}, Validation samples: {len(valid_ds)}")

    model = get_model()
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    # 策略: 使用带重启的学习率调度器和早停
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=16, T_mult=2, eta_min=1e-7)
    patience = 6  # 早停的耐心值
    epochs_no_improve = 0

    header = r'''
            Train | Valid
    Epoch |  Loss |  Loss | Dice |   LR    | Time, m
    '''
    raw_line = '{:6d}' + '\u2502{:7.3f}' * 2 + '\u2502{:6.3f}' + '\u2502{:9.2e}' + '\u2502{:6.2f}'
    print(header)
    best_dice = 0.0
    #绘loss图
    history = {'train_loss': [], 'valid_loss': [], 'valid_dice': []}

    for epoch in range(1, EPOCHES + 1):
        losses = []
        start_time = time.time()
        model.train()
        for image, target in tqdm.tqdm(loader, desc=f"Epoch {epoch}/{EPOCHES}"):
            image, target = image.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(image)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        train_loss = np.array(losses).mean()
        vloss, vdice = validation(model, vloader, loss_fn)
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(vloss)
        history['valid_dice'].append(vdice)
        scheduler.step()
        print(raw_line.format(epoch, train_loss, vloss, vdice, optimizer.param_groups[0]['lr'],
                              (time.time() - start_time) / 60))

        if vdice > best_dice:
            best_dice = vdice
            torch.save(model.state_dict(), 'model_best_multitask.pth')
            print(f"  >> New best model saved with validation Dice: {best_dice:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
            break

    # <-- 新增: 绘制训练和验证损失曲线 -->
    plt.figure(figsize=(20, 8))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['valid_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Dice分数曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['valid_dice'], label='Validation Dice Score', color='orange')
    plt.title('Validation Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)

    plt.show()

    # --- 推理部分 ---
    print("\nLoading best model for inference...")
    model.load_state_dict(torch.load('model_best_multitask.pth'))

    # 阈值搜索（逻辑不变，但很重要）
    print("Finding the best threshold on the validation set...")

    # =========================================================================
    # 核心修改：为这个特定环节创建一个超低 BATCH_SIZE 的 DataLoader
    # =========================================================================
    vloader_for_threshold_finding = D.DataLoader(valid_ds,
                                                 batch_size=4,  # <-- 关键！使用一个极小的值，比如 2 或 4
                                                 shuffle=False,
                                                 num_workers=num_workers,
                                                 pin_memory=True)  # pin_memory 在这里也可以设为 False

    all_outputs, all_targets_mask = [], []
    with torch.no_grad():
        for image, target in tqdm.tqdm(vloader_for_threshold_finding, desc="Predicting on validation set"):
            image = image.to(DEVICE)
            output = model(image)[:, 0, ...].sigmoid()  # 只取第一个通道
            all_outputs.append(output.cpu().numpy())
            all_targets_mask.append(target[:, 0, ...].cpu().numpy())
    all_outputs = np.concatenate(all_outputs)
    all_targets_mask = np.concatenate(all_targets_mask)
    thresholds = np.linspace(0.3, 0.7, 41)
    dice_scores = [np_dice_score(all_outputs, all_targets_mask, thr) for thr in thresholds]
    best_threshold = thresholds[np.argmax(dice_scores)]
    print(f"Best threshold found: {best_threshold:.4f}")

    # 使用TTA和后处理进行最终预测
    print(f"\nStarting final inference with TTA, best threshold, and post-processing...")
    base_model = get_model()
    base_model.load_state_dict(torch.load('model_best_multitask.pth'))
    tta_transforms = tta.Compose([tta.HorizontalFlip()])
    tta_model = tta.SegmentationTTAWrapper(base_model, tta_transforms, merge_mode='mean')
    tta_model.to(DEVICE);
    tta_model.eval()

    test_mask = pd.read_csv(os.path.join(DATA_DIR, 'test_a_samplesubmit.csv'), sep='\t', names=['name', 'mask'])
    test_mask['name'] = test_mask['name'].apply(lambda x: os.path.join(DATA_DIR, 'test_a', x))
    test_transform = get_transforms('test')

    subm = []
    for idx, name in enumerate(tqdm.tqdm(test_mask['name'].iloc[:], desc="Predicting")):
        image = cv2.imread(name);
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = test_transform(image=image);
        image = transformed['image']

        with torch.no_grad():
            image = image.to(DEVICE).unsqueeze(0)
            output = tta_model(image)
            score_sigmoid = output[:, 0, ...].squeeze().cpu().numpy()  # 只取建筑物通道

            pred_mask = (score_sigmoid > best_threshold).astype(np.uint8)

            # 策略1: 应用后处理，移除小的噪声点
            pred_mask = remove_small_objects(pred_mask, min_size=150)  # 阈值可以调整

            pred_mask_resized = cv2.resize(pred_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        subm.append([os.path.basename(name), rle_encode(pred_mask_resized)])

    subm_df = pd.DataFrame(subm)
    subm_df.to_csv('./submission_advanced.csv', index=None, header=None, sep='\t')
    print("Advanced submission file created: submission_advanced.csv")


    # # --- 可视化部分 ---
    # print("\nVisualizing one multi-task sample from validation set...")
    # sample_idx = random.randint(0, len(valid_ds) - 1)
    # img_tensor, mask_tensors = valid_ds[sample_idx]
    #
    # with torch.no_grad():
    #     pred_tensors = tta_model(img_tensor.to(DEVICE).unsqueeze(0)).squeeze().cpu()
    #
    # img_display = valid_ds.denormalize(img_tensor)
    #
    # # 分离真实标签
    # true_mask = mask_tensors[0].numpy()
    # true_boundary = mask_tensors[1].numpy()
    # true_contact = mask_tensors[2].numpy()
    #
    # # 分离预测结果
    # pred_mask = (pred_tensors[0].sigmoid() > best_threshold).numpy()
    # pred_boundary = (pred_tensors[1].sigmoid() > 0.5).numpy()
    # pred_contact = (pred_tensors[2].sigmoid() > 0.5).numpy()
    #
    # plt.figure(figsize=(24, 12))
    # plt.subplot(2, 4, 1);
    # plt.imshow(img_display);
    # plt.title("Original Image");
    # plt.axis('off')
    # plt.subplot(2, 4, 2);
    # plt.imshow(true_mask, cmap='gray');
    # plt.title("GT Mask");
    # plt.axis('off')
    # plt.subplot(2, 4, 3);
    # plt.imshow(true_boundary, cmap='gray');
    # plt.title("GT Boundary");
    # plt.axis('off')
    # plt.subplot(2, 4, 4);
    # plt.imshow(true_contact, cmap='gray');
    # plt.title("GT Contact");
    # plt.axis('off')
    #
    # plt.subplot(2, 4, 5);
    # plt.imshow(img_display);
    # plt.title("Original Image");
    # plt.axis('off')  # 重复显示以便对比
    # plt.subplot(2, 4, 6);
    # plt.imshow(pred_mask, cmap='gray');
    # plt.title(f"Pred Mask (Dice: {np_dice_score(pred_tensors[0].numpy(), true_mask, best_threshold):.4f})");
    # plt.axis('off')
    # plt.subplot(2, 4, 7);
    # plt.imshow(pred_boundary, cmap='gray');
    # plt.title("Pred Boundary");
    # plt.axis('off')
    # plt.subplot(2, 4, 8);
    # plt.imshow(pred_contact, cmap='gray');
    # plt.title("Pred Contact");
    # plt.axis('off')
    #
    # plt.tight_layout()
    # plt.show()

    ##对可视化进行集成：
    print("\nVisualizing random samples from validation set with enhanced layout...")
    visualize_multitask_predictions(tta_model, valid_ds, DEVICE, best_threshold, num_samples=5)