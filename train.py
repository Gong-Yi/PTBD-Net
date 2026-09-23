# -*- coding: utf-8 -*-
"""PTBD-Net Training Script.

Usage:
  python train.py --dataset_names NUAA-SIRST --batchSize 16
  python train.py --dataset_names IRSTD-1K NUDT-SIRST NUAA-SIRST --batchSize 16
"""

import argparse, time, os, random
import numpy as np
import torch, torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from scipy import ndimage

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from model.config import get_config
from model.ptbd_net import PTBDNet

# ========================================================================
#  Utilities
# ========================================================================

def seed_pytorch(seed=42):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def router_temperature(epoch, total_epochs):
    if epoch < 100: return 1.5 - 0.3 * epoch / 100
    if epoch < 500: return 1.2 - 0.2 * (epoch - 100) / 400
    return 1.0 - 0.2 * (epoch - 500) / max(1, total_epochs - 500)

# ========================================================================
#  Dataset
# ========================================================================

IMG_NORM = {
    'NUAA-SIRST': {'mean': 101.06385040283203, 'std': 34.619606018066406},
    'NUDT-SIRST': {'mean': 107.80905151367188, 'std': 33.02274703979492},
    'IRSTD-1K':   {'mean': 87.4661865234375,   'std': 39.71953201293945},
}

def get_norm(dataset_name, dataset_dir):
    if dataset_name in IMG_NORM: return IMG_NORM[dataset_name]
    with open(f'{dataset_dir}/{dataset_name}/img_idx/train_{dataset_name}.txt') as f:
        train_list = f.read().splitlines()
    with open(f'{dataset_dir}/{dataset_name}/img_idx/test_{dataset_name}.txt') as f:
        test_list = f.read().splitlines()
    mean_list, std_list = [], []
    for pth in train_list + test_list:
        try: img = Image.open(f'{dataset_dir}/{dataset_name}/images/{pth}.png').convert('I')
        except: img = Image.open(f'{dataset_dir}/{dataset_name}/images/{pth}.bmp').convert('I')
        arr = np.array(img, dtype=np.float32)
        mean_list.append(arr.mean()); std_list.append(arr.std())
    return {'mean': float(np.mean(mean_list)), 'std': float(np.mean(std_list))}

def normalize(img, cfg): return (img - cfg['mean']) / cfg['std']

def random_crop(img, mask, patch_size, pos_prob=0.5):
    h, w = img.shape
    if min(h, w) < patch_size:
        img = np.pad(img, ((0, max(h, patch_size)-h), (0, max(w, patch_size)-w)), mode='constant')
        mask = np.pad(mask, ((0, max(h, patch_size)-h), (0, max(w, patch_size)-w)), mode='constant')
        h, w = img.shape
    while True:
        hs, ws = random.randint(0, h-patch_size), random.randint(0, w-patch_size)
        ip, mp = img[hs:hs+patch_size, ws:ws+patch_size], mask[hs:hs+patch_size, ws:ws+patch_size]
        if pos_prob is None or random.random() > pos_prob or mp.sum() > 0: break
    return ip, mp

class Augmentation:
    def __call__(self, img, mask):
        if random.random() < 0.5: img, mask = img[::-1,:], mask[::-1,:]
        if random.random() < 0.5: img, mask = img[:,::-1], mask[:,::-1]
        if random.random() < 0.5: img, mask = img.transpose(1,0), mask.transpose(1,0)
        return img, mask

class TrainSet(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None):
        self.dataset_dir = f'{dataset_dir}/{dataset_name}'; self.patch_size = patch_size
        with open(f'{self.dataset_dir}/img_idx/train_{dataset_name}.txt') as f:
            self.train_list = f.read().splitlines()
        self.norm = img_norm_cfg or get_norm(dataset_name, dataset_dir)
        self.aug = Augmentation()
    def __getitem__(self, idx):
        stem = self.train_list[idx]
        try: img=Image.open(f'{self.dataset_dir}/images/{stem}.png').convert('I'); mask=Image.open(f'{self.dataset_dir}/masks/{stem}.png')
        except: img=Image.open(f'{self.dataset_dir}/images/{stem}.bmp').convert('I'); mask=Image.open(f'{self.dataset_dir}/masks/{stem}.bmp')
        img = normalize(np.array(img, dtype=np.float32), self.norm)
        mask = np.array(mask, dtype=np.float32)/255.0
        if len(mask.shape)>2: mask=mask[:,:,0]
        ip, mp = random_crop(img, mask, self.patch_size, pos_prob=0.5)
        ip, mp = self.aug(ip, mp)
        return torch.from_numpy(np.ascontiguousarray(ip[np.newaxis])), torch.from_numpy(np.ascontiguousarray(mp[np.newaxis]))
    def __len__(self): return len(self.train_list)

def pad_img(img, times=32):
    h,w=img.shape
    if h%times: img=np.pad(img,((0,(h//times+1)*times-h),(0,0)),mode='constant')
    if w%times: img=np.pad(img,((0,0),(0,(w//times+1)*times-w)),mode='constant')
    return img

class TestSet(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, train_ds, test_ds, img_norm_cfg=None):
        self.dataset_dir = f'{dataset_dir}/{test_ds}'
        with open(f'{self.dataset_dir}/img_idx/test_{test_ds}.txt') as f: self.test_list=f.read().splitlines()
        self.norm = img_norm_cfg or get_norm(train_ds, dataset_dir)
    def __getitem__(self, idx):
        stem = self.test_list[idx]
        try: img=Image.open(f'{self.dataset_dir}/images/{stem}.png').convert('I'); msk=Image.open(f'{self.dataset_dir}/masks/{stem}.png')
        except: img=Image.open(f'{self.dataset_dir}/images/{stem}.bmp').convert('I'); msk=Image.open(f'{self.dataset_dir}/masks/{stem}.bmp')
        img = normalize(np.array(img, dtype=np.float32), self.norm)
        msk = np.array(msk, dtype=np.float32)/255.0
        if len(msk.shape)>2: msk=msk[:,:,0]
        h,w=img.shape; img=pad_img(img); msk=pad_img(msk)
        return torch.from_numpy(np.ascontiguousarray(img[np.newaxis])), torch.from_numpy(np.ascontiguousarray(msk[np.newaxis])), [h,w], stem
    def __len__(self): return len(self.test_list)

# ========================================================================
#  Metrics
# ========================================================================

class mIoU:
    def __init__(self): self.inter, self.union = 0, 0
    def update(self, pred, gt):
        pred, gt = pred.bool(), gt.bool()
        self.inter += (pred & gt).sum().item(); self.union += (pred | gt).sum().item()
    def get(self):
        iou = self.inter / max(self.union, 1)
        return (iou, np.float64(iou))

class PD_FA:
    def __init__(self): self.tp, self.fp, self.fn = 0, 0, 0
    def update(self, pred, gt, size):
        h, w = size[0].item(), size[1].item()
        pred_np, gt_np = pred.numpy()[:h,:w], gt.numpy()[:h,:w]
        label, num = ndimage.label(gt_np)
        for i in range(1, num+1):
            if pred_np[label==i].max() > 0.5: self.tp += 1
            else: self.fn += 1
        label_p, num_p = ndimage.label(pred_np > 0.5)
        for i in range(1, num_p+1):
            if gt_np[label_p==i].max() == 0: self.fp += 1
    def get(self):
        pd = self.tp / max(self.tp+self.fn, 1)
        fa = self.fp / max(self.fp+self.tp, 1) / (pred_np.size if 'pred_np' in dir() else 1)
        return (pd, fa)

# ========================================================================
#  Weight init
# ========================================================================

def weights_init_kaiming(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)): init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.BatchNorm2d): init.normal_(m.weight.data, 1.0, 0.02); init.constant_(m.bias.data, 0.0)

# ========================================================================
#  Model wrapper
# ========================================================================

class Net(nn.Module):
    def __init__(self, config, mode='train'):
        super().__init__()
        self.criterion = nn.BCELoss(size_average=True)
        self.model = PTBDNet(config, mode=mode, deepsuper=True, use_dhif=True)
    def forward(self, img): return self.model(img)
    def loss(self, preds, gt_mask):
        if isinstance(preds, tuple): main_loss = sum(self.criterion(p, gt_mask) for p in preds)
        else: main_loss = self.criterion(preds, gt_mask)
        if self.model.training:
            aux = self.model.get_moe_aux_loss()
            if aux is not None: main_loss = main_loss + aux
        return main_loss

# ========================================================================
#  Training
# ========================================================================

def train(args, config):
    train_set = TrainSet(args.dataset_dir, args.dataset_name, args.patchSize)
    train_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batchSize, shuffle=True)
    net = Net(config, mode='train').cuda(); net.apply(weights_init_kaiming); net.train()
    epoch_state, total_loss_list, total_loss_epoch = 0, [], []
    os.makedirs(args.log_dir, exist_ok=True); writer = SummaryWriter(args.log_dir)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for idx_epoch in range(epoch_state, args.epochs):
        t_val = router_temperature(idx_epoch, args.epochs)
        net.model.set_router_temperature(t_val)
        if idx_epoch < 10:
            for pg in optimizer.param_groups: pg['lr'] = 0.001 * (idx_epoch+1)/10
        net.train(); results1, results2 = (0,0), (0,0)
        for img, gt_mask in train_loader:
            img, gt_mask = Variable(img).cuda(), Variable(gt_mask).cuda()
            if img.shape[0] == 1: continue
            preds = net.forward(img); loss = net.loss(preds, gt_mask)
            total_loss_epoch.append(loss.detach().cpu())
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        if idx_epoch >= 10:
            if idx_epoch == 10:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs-10, eta_min=1e-5)
            scheduler.step()
        if (idx_epoch+1) % args.every_print == 0:
            avg = float(np.array(total_loss_epoch).mean()); total_loss_list.append(avg); total_loss_epoch = []
            print(f'{time.ctime()[4:-5]}  Epoch {idx_epoch+1:4d}  loss={avg:.6f}  lr={optimizer.param_groups[0]["lr"]:.6f}')
            writer.add_scalar('loss', avg, idx_epoch+1)
        if (idx_epoch+1) >= args.begin_test and (idx_epoch+1) % args.every_test == 0:
            test_set = TestSet(args.dataset_dir, args.dataset_name, args.dataset_name)
            test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
            net.eval(); eval_mIoU, eval_PD_FA = mIoU(), PD_FA(); test_loss = []
            with torch.no_grad():
                for img_t, gt_t, size, _ in test_loader:
                    img_t = Variable(img_t).cuda(); pred = net.forward(img_t)
                    if isinstance(pred, tuple): pred = pred[-1]
                    pred = pred[:,:,:size[0],:size[1]]; gt_t = gt_t[:,:,:size[0],:size[1]]
                    test_loss.append(net.loss(pred, gt_t.cuda()).detach().cpu())
                    eval_mIoU.update((pred>0.5).cpu(), gt_t.cpu())
                    eval_PD_FA.update((pred[0,0]>0.5).cpu(), gt_t[0,0], size)
            results1, results2 = eval_mIoU.get(), eval_PD_FA.get()
            writer.add_scalar('mIOU', results1[1], idx_epoch+1)
            print(f'    Test: mIoU={results1[1]*100:.2f}%  Pd={results2[0]*100:.2f}%  Fa={results2[1]*1e6:.2f} x10-6')
            if idx_epoch == 0: best_mIoU = results1
            if results1[1] > best_mIoU[1]:
                best_mIoU = results1
                sp = f'{args.save}/{args.dataset_name}/best.pth.tar'; os.makedirs(os.path.dirname(sp), exist_ok=True)
                torch.save({'epoch': idx_epoch+1, 'state_dict': net.state_dict(), 'total_loss': total_loss_list}, sp)
                print(f'    >>> Best mIoU={best_mIoU[1]*100:.2f}% saved')
        if (idx_epoch+1) % args.every_save_pth == 0:
            sp = f'{args.save}/{args.dataset_name}/epoch_{idx_epoch+1}.pth.tar'; os.makedirs(os.path.dirname(sp), exist_ok=True)
            torch.save({'epoch': idx_epoch+1, 'state_dict': net.state_dict(), 'total_loss': total_loss_list}, sp)

# ========================================================================
#  Main
# ========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PTBD-Net Training")
    parser.add_argument("--dataset_names", nargs="+", default=['NUAA-SIRST'])
    parser.add_argument("--dataset_dir", type=str, default='./data', help='Path to the dataset root directory.')
    parser.add_argument("--batchSize", type=int, default=16)
    parser.add_argument("--patchSize", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--begin_test", type=int, default=500)
    parser.add_argument("--every_test", type=int, default=5)
    parser.add_argument("--every_save_pth", type=int, default=50)
    parser.add_argument("--every_print", type=int, default=10)
    parser.add_argument("--save", type=str, default='./checkpoints')
    parser.add_argument("--log_dir", type=str, default='./logs/training')
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_pytorch(args.seed); config = get_config()
    for ds_name in args.dataset_names:
        args.dataset_name = ds_name
        print(f'\n{"="*60}\n  PTBD-Net | {ds_name} | bs={args.batchSize} | epochs={args.epochs}\n{"="*60}')
        train(args, config)

