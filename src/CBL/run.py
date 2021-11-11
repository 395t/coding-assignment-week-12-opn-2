import os
from types import SimpleNamespace

import fire
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from PIL import Image

class JointClassifier(torch.nn.Module):
    def __init__(self, image_size, num_class):
        super().__init__()
        self.backbone_nn = torchvision.models.resnext50_32x4d(pretrained=False, progress=True)
        self.backbone_nn.fc = torch.nn.Linear(in_features=2048, out_features = 3 * image_size * image_size)
        self.fc = torch.nn.Linear(3 * image_size * image_size, num_class)

    def forward(self, x):
        feature_representation = self.backbone_nn(x)
        pred = self.fc(F.relu(feature_representation))
        return pred


def get_data(name, device=None):
    tr_x, tr_y, te_x, te_y, class_counts, image_size = {
        'cifar10': _get_cifar10,
        'cifar100': _get_cifar100,
        'tinyimagenet': _get_tinyimagenet,
        }[name]()
    assert tr_x.dtype == te_x.dtype == torch.uint8
    assert tr_y.dtype == te_y.dtype == torch.int64
    tr_x = tr_x.float() / 127.5 - 1.0
    te_x = te_x.float() / 127.5 - 1.0

    if device is not None:
        tr_x, tr_y, te_x, te_y = tr_x.to(device), tr_y.to(device), te_x.to(device), te_y.to(device)
    
    return SimpleNamespace(tr_x=tr_x, tr_y=tr_y, te_x=te_x, te_y=te_y,
                           name=name, class_counts=class_counts, image_size=image_size)

def _get_cifar10():
    root_dir = os.path.join('./data/cifar10')
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ColorJitter(brightness=0.5, hue=0.25),
                                          transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root=root_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=root_dir, train=False, download=True, transform=transform_test)

    # Create long-tail distribution
    class_counts = [5000, 3000, 1500, 1000, 800, 600, 400, 200, 100, 50]
    distribution = list(class_counts)
    tr_x, tr_y = [], []
    for (image, label) in train_dataset:
        if distribution[label] <= 0:
            continue
        tr_x.append(image)
        tr_y.append(label)
        distribution[label] -= 1

    tr_x = torch.stack(tr_x)
    tr_y = torch.Tensor(tr_y).long()
    assert tr_x.dtype == torch.float32 and 0 <= tr_x.min().item() and tr_x.max().item() <= 1

    tr_x = (tr_x * 255.).round().byte()
    te_x, te_y = torch.from_numpy(test_dataset.data), torch.Tensor(test_dataset.targets).long()
    te_x = te_x.permute(0, 3, 1, 2).contiguous()
    assert tr_x.dtype == te_x.dtype == torch.uint8 and tr_y.dtype == te_y.dtype == torch.int64
    assert len(tr_x) == len(tr_y) and len(te_x) == len(te_y) and tr_x.shape[1:] == te_x.shape[1:]

    return tr_x, tr_y, te_x, te_y, class_counts, 32

def _get_cifar100():
    root_dir = os.path.join('./data/cifar100')
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ColorJitter(brightness=0.5, hue=0.25),
                                          transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR100(root=root_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root=root_dir, train=False, download=True, transform=transform_test)

    # Create long-tail distribution
    class_counts = [500] * 100
    class_counts[10:20] = [300] * 10
    class_counts[20:30] = [250] * 10
    class_counts[30:40] = [200] * 10
    class_counts[40:50] = [100] * 10
    class_counts[50:60] = [80] * 10
    class_counts[60:70] = [75] * 10
    class_counts[70:80] = [60] * 10
    class_counts[80:90] = [55] * 10
    class_counts[90:100] = [50] * 10
    distribution = list(class_counts)

    tr_x, tr_y = [], []
    for (image, label) in train_dataset:
        if distribution[label] <= 0:
            continue
        tr_x.append(image)
        tr_y.append(label)
        distribution[label] -= 1

    tr_x = torch.stack(tr_x)
    tr_y = torch.Tensor(tr_y).long()
    assert tr_x.dtype == torch.float32 and 0 <= tr_x.min().item() and tr_x.max().item() <= 1

    tr_x = (tr_x * 255.).round().byte()
    te_x, te_y = torch.from_numpy(test_dataset.data), torch.Tensor(test_dataset.targets).long()
    te_x = te_x.permute(0, 3, 1, 2).contiguous()
    assert tr_x.dtype == te_x.dtype == torch.uint8 and tr_y.dtype == te_y.dtype == torch.int64
    assert len(tr_x) == len(tr_y) and len(te_x) == len(te_y) and tr_x.shape[1:] == te_x.shape[1:]

    return tr_x, tr_y, te_x, te_y, class_counts, 32

def _get_tinyimagenet():
    dd = torch.load('./data/tiny_imagenet.pt')
    tr_x, tr_y = [], []
    orig_tr_x, orig_tr_y = dd['tr_x'], dd['tr_y']
    te_x, te_y = dd['te_x'], dd['te_y']

      # Goal Long-tail distribution
    class_counts = [500] * 200
    class_counts[20:40] = [300] * 20
    class_counts[40:60] = [250] * 20
    class_counts[60:80] = [200] * 20
    class_counts[80:100] = [100] * 20
    class_counts[100:120] = [80] * 20
    class_counts[120:140] = [75] * 20
    class_counts[140:160] = [60] * 20
    class_counts[160:180] = [55] * 20
    class_counts[180:200] = [50] * 20
    distribution = list(class_counts)
    for image, label in zip(orig_tr_x, orig_tr_y):
        label = int(label)
        if distribution[label] <= 0:
            continue
        tr_x.append(image)
        tr_y.append(label)
        distribution[label] -= 1

    tr_x = torch.stack(tr_x)
    tr_y = torch.Tensor(tr_y).long()
    assert tr_x.dtype == torch.float32 and 0 <= tr_x.min().item() and tr_x.max().item() <= 1
    assert te_x.dtype == torch.float32 and 0 <= te_x.min().item() and te_x.max().item() <= 1

    tr_x = (tr_x * 255.).round().byte()
    te_x = (te_x * 255.).round().byte()
    assert tr_x.dtype == te_x.dtype == torch.uint8 and tr_y.dtype == te_y.dtype == torch.int64
    assert len(tr_x) == len(tr_y) and len(te_x) == len(te_y) and tr_x.shape[1:] == te_x.shape[1:]

    return tr_x, tr_y, te_x, te_y, class_counts, 64

@torch.no_grad()
def evaluate(model, dset, bs):
    eval_result = {}
    model.eval()
    num_batches = int(np.ceil(len(dset.te_x) / bs))
    cpu_y = dset.te_y.cpu()
    preds, losses = [], []
    for i in range(0, num_batches):
        batch_x = dset.te_x[i*bs:(i+1)*bs]
        batch_y = dset.te_y[i*bs:(i+1)*bs]
        pred_y = model(batch_x)
        loss = F.cross_entropy(pred_y, batch_y, reduction='none')
        pred = (batch_y == pred_y.argmax(dim=1)).float()
        losses.append(loss)
        preds.append(pred)
    losses = torch.cat(losses).cpu().numpy()
    preds = torch.cat(preds).cpu().numpy()
    model.train()
    eval_result['losses'] = losses
    eval_result['preds'] = preds
    eval_result['loss_per_class'] = []
    eval_result['acc_per_class'] = []
    assert len(losses) == len(preds) == len(dset.te_x)
    for i, _ in enumerate(dset.class_counts):
        mask = (cpu_y == i)
        eval_result['loss_per_class'].append(losses[mask].mean().item())
        eval_result['acc_per_class'].append(preds[mask].mean().item())
    return SimpleNamespace(**eval_result)

def get_cb_weights(beta, class_counts):
    class_counts = torch.tensor(class_counts, dtype=torch.float32)
    cb_weights = torch.ones_like(class_counts)
    if beta is not None:
        alpha = torch.empty_like(class_counts)
        beta_t = torch.tensor(beta, dtype=torch.float64) 
        for c, ny in enumerate(class_counts):
            alpha[c] = (1 - beta_t) / (beta_t ** torch.arange(ny)).sum()
        cb_weights = alpha * len(class_counts) / alpha.sum()
    return cb_weights

def train(model, dset, num_epochs, bs, lr, beta):
    tr_size = len(dset.tr_x)
    num_batches = int(np.ceil(tr_size / bs))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    result = SimpleNamespace()
    result.dataset = dset.name
    result.class_counts = dset.class_counts
    result.beta = beta
    result.num_epochs = num_epochs
    result.batch_size = bs
    result.learning_rate = lr
    result.train_losses = []
    result.eval_results = []

    cb_weights = get_cb_weights(beta, dset.class_counts)
    cb_weights = cb_weights.to(dset.tr_x)

    for ep in range(1, num_epochs+1):
        perm = np.random.choice(tr_size, tr_size, replace=False)
        for i in range(0, num_batches):
            idx = perm[i*bs:(i+1)*bs]
            batch_x = dset.tr_x[idx]
            batch_y = dset.tr_y[idx]
            pred_y = model(batch_x)

            weights = cb_weights[batch_y]
            loss = F.cross_entropy(pred_y, batch_y, reduction='none') * weights
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            result.train_losses.append(loss.item())

            print(f'\rep {ep:02} step {i+1:03d}/{num_batches:03d} ',
                  f'loss {loss:.4f} ', end='', flush=True) 

        eval_result = evaluate(model, dset, bs)
        result.eval_results.append(eval_result)
        print(f' -> eval_acc {eval_result.preds.mean():.3f} eval_loss {eval_result.losses.mean():.4f}')

    out_file = os.path.join('results', f'dataset={result.dataset}..beta={result.beta}.pt')
    print(f'Training finished.  Saving model and result to {out_file}')
    model.eval()
    model = model.to(torch.device('cpu'))
    torch.save({'state_dict': model.state_dict(),
                'result': result}, out_file)

    print('Per-class eval result:')
    for c, acc in enumerate(result.eval_results[-1].acc_per_class):
        class_loss = result.eval_results[-1].loss_per_class[c]
        print(f'  - [class {c:03d}] acc: {acc:.5f} / loss: {class_loss:.4f} (count: {result.class_counts[c]})')

def prep_data():
    val_img_dir = './data/tiny-imagenet-200/val/images'
    if os.path.exists(val_img_dir):
        print('Fixing TinyImagennet validation set.')
        data = open('./data/tiny-imagenet-200/val/val_annotations.txt', 'r').readlines()
        val_img_dict = {}
        for line in data:
            words = line.split('\t')
            val_img_dict[words[0]] = words[1]

        for img, folder in val_img_dict.items():
            new_dir = f'./data/tiny-imagenet-200/val/{folder}'
            os.makedirs(new_dir, exist_ok=True)
            old_file = f'{val_img_dir}/{img}'
            if os.path.exists(old_file):
                os.rename(old_file, f'{new_dir}/{img}')

        import shutil
        shutil.rmtree(val_img_dir)

    root_dir = os.path.join('./data/tiny-imagenet-200')
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ColorJitter(brightness=0.5, hue=0.25),
                                          transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(root=f'{root_dir}/train', transform=transform_train)
    test_dataset = datasets.ImageFolder(root=f'{root_dir}/val', transform=transform_test)

    len_tr, len_te = len(train_dataset), len(test_dataset)
    tr_x = torch.empty(len_tr, *train_dataset[0][0].shape, dtype=torch.float32)
    tr_y = torch.empty(len_tr, dtype=torch.int64)
    te_x = torch.empty(len_te, *test_dataset[0][0].shape, dtype=torch.float32)
    te_y = torch.empty(len_te, dtype=torch.int64)

    for i in range(len_tr):
        x, y = train_dataset[i]
        tr_x[i] = x
        tr_y[i] = y
    for i in range(len_te):
        x, y = test_dataset[i]
        te_x[i] = x
        te_y[i] = y

    torch.save({'tr_x': tr_x, 'tr_y': tr_y, 'te_x': te_x, 'te_y': te_y},
                './data/tiny_imagenet.pt')
    print(f'Finished preparing TinyImagenet: {tr_x.shape} / {te_x.shape}')

def main(*,
         dataset, beta,
         epochs: int = 30, bs: int = 128, lr: float = 1e-3):
    print(f'Training for dataset {dataset} and beta {beta}')
    assert dataset in ('cifar10', 'cifar100', 'tinyimagenet')
    device = torch.device('cuda:0')
    dset = get_data(dataset, device)
    model = JointClassifier(dset.image_size, len(dset.class_counts)).to(device)
    train(model, dset, epochs, bs, lr, beta)
    print()

if __name__ == '__main__':
    fire.Fire({
        'train': main,
        'prep_data': prep_data,
    })