from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import torch.utils.data as data
import os, torch, argparse
from torch import nn
import torch.nn.functional as F


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--root', type=str, default='.', help='root dir of training')
parser.add_argument('--scale_size', type=int, default=299, help='resize images to this size')
parser.add_argument('--crop_size', type=int, default=299, help='then crop images to this size')
parser.add_argument('--batch_size', type=int, default=64, help='size of mini batch')
parser.add_argument('--cuda', action='store_true', default=False, help='enable CUDA training')

parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2')
parser.add_argument('--eps', type=float, default=1e-8, help='Adam epsilon')

parser.add_argument('--epochs', type=int, default=50, help='# of epochs to train')
parser.add_argument('--log_freq', type=int, default=100, help='how many batches before logging training status')
parser.add_argument('--save_freq', type=int, default=10, help='how many epochs before saving model parameters')
parser.add_argument('--model', type=str, default='inception_v3.pth', help='filename of pretrained parameters')

opt = parser.parse_args()

transform = transforms.Compose([
    transforms.Resize(opt.scale_size),
    transforms.CenterCrop(opt.crop_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229,0.224,0.225])
])

checkpoints_dir = os.path.join(opt.root, 'checkpoints')
os.makedirs(checkpoints_dir, exist_ok=True)
kwargs = {'num_workers': 8, 'pin_memory': True}

print('+ loading data')
print('|  - loading train images')
train_dataset = ImageFolder(os.path.join(opt.root, 'images', 'train'), transform)
train_loader = data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, **kwargs)
print('|  - loading test images')
test_dataset = ImageFolder(os.path.join(opt.root, 'images', 'test'), transform)
test_loader = data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, **kwargs)
print('|  - done.')

print('+ creating model')
device = torch.device('cuda:0') if opt.cuda else torch.device('cpu')

model = models.inception_v3(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.AuxLogits.fc = nn.Linear(768, 16)
model.fc = nn.Linear(2048, 16)
model.aux_logits = False
model = model.to(device)
print('params to update:')
params_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print('\t', name)
optimizer = torch.optim.Adam(params_to_update, lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)

print('|  - training...')
for epoch in range(1, opt.epochs + 1):
    model.train()
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if idx % opt.log_freq:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(data), len(train_loader.dataset),
                100. * idx / len(train_loader), loss.item()))
    model.eval()
    test_loss, correct = 0, 0
    batch_size = test_loader.batch_size
    error, i = [], 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    if epoch % opt.save_freq == 0:
        filename = 'params_%depoch.pth' % epoch
        torch.save(model.state_dict(), os.path.join(checkpoints_dir, filename))

