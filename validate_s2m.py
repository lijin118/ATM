import torch
import torch.nn as nn
from torchvision import transforms
from data.data_list import ImageList

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            feature, output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.data.cpu().max(1, keepdim=True)[1]
            correct += pred.eq(target.data.cpu().view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)

    return acc


test_list = 'data/mnist_test.txt'
model = torch.load('model/s2m_94.84.pth')
test_loader = torch.utils.data.DataLoader(
        ImageList(open(test_list).readlines(), transform=transforms.Compose([
                           transforms.Resize((32,32)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]), mode='RGB'),
        batch_size=1000, shuffle=True, num_workers=1)
acc = test(model, test_loader)
print(acc)
