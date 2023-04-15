import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = None

rgb_avg = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]

transdata = transforms.Compose(
	[transforms.Resize(256,interpolation=1),
	 transforms.CenterCrop(224),
	 transforms.ToTensor(),
	 transforms.Normalize(rgb_avg, rgb_std)])

def loadvaldata(datapath, gpuid, testsize=-1):
    global device
    device = torch.device("cuda:" + str(gpuid) if torch.cuda.is_available() else "cpu")

    images = datasets.ImageNet(\
                root=datapath,\
                split='val',transform=transdata)

    print(len(images.samples))
    if testsize != -1:
        images.samples = images.samples[::len(images.samples) // testsize]
    labels = torch.tensor([images.samples[i][1] for i in range(0, len(images))])

    # for i in range(0,len(images)):
    #     print(images.samples[i])

    return images, labels.to(device)


def loadtraindata(datapath, gpuid):
    global device
    device = torch.device("cuda:" + str(gpuid) if torch.cuda.is_available() else "cpu")

    images = datasets.ImageNet(\
                root=datapath,\
                split='train',transform=transdata)

    labels = torch.tensor([images.samples[i][1] for i in range(0, len(images))])

    return images, labels.to(device)


def predict(net, images, batch_size=256, num_workers=16):
    global device
    y_hat = torch.zeros(0, device=device)
    loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            y_hat = torch.cat((y_hat,net(x)))
    return y_hat

def predict2(net, loader):
    global device
    i = 0
    y_hat = torch.zeros(0, device=device)
    # print(y_hat.shape)
    #loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    net.eval()
    with torch.no_grad():
        for x, _ in iter(loader):
            print(f'the current i is {i}')
            x = x.to(device)
            y_hat = torch.cat((y_hat,net(x)))
            # print(y_hat.shape)
            i += 1
    return y_hat

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        # print(maxk)
        batch_size = target.size(0)
        # print(batch_size)

        _, pred = output.topk(maxk, 1, True, True)
        # print(_)
        # print(pred[0])
        pred = pred.t()
        # print(pred)
        #pred.reshape(pred.shape[0], -1)
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # print(target.view(1, -1).expand_as(pred))
        # print(correct)

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            # print(correct[:k].contiguous().view(-1).float().sum(0,keepdim=True))
            res.append(correct_k.mul_(100.0 / batch_size))
        return res