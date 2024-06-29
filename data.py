import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision import datasets, transforms

def get_cifar100_loaders(img_size=224, train_batch_size=128, eval_batch_size=128, num_workers=4):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    trainset = datasets.CIFAR100(root="autodl-tmp/VIT/data",
                                 train=True,
                                 download=True,
                                 transform=transform_train)
    testset = datasets.CIFAR100(root="autodl-tmp/VIT/data",
                                train=False,
                                download=True,
                                transform=transform_test)

    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=train_batch_size,
                              num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=eval_batch_size,
                             num_workers=num_workers,
                             pin_memory=True)

    return train_loader, test_loader
