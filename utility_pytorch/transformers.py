from torchvision import transforms
try:
    from . import custom_transformers
except:
    import custom_transformers

num_frames = 5

imagenet_train_transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
])


imagenet_test_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
])


cifar10_train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # custom_transformers.LinearMix(),
])


cifar10_test_transform = transforms.Compose([
    transforms.ToTensor(),
])
