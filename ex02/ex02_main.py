import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os

from ex02_model import Unet
from ex02_diffusion import Diffusion, linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from ex02_helpers import num_to_groups
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to diffuse images')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--timesteps', type=int, default=100, help='number of timesteps for diffusion model (default: 100)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    # parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--run_name', type=str, default="DDPM")
    parser.add_argument('--dry_run', action='store_true', default=False, help='quickly check a single pass')
    return parser.parse_args()


def sample_and_save_images(n_images, diffusor, model, device, store_path, transform=None):
    # TODO: Implement - adapt code and method signature as needed

    class_labels = diffusor.class_labels
    if class_labels is not None:
        class_labels = torch.tensor(class_labels, device=device).long()
    image_size = diffusor.img_size
    channels = 3

    # sample 64 images from the model
    sampled_images = diffusor.sample(model, image_size=image_size, batch_size=n_images, channels=channels,
                                     class_labels=class_labels)

    for t in range(diffusor.timesteps)[-3:-1]:
        # save the images
        for image in range(sampled_images[0].shape[0]):
            image_transform = transform(sampled_images[t][image])
            plt.imshow(image_transform)
            save_image(sampled_images[t][image], os.path.join(store_path, "sampled_image_{}_t_{}.png".format(image, t)))
            plt.show()


def test(model, testloader, diffusor, device, args):
    # TODO: Implement - adapt code and method signature as needed

    batch_size = args.batch_size
    timesteps = args.timesteps

    pbar = tqdm(testloader)

    for step, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        t = torch.randint(0, timesteps, (len(images),), device=device).long()
        loss = diffusor.p_losses(model, images, t, noise=None, class_labels=labels, loss_type="l2")

        if step % args.log_interval == 0:
            print('Test Step: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                step, step * len(images), len(testloader.dataset),
                100. * step / len(testloader), loss.item()))
        if args.dry_run:
            break


def train(model, trainloader, optimizer, diffusor, epoch, device, args):
    batch_size = args.batch_size
    timesteps = args.timesteps

    pbar = tqdm(trainloader)
    for step, (images, labels) in enumerate(pbar):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Algorithm 1 line 3: sample t uniformly for every example in the batch
        t = torch.randint(0, timesteps, (len(images),), device=device).long()
        loss = diffusor.p_losses(model, images, t, noise=None, class_labels=labels, loss_type="l2")

        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(images), len(trainloader.dataset),
                100. * step / len(trainloader), loss.item()))
        if args.dry_run:
            break


def test_test(args):
    # TODO (2.2): implement testing functionality, including generation of stored images.

    # test beta schedules
    timesteps = args.timesteps

    # Standard linear schedule
    linear_schedule = linear_beta_schedule(0.01, 1.0, timesteps)

    # Cosine schedule
    cosine_schedule = cosine_beta_schedule(timesteps)

    # Sigmoid schedule
    sigmoid_schedule = sigmoid_beta_schedule(0.01, 1.0, timesteps)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(linear_schedule, label='Linear Schedule')
    plt.plot(cosine_schedule, label='Cosine Schedule')
    plt.plot(sigmoid_schedule, label='Sigmoid Schedule')
    plt.xlabel('Timesteps')
    plt.ylabel('Beta Values')
    plt.title('Comparison of Beta Schedulers')
    plt.legend()
    plt.show()


def run(args):
    timesteps = args.timesteps
    image_size = 32  # TODO (2.5): Adapt to new dataset
    channels = 3
    epochs = args.epochs
    batch_size = args.batch_size
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"

    model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,), class_free_guidance=True).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    my_scheduler = lambda x: linear_beta_schedule(0.0001, 0.02, x)

    diffusor = Diffusion(timesteps, my_scheduler, image_size, device)

    # define image transformations (e.g. using torchvision)
    transform = Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),    # turn into torch Tensor of shape CHW, divide by 255
        transforms.Lambda(lambda t: (t * 2) - 1)   # scale data to [-1, 1] to aid diffusion process
    ])
    reverse_transform = Compose([
        Lambda(lambda t: (t.clamp(-1, 1) + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    dataset = datasets.CIFAR10('/proj/aimi-adl/CIFAR10/', download=True, train=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    # Download and load the test data
    testset = datasets.CIFAR10('/proj/aimi-adl/CIFAR10/', download=True, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=int(batch_size/2), shuffle=True)

    for epoch in range(epochs):
        train(model, trainloader, optimizer, diffusor, epoch, device, args)
        test(model, valloader, diffusor, device, args)

    test(model, testloader, diffusor, device, args)

    save_path = Path("/home/cip/medtech2021/ez72oxib/Desktop/AdvancedDeepLearning/generated_images")  # TODO: Adapt to your needs
    save_path.mkdir(exist_ok=True)

    n_images = 8
    sample_and_save_images(n_images, diffusor, model, device, save_path, reverse_transform)

    # TODO (2.2):comparison of beta schedules
    test_test(args)
    # Create the directory if it doesn't exist
    checkpoint_dir = os.path.join("/home/cip/medtech2021/ez72oxib/Desktop/AdvancedDeepLearning/models", args.run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "ckpt.pt"))


if __name__ == '__main__':
    args = parse_args()
    # TODO (2.2): Add visualization capabilities
    run(args)

