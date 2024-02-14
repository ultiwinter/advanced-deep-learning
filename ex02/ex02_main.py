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
import matplotlib.animation as animation
from ex02_model import Unet
from ex02_diffusion import Diffusion, linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from ex02_helpers import num_to_groups
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to diffuse images')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
<<<<<<< HEAD
    parser.add_argument('--timesteps', type=int, default=100, help='number of timesteps for diffusion model (default: 100)')
    parser.add_argument('--epochs', type=int, default=15, help='number of epochs to train (default: 5)')
=======
    parser.add_argument('--timesteps', type=int, default=700, help='number of timesteps for diffusion model (default: 100)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 5)')
>>>>>>> main
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    # parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--class_conditional', action='store_true', default=False,
                        help='Enable class conditional training')
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--run_name', type=str, default="DDPM")
    parser.add_argument('--dry_run', action='store_true', default=False, help='quickly check a single pass')
    return parser.parse_args()


def sample_and_save_images(n_images, diffusor, model, device, class_labels=None, store_path=None):
    # TODO: Implement - adapt code and method signature as needed

    reverse_transform = Compose([
        Lambda(lambda t: (t.clamp(-1, 1) + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    if class_labels is not None:
        class_labels = torch.tensor(class_labels, device=device).long()
    image_size = diffusor.img_size
    channels = 3

    # sample images from the model
    sampled_images = diffusor.sample(model, image_size=image_size, batch_size=n_images, channels=channels,
                                     class_labels=class_labels)

    # pick the last timestep
    last_timestep = -1
    for img_idx in range(n_images):
        image_transform = reverse_transform(sampled_images[last_timestep][img_idx])
        plt.imshow(image_transform)
<<<<<<< HEAD
        save_image(sampled_images[img_idx], os.path.join(store_path, "sampled_image_{}.png".format(img_idx)))
=======
        #save_image(sampled_images[last_timestep][img_idx], os.path.join(store_path, "sampled_image_{}.png".format(img_idx)))

        # save the transformed image
        image_transform.save(os.path.join(store_path, "sampled_image_{}.png".format(img_idx)))

>>>>>>> 2e72ed4817817ee450006faa041ed0d2962dbfe1
        plt.show()
    return sampled_images


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
<<<<<<< HEAD
=======

>>>>>>> main
        optimizer.zero_grad()

        # Algorithm 1 line 3: sample t uniformly for every example in the batch
        t = torch.randint(0, timesteps, (len(images),), device=device).long()
<<<<<<< HEAD
        loss = diffusor.p_losses(model, images, t, class_label=labels, loss_type="l2")
=======
        loss = diffusor.p_losses(model, images, t, noise=None, class_labels=labels, loss_type="l2")
>>>>>>> main

        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            # n_images = 10
            # labels_sample = torch.zeros(n_images, device=device) + 1
            # sample_and_save_images(n_images, diffusor, model, device, class_labels=labels_sample)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(images), len(trainloader.dataset),
                100. * step / len(trainloader), loss.item()))
        if args.dry_run:
            break


def beta_show(args):

    # TODO (2.2):comparison of beta schedules
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

    # save the plot in a folder called plots
    # Create the directory if it doesn't exist

    plot_dir = os.path.join("/proj/ciptmp/af23aduk/adl_ex02/classfreeFalse/plots")
    checkpoint_dir = os.path.join(plot_dir, args.run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    plt.savefig(os.path.join(checkpoint_dir, "beta_schedulers.png"))


def add_visualization(timesteps, image_size, channels, sampled_images):
    # TODO (2.2): Add visualization capabilities

    reverse_transform = Compose([
        Lambda(lambda t: (t.clamp(-1, 1) + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    random_index = 5

    fig = plt.figure()
    ims = []
    for i in range(timesteps):
        im = plt.imshow(reverse_transform(sampled_images[i][random_index]), cmap="gray", animated=True)
        ims.append([im])

    animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    # create a directory for the GIFs
    # Create the directory if it doesn't exist
    gifs_dir = os.path.join("/proj/ciptmp/af23aduk/adl_ex02/classfreeFalse/GIFs")
    checkpoint_dir = os.path.join(gifs_dir, args.run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    animate.save(os.path.join(checkpoint_dir, "diffusion_last.gif"))
    plt.show()


def run(args):
    timesteps = args.timesteps
    image_size = 32  # TODO (2.5): Adapt to new dataset
    channels = 3
    epochs = args.epochs
    batch_size = args.batch_size
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"

<<<<<<< HEAD
    #model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,)).to(device)
    model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,),class_free_guidance=args.class_conditional).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    my_scheduler = lambda x: linear_beta_schedule(0.0001, 0.02, x)
    diffusor = Diffusion(timesteps, my_scheduler, image_size, device) #################
=======
    model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,), class_free_guidance=False).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    #my_scheduler = lambda x: linear_beta_schedule(0.0001, 0.02, x)

    # try out different beta schedules for example sigmoid_beta_schedule
    my_scheduler = lambda x: sigmoid_beta_schedule(0.0001, 0.02, x)
    # my_scheduler = lambda x: cosine_beta_schedule(x)

    diffusor = Diffusion(timesteps, my_scheduler, image_size, device)
>>>>>>> main

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

<<<<<<< HEAD
    save_path = Path("/proj/ciptmp/af23aduk/adl_ex02/generated_images")  # TODO: Adapt to your needs
=======
    save_path = Path("/proj/ciptmp/af23aduk/adl_ex02/classfreeFalse/generated_images")  # TODO: Adapt to your needs
>>>>>>> main
    save_path.mkdir(exist_ok=True)

    n_images = 10
    #labels = list(range(n_images))
    #labels = torch.tensor(labels, device=device).long()
    sampled_images = sample_and_save_images(n_images, diffusor, model, device, None, save_path)

    add_visualization(timesteps, image_size, channels, sampled_images)

    # Create the directory if it doesn't exist
<<<<<<< HEAD
    checkpoint_dir = os.path.join("/proj/ciptmp/af23aduk/adl_ex02/models", args.run_name)
=======
    checkpoint_dir = os.path.join("/proj/ciptmp/af23aduk/adl_ex02/classfreeFalse/models", args.run_name)
>>>>>>> main
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "ckpt.pt"))


if __name__ == '__main__':
    args = parse_args()
    # TODO (2.2): Add visualization capabilities
    beta_show(args)
    run(args)

