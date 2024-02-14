## Standard libraries
import os
import numpy as np
import tqdm
import pandas as pd
import argparse
from typing import Union, Dict
import random

## Imports for plotting
import matplotlib.pyplot as plt
import seaborn as sns

## Imports for data loading
from pathlib import Path

## PyTorch & DL
import torch
import torch.utils.data as data
import torch.optim as optim
import torchmetrics
import torchvision

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Deterministic operations on GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

## Misc
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc

from ex03_data import get_datasets, TransformTensorDataset
from ex03_model import ShallowCNN
from ex03_ood import score_fn

# TODO: My imports
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import warnings
from pytorch_lightning.utilities.warnings import PossibleUserWarning

# Filter out the PossibleUserWarning
warnings.filterwarnings("ignore", category=PossibleUserWarning)


def parse_args():
    parser = argparse.ArgumentParser(description='Configure training/inference/sampling for EBMs')
    # TODO: Change the default path to the correct one.
    parser.add_argument('--data_dir', type=str, default="/proj/aimi-adl/GLYPHS/.",
                        help='path to directory with glyph image data')
    # TODO: Save model checkpoints in a local directory.
    parser.add_argument('--ckpt_dir', type=str, default="/proj/ciptmp/af23aduk/adl_ex03/saved_models/",
                        help='path to directory where model checkpoints are stored')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--num_epochs', type=int, default=22,  # Default 120
                        help='number of epochs to train (default: 120)')
    parser.add_argument('--cbuffer_size', type=int, default=128,
                        help='num. images per class in the sampling reservoir (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--lr_gamma', type=float, default=0.97,
                        help='exponentional learning rate decay factor (default: 0.97)')
    parser.add_argument('--lr_stepsize', type=int, default=2,
                        help='learning rate decay step size (default: 2)')
    parser.add_argument('--alpha', type=int, default=0.1,
                        help='strength of L2 regularization (default: 0.1)')
    parser.add_argument('--num_classes', type=int, default=42,
                        help='number of output nodes/classes (default: 1 (EBM), 42 (JEM))')
    parser.add_argument('--ccond_sample', type=bool, default=True,
                        help='flag that specifies class-conditional or unconditional sampling (default: false')
    parser.add_argument('--num_workers', type=int, default="0",
                        help='number of loading workers, needs to be 0 for Windows')
    return parser.parse_args()


class MCMCSampler:
    def __init__(self, model, img_shape, sample_size, num_classes, cbuffer_size=256):
        """
        MCMC sampler that uses SGLD.

        :param model: Neural network to use for modeling the energy function E_\theta
        :param img_shape: Image shape (height x width)
        :param sample_size: Number of images to sample
        :param num_classes: Number of output nodes, i.e., number of classes
        :param cbuffer_size: Size of the buffer per class the is being retained for reservoir sampling
        """
        super().__init__()
        self.model = model
        self.img_shape = img_shape
        self.sample_size = sample_size
        self.num_classes = num_classes
        self.cbuffer_size = cbuffer_size
        # Initialize with fresh gaussian noise 100%
        self.images = [(torch.rand((1,) + self.img_shape) * 2 - 1) for _ in range(self.sample_size)]

    def synthesize_samples(self, clabel=None, steps=60, step_size=10, return_img_per_step=False):
        """
        Synthesize images from the current parameterized q_\theta

        :param model: Neural network to use to model E_theta
        :param clabel: Class label(s) used to sample the buffer
        :param steps: Number of iterations in the MCMC algorithm.
        :param step_size: Learning rate/update step size
        :param return_img_per_step: images during MCMC-based synthesis
        :return: synthesized images
        """
        # Before MCMC: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input.
        is_training = self.model.training
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Enable gradient calculation if not already the case
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # TODO (3.3): Implement SGLD-based synthesis with reservoir sampling

        # Sample initial data points x^0 to get a starting point for the sampling process.
        # As seen in the lecture and the theoretical recap, there exist multiple variants how we can approach this task.

        # --> Here, you should use non-persistent short-run MCMC and combine it with reservoir sampling. This means that
        # you sample a small portion of new images from random Gaussian noise, while the rest is taking from a buffer
        # that is re-populated at the end of synthesis.

        # In practical terms, you want to create a buffer that persists across epochs
        # (consider saving that into a field of this class). In this buffer, you store the synthesized samples after
        # each SGLD procedure. In the class-conditional setting, you want to have individual buffers per class.
        # Please make sure that you keep the buffer finite to not run into memory-related problems.

        # Initialize a list to track buffer indices for each label
        buffer_indices = []

        percent_new = 0.05

        # Choose 95% of the batch from the buffer, 5% generate from scratch
        n_new = np.random.binomial(self.sample_size, percent_new)
        rand_imgs = torch.rand((n_new,) + self.img_shape, device=device) * 2 - 1

        # Initialize reservoir buffer if it doesn't exist for class-conditional sampling
        if not hasattr(self, 'reservoir_buffer'):
            self.reservoir_buffer = [torch.randn(self.cbuffer_size, *self.img_shape, device=device) * 0.01 for _ in range(self.num_classes)]

        # Sample from the buffer if class-conditional sampling is enabled
        if clabel is not None:
            buffer_samples = []
            for label in clabel:
                # Convert the label to an integer if it's a tensor
                label = label.item() if isinstance(label, torch.Tensor) else label

                if random.random() < percent_new:
                    # New image case, no index to track
                    buffer_indices.append(None)
                    # Generate new image
                    new_img = torch.rand((1,) + self.img_shape, device=device) * 2 - 1
                else:
                    # Select image from the class-specific buffer
                    # Existing buffer index case
                    buffer_index = random.choice(range(self.cbuffer_size))
                    buffer_indices.append((label, buffer_index))
                    new_img = random.choice(self.reservoir_buffer[label]).to(device)

                # Ensure new_img has the correct shape, e.g., [1, C, H, W]
                if len(new_img.shape) == 3:  # If the shape is [C, H, W]
                    new_img = new_img.unsqueeze(0)  # Reshape to [1, C, H, W]

                buffer_samples.append(new_img)
            buffer_samples = torch.stack(buffer_samples).to(device).detach()
            buffer_samples.squeeze_(1)  # Remove channel dimension
            inp_imgs = buffer_samples.detach().to(device)
        else:
            old_imgs = torch.cat(random.choices(self.images, k=self.sample_size - n_new), dim=0).detach().to(device)
            inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).detach().to(device)

        inp_imgs.requires_grad = True

        # More efficient than creating a new tensor every iteration. (Gaussian noise)
        # noise = torch.randn(inp_imgs.shape, device=inp_imgs.device)
        gaussian_noise = torch.randn(inp_imgs.shape, device=inp_imgs.device)
        noise = torch.cumsum(gaussian_noise, dim=0)  # Brownian noise

        imgs_per_step = []

        # Execute K MCMC steps
        for _ in range(steps):
            # (1) Add small noise to the input 'inp_imgs' (which are normalized to a range of -1 to 1).
            # This corresponds to the Brownian noise that allows to explore the entire parameter space.

            # (2) Calculate gradient-based score function at the current step. In case of the JEM implementation AND
            # class-conditional sampling (which is optional from a methodological point of view), make sure that you
            # plug in some label information as well as we want to calculate E(x,y) and not only E(x).

            # (3) Perform gradient ascent to regions of higher probability
            # (gradient descent if we consider the energy surface!). You can use the parameter 'step_size' which can be
            # considered the learning rate of the SGLD update.

            # Part 1: Add noise to the input.
            noise.normal_(0, 0.01)
            inp_imgs.data.add_(noise.data).clamp_(-1.0, 1.0)

            # Calculate gradients
            self.model.zero_grad()
            if clabel is not None:
                output = self.model(inp_imgs, clabel)
            else:
                output = self.model(inp_imgs)

            # we are estimating negative energy
            energy = -output.sum()
            energy.backward()

            # Gradient ascent/descent
            inp_imgs.data.add_(-step_size * inp_imgs.grad.data).clamp_(-1.0, 1.0)
            inp_imgs.grad.detach_()
            inp_imgs.grad.zero_()

            if return_img_per_step:
                imgs_per_step.append(inp_imgs.clone().detach())

        # Update class-specific buffer for each image and label

        if clabel is not None:
            updated_images = []
            for img, buffer_info in zip(inp_imgs, buffer_indices):
                if buffer_info is not None:
                    label, buffer_index = buffer_info  # Unpack the tuple
                    # Convert the label to an integer if it's a tensor
                    label = label.item() if isinstance(label, torch.Tensor) else label

                    # Ensure the buffer for this class is initialized
                    if label >= len(self.reservoir_buffer):
                        raise IndexError(f"Class label {label} is out of range for the reservoir buffer.")

                    # Update the buffer for this class
                    # self.reservoir_buffer[label] = torch.cat((self.reservoir_buffer[label], img.unsqueeze(0)), dim=0)[
                                               #-self.cbuffer_size:]
                    # Update the buffer at the specific index for this class
                    self.reservoir_buffer[label][buffer_index] = img.detach()

                # Collect updated images for self.images
                updated_images.append(img.unsqueeze(0))

            # Update self.images with new synthesized images for the specified classes
            #self.images.extend(updated_images)
            # Ensure the overall buffer size doesn't exceed a predefined limit (adjust this limit as needed)
            #self.images = self.images[-8192:]  # Example: Keep only the latest 8192 images
        else:
            # Add new images to the buffer and remove old ones if needed
            self.images = list(inp_imgs.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.images
            self.images = self.images[:8192]

        for p in self.model.parameters():
            p.requires_grad = True
        self.model.train(is_training)

        torch.set_grad_enabled(had_gradients_enabled)

        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return inp_imgs


class JEM(pl.LightningModule):
    def __init__(self, img_shape, batch_size, num_classes=42, cbuffer_size=256, ccond_sample=False, alpha=0.1, lmbd=0.1,
                 lr=1e-4, lr_stepsize=1, lr_gamma=0.97, m_in=0, m_out=-10, steps=60, step_size_decay=1.0, **MODEL_args):
        super().__init__()
        self.save_hyperparameters()

        self.img_shape = img_shape
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.ccond_sample = ccond_sample
        self.cnn = ShallowCNN(**MODEL_args)

        # During training, we want to use the MCMC-based sampler to synthesize images from the current q_\theta and
        # use these in the contrastive loss functional to update the model parameters \theta.
        # (Intuitively, we alternate between sampling from q_\theta and updating q_\theta, which is a quite challenging
        # minmax setting with an adversarial interpretation.)
        self.sampler = MCMCSampler(self.cnn, img_shape=img_shape, sample_size=batch_size, num_classes=num_classes,
                                   cbuffer_size=cbuffer_size)
        self.example_input_array = torch.zeros(1, *img_shape)  # this is used to validate data and model compatability

        # If you want, you can use Torchmetrics to evaluate your classification performance!
        # For example, if we want to populate the metrics after each training step using the predicted logits and
        # classification ground truth y:
        #         self.train_metrics.update(logits, y) --> populate the running metrics buffer
        # We can then log the metrics using on_step=False and on_epoch=True so that they only get computed at the
        # end of each epoch.
        #         self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        # Please refer to the torchmetrics documentation if this process is not clear.
        metrics = torchmetrics.MetricCollection([torchmetrics.CohenKappa(num_classes=num_classes, task='multiclass'),
                                                 torchmetrics.AveragePrecision(num_classes=num_classes,
                                                                               task='multiclass'),
                                                 torchmetrics.AUROC(num_classes=num_classes, task='multiclass'),
                                                 torchmetrics.MatthewsCorrCoef(num_classes=num_classes,
                                                                               task='multiclass'),
                                                 torchmetrics.CalibrationError(task='multiclass',
                                                                               num_classes=num_classes)])
        dyna_metrics = [torchmetrics.Accuracy,
                        torchmetrics.Precision,
                        torchmetrics.Recall,
                        torchmetrics.Specificity,
                        torchmetrics.F1Score]

        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        for mode in ['micro', 'macro']:
            self.train_metrics.add_metrics(
                {f"{mode}_{m.__name__}": m(average=mode, num_classes=num_classes, task='multiclass') for m in
                 dyna_metrics})
            self.valid_metrics.add_metrics(
                {f"{mode}_{m.__name__}": m(average=mode, num_classes=num_classes, task='multiclass') for m in
                 dyna_metrics})

        self.hp_metric = torchmetrics.AveragePrecision(num_classes=num_classes, task='multiclass')

    def forward(self, x, labels=None):
        z = self.cnn(x, labels)
        return z

    def configure_optimizers(self):
        # We typically do not want to have momentum enabled. This is because when training the EBM using alternating
        # steps of synthesis and model update, we constantly shift the energy surface, making it hard to make momentum
        # helpful.
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.0, 0.999))

        # Exponential decay over epochs
        scheduler = optim.lr_scheduler.StepLR(optimizer, self.hparams.lr_stepsize,
                                              gamma=self.hparams.lr_gamma)
        return [optimizer], [scheduler]

    def px_step(self, batch): #ccond_sample=True):
        # TODO (3.4): Implement p(x) step.
        # In addition to calculating the contrastive loss, also consider using an L2 regularization loss. This allows us
        # to constrain the Lipshitz constant by penalizes too large energies and makes sure that the energies maintain
        # similar magnitudes across epochs.
        # E.g.:
        #         reg_loss = self.hparams.alpha * (real_out ** 2 + synth_out ** 2).mean()
        #         cdiv_loss = ...
        #         loss = reg_loss + cdiv_loss

        real_imgs, labels = batch

        if self.ccond_sample:
            synth_imgs = self.sampler.synthesize_samples(clabel=labels)
        else:
            synth_imgs = self.sampler.synthesize_samples()

        real_out = self.forward(real_imgs)
        synth_out = self.forward(synth_imgs)

        # L2 regularization
        reg_loss = self.hparams.alpha * (real_out ** 2 + synth_out ** 2).mean()

        # Contrastive divergence loss
        cdiv_loss = synth_out.mean() - real_out.mean()

        loss = reg_loss + cdiv_loss

        # Logging
        self.log('loss', loss)
        self.log('loss_contrastive_divergence', cdiv_loss)
        self.log('loss_regularization', reg_loss)

        return loss

    def pyx_step(self, batch):
        # TODO (3.4): Implement p(y|x) step.
        # Here, we want to calculate the classification loss using the class logits infered by the neural network.

        imgs, labels = batch
        logits = self.cnn.get_logits(imgs)

        # Classification loss, e.g., cross-entropy
        classification_loss = F.cross_entropy(logits, labels)
        return classification_loss

    def training_step(self, batch, batch_idx):
        # Note: batch_idx just needed due to pytorch lightning
        # TODO (3.4): Implement joint density p(x,y) step using p(x) and p(y|x)
        # Here, we specify the update equation used to tune the model parameters.
        # Ideally, we only need to call the px_step() and pyx_step() methods and combine their loss terms to build up
        # the factorized joint density loss introduced by Gratwohl et al. .
        px_loss = self.px_step(batch)
        pyx_loss = self.pyx_step(batch)

        loss = px_loss + pyx_loss

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=None):
        # Note: batch_idx and dataset_idx not needed (just there for PyTorch
        # Lightning)
        # TODO (3.4)

        real_imgs, _ = batch
        fake_imgs = torch.rand_like(real_imgs) * 2 - 1

        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)

        cdiv = fake_out.mean() - real_out.mean()
        self.log('val_contrastive_divergence', cdiv)


def run_training(args) -> pl.LightningModule:
    """
    Perform EBM/JEM training using a set of hyper-parameters

    Visualization can be either done showcasing different image states during synthesis or by showcasing the
    final results.

    :param args: hyper-parameter
    :return: pl.LightningModule: the trained model
    """
    # Hyper-parameters
    ckpt_dir = args.ckpt_dir
    data_dir = args.data_dir
    num_workers = args.num_workers  # 0 for Windows, can be set higher for linux
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    num_classes = args.num_classes
    lr = args.lr
    lr_stepsize = args.lr_stepsize
    lr_gamma = args.lr_gamma
    alpha = args.alpha
    cbuffer_size = args.cbuffer_size
    ccond_sample = args.ccond_sample

    # Create checkpoint path if it doesn't exist yet
    os.makedirs(ckpt_dir, exist_ok=True)

    # Datasets & Dataloaders
    datasets: Dict[str, TransformTensorDataset] = get_datasets(data_dir)
    train_loader = data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, drop_last=True,
                                   num_workers=num_workers, pin_memory=True)
    val_loader = data.DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, drop_last=False,
                                 num_workers=num_workers)

    trainer = pl.Trainer(default_root_dir=ckpt_dir,
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=num_epochs,
                         gradient_clip_val=0.1,
                         callbacks=[
                             ModelCheckpoint(save_weights_only=True, mode="min", monitor='val_contrastive_divergence',
                                             filename='val_condiv_{epoch}-{step}'),
                             # ModelCheckpoint(save_weights_only=True, mode="max",
                             # monitor='val_MulticlassAveragePrecision',
                             # filename='val_mAP_{epoch}-{step}'),
                             ModelCheckpoint(save_weights_only=True, filename='last_{epoch}-{step}'),
                             LearningRateMonitor("epoch")
                         ])
    pl.seed_everything(42)
    model = JEM(num_epochs=num_epochs,
                img_shape=(1, 56, 56),
                batch_size=batch_size,
                num_classes=num_classes,
                hidden_features=32,  # size of the hidden dimension in the Shallow CNN model
                cbuffer_size=cbuffer_size,  # size of the reservoir for sampling (class-specific)
                ccond_sample=ccond_sample,  # Should we do class-conditional sampling?
                lr=lr,  # General Learning rate
                lr_gamma=lr_gamma,  # Multiplicative factor for exponential learning rate decay
                lr_stepsize=lr_stepsize,  # Step size for exponential learning rate decay
                alpha=alpha,  # L2 regularization of energy terms
                step_size_decay=1.0  # Multiplicative factor for SGLD step size decay)
                )
    trainer.fit(model, train_loader, val_loader)
    model = JEM.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    return model


def run_generation(args, ckpt_path: Union[str, Path], conditional: bool = False):
    """
    With a trained model we can synthesize new examples from q_\theta using SGLD.

    :param args: hyper-parameter
    :param ckpt_path: local path to the trained checkpoint.
    :param conditional: flag to specify if we want to generate conditioned on a specific class label or not
    :return: None
    """
    model = JEM.load_from_checkpoint(ckpt_path)
    model.to(device)
    pl.seed_everything(42)

    def gen_imgs(model, clabel=None, step_size=10, batch_size=24, num_steps=256):
        model.eval()
        torch.set_grad_enabled(True)  # Tracking gradients for sampling necessary
        mcmc_sampler = MCMCSampler(model, model.img_shape, batch_size, model.num_classes)
        img = mcmc_sampler.synthesize_samples(clabel, steps=num_steps, step_size=step_size, return_img_per_step=True)
        torch.set_grad_enabled(False)
        model.train()
        return img

    k = 8
    bs = 8
    num_steps = 256
    conditional_labels = [1, 4, 5, 10, 17, 18, 39, 23]

    synth_imgs = []
    for label in tqdm.tqdm(conditional_labels):
        clabel = (torch.ones(bs) * label).type(torch.LongTensor).to(model.device)
        generated_imgs = gen_imgs(model, clabel=clabel if conditional else None, step_size=10, batch_size=bs,
                                  num_steps=num_steps).cpu()
        synth_imgs.append(generated_imgs[-1])

        # Visualize sampling process
        i = 0
        step_size = num_steps // 8
        imgs_to_plot = generated_imgs[step_size - 1::step_size, i]
        imgs_to_plot = torch.cat([generated_imgs[0:1, i], imgs_to_plot], dim=0)
        grid = torchvision.utils.make_grid(imgs_to_plot, nrow=imgs_to_plot.shape[0], normalize=True,
                                           value_range=(-1, 1), pad_value=0.5, padding=2)
        grid = grid.permute(1, 2, 0)
        plt.figure(figsize=(8, 8))
        plt.imshow(grid)
        plt.xlabel("Generation iteration")
        plt.xticks([(generated_imgs.shape[-1] + 2) * (0.5 + j) for j in range(8 + 1)],
                   labels=[1] + list(range(step_size, generated_imgs.shape[0] + 1, step_size)))
        plt.yticks([])
        plt.savefig(f"{'conditional' if conditional else 'unconditional'}_sample_label={label}.png")

    # Visualize end results
    grid = torchvision.utils.make_grid(torch.cat(synth_imgs), nrow=k, normalize=True, value_range=(-1, 1),
                                       pad_value=0.5,
                                       padding=2)
    grid = grid.permute(1, 2, 0)
    grid = grid[..., 0].numpy()
    plt.figure(figsize=(12, 24))
    plt.imshow(grid, cmap='Greys')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"{'conditional' if conditional else 'unconditional'}_samples.png")


def run_evaluation(args, ckpt_path: Union[str, Path]):
    """
    Evaluate the predictive performance of the JEM model.
    :param args: hyper-parameter
    :param ckpt_path: local path to the trained checkpoint.
    :return: None
    """
    model = JEM.load_from_checkpoint(ckpt_path)
    model.to(device)
    pl.seed_everything(42)

    # Datasets & Dataloaders
    batch_size = args.batch_size
    data_dir = args.data_dir
    num_workers = args.num_workers
    datasets: Dict[str, TransformTensorDataset] = get_datasets(data_dir)

    # Test loader
    test_loader = data.DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, drop_last=False,
                                  num_workers=num_workers)

    trainer = pl.Trainer()  # gpus=1 if str(device).startswith("cuda") else 0)
    results = trainer.validate(model, dataloaders=test_loader)
    print(results)
    return results


def run_ood_analysis(args, ckpt_path: Union[str, Path]):
    """
    Run out-of-distribution (OOD) analysis. First, you evaluate the scores for the training samples (in-distribution),
    a random noise distribution, and two different distributions that share some resemblence with the training data.

    :param args: hyper-parameter
    :param ckpt_path: local path to the trained checkpoint.
    :return: None
    """
    model = JEM.load_from_checkpoint(ckpt_path)
    model.to(device)
    pl.seed_everything(42)

    # Datasets & Dataloaders
    batch_size = args.batch_size
    data_dir = args.data_dir
    num_workers = args.num_workers
    datasets: Dict[str, TransformTensorDataset] = get_datasets(data_dir)

    # Test loader
    test_loader = data.DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, drop_last=False,
                                  num_workers=num_workers)
    # OOD loaders for OOD types a and b
    ood_ta_loader = data.DataLoader(datasets['ood_ta'], batch_size=batch_size, shuffle=False, drop_last=False,
                                    num_workers=num_workers)
    ood_tb_loader = data.DataLoader(datasets['ood_tb'], batch_size=batch_size, shuffle=False, drop_last=False,
                                    num_workers=num_workers)

    # TODO (3.6): Calculate and visualize the score distributions, e.g. with a histogram. Analyze whether we can
    #  visualy tell apart the different data distributions based on their assigned score.

    # TODO (3.6): Solve a binary classification on the soft scores and evaluate and AUROC and/or AUPRC score for
    #  discrimination between the training samples and one of the OOD distributions.

    # Implement OOD score calculation and visualization
    scores_in, scores_out_a, scores_out_b = [], [], []
    for batch in test_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        scores_in.append(score_fn(model, x, y, score="py"))

    for batch in ood_ta_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        scores_out_a.append(score_fn(model, x, y, score="py"))

    for batch in ood_tb_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        scores_out_b.append(score_fn(model, x, y, score="py"))

    scores_in = torch.cat(scores_in).cpu().numpy()  # Convert to NumPy array
    scores_out_a = torch.cat(scores_out_a).cpu().numpy()  # Convert to NumPy array
    scores_out_b = torch.cat(scores_out_b).cpu().numpy()  # Convert to NumPy array

    # Plot histograms
    plt.figure(figsize=(10, 6))
    sns.histplot(scores_in, color='blue', label='In-distribution')
    sns.histplot(scores_out_a, color='red', label='OOD Type A')
    sns.histplot(scores_out_b, color='green', label='OOD Type B')
    plt.legend()
    plt.show()

    # Concatenate scores for AUROC and AUPRC calculation
    y_scores = np.concatenate([scores_in, scores_out_a, scores_out_b])

    # Binary classification labels: 1 for in-distribution, 0 for OOD
    y_true = [1] * len(scores_in) + [0] * (len(scores_out_a) + len(scores_out_b))

    # Calculate AUROC and AUPRC
    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)

    print("AUROC:", auroc)
    print("AUPRC:", auprc)



def visualize_data(data_dir: str) -> None:
    """

    :param data_dir:
    :return:
    """

    datasets: Dict[str, TransformTensorDataset] = get_datasets(data_dir)
    train_loader = data.DataLoader(datasets['train'], batch_size=8, shuffle=True, drop_last=True,
                                   num_workers=0, pin_memory=True)
    val_loader = data.DataLoader(datasets['val'], batch_size=8, shuffle=False, drop_last=False,
                                 num_workers=0)
    test_loader = data.DataLoader(datasets['test'], batch_size=8, shuffle=False, drop_last=False,
                                  num_workers=0)
    ood_data_loader = data.DataLoader(datasets['ood_ta'], batch_size=8, shuffle=False, drop_last=False,
                                      num_workers=0)

    def show_images(images, row_title, row_index):
        for i, img in enumerate(images):
            ax = plt.subplot(4, 8, i + 1 + 8 * row_index)
            ax.axis('off')
            ax.set_title(row_title if i == 0 else "")

            # Convert to grayscale if the image has 3 channels
            if img.shape[0] == 3:
                img = transforms.Grayscale(num_output_channels=1)(img)

            img = img.numpy().squeeze()  # Remove channel dimension for grayscale and convert to numpy array
            plt.imshow(img, cmap='gray')

    # Assuming the datasets are image datasets
    fig = plt.figure(figsize=(25, 10))

    # Extract and show 8 images from each loader in grayscale
    for row_index, (loader, row_title) in enumerate(zip([train_loader, val_loader, test_loader, ood_data_loader],
                                                        ['Train', 'Validation', 'Test', 'OOD'])):
        images, _ = next(iter(loader))
        show_images(images, row_title, row_index)

    plt.show()


if __name__ == '__main__':
    args = parse_args()

    # 0) TODO: Visualize data
    # visualize_data(data_dir=args.data_dir)

    # 1) Run training
    # run_training(args)

    # 2) Evaluate model
    # V18 unconditional final
    ckpt_path: str = "/proj/ciptmp/af23aduk/adl_ex03/saved_models//lightning_logs/version_18/checkpoints/last_epoch=19-step=7060.ckpt"
    # V19 conditional final
    # ckpt_path: str = "/proj/ciptmp/af23aduk/adl_ex03/saved_models//lightning_logs/version_19/checkpoints/last_epoch=21-step=7766.ckpt"

    # Classification performance
    run_evaluation(args, ckpt_path)

    # Image synthesis
    run_generation(args, ckpt_path, conditional=False)
    # run_generation(args, ckpt_path, conditional=True)

    # OOD Analysis
    run_ood_analysis(args, ckpt_path)
