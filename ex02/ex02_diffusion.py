import torch
import torch.nn.functional as F
from ex02_helpers import extract
from tqdm import tqdm

# The main problem with the linear scheduler is for small images.
# The image is not far from pure Gaussian noise too early in the diffusing process,
# which may make it hard for the model to learn the reverse process. Essentially, noise is being added too fast.

def linear_beta_schedule(beta_start, beta_end, timesteps):
    """
    standard linear beta/variance schedule as proposed in the original paper
    """
    return torch.linspace(beta_start, beta_end, timesteps)


# TODO: Transform into task for students
# The cosine scheduler adds noise slower to retain image information for later timesteps.
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    # TODO (2.3): Implement cosine beta/variance schedule as discussed in the paper mentioned above

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def sigmoid_beta_schedule(beta_start, beta_end, timesteps):
    """
    sigmoidal beta schedule - following a sigmoid function
    """
    # TODO (2.3): Implement a sigmoidal beta schedule. Note: identify suitable limits of where you want to sample the sigmoid function.
    # Note that it saturates fairly fast for values -x << 0 << +x

    s_limit = 5

    betas = torch.linspace(-s_limit, s_limit, timesteps)
    return beta_start + torch.sigmoid(betas) * (beta_end - beta_start)


class Diffusion:

    # TODO (2.4): Adapt all methods in this class for the conditional case. You can use y=None to encode that you want to train the model fully unconditionally.

    def __init__(self, timesteps, get_noise_schedule, img_size, device="cuda"):
        """
        Takes the number of noising steps, a function for generating a noise schedule as well as the image size as input.
        """
        self.timesteps = timesteps

        self.img_size = img_size
        self.device = device

        # define beta schedule
        self.betas = get_noise_schedule(self.timesteps)

        print(self.betas.shape)

        # TODO (2.2): Compute the central values for the equation in the forward pass already here so you can quickly use them in the forward pass.
        # Note that the function torch.cumprod may be of help
        # torch.cumprod(input, dim, *, dtype=None, out=None) → Tensor
        #   Returns the cumulative product of elements of input in the dimension dim.
        # define alphas
        # TODO
        self.alphas = 1. - self.betas
        self.alphas_bar_prod = torch.cumprod(self.alphas, dim=0)

        # It is common to pad sequences for various reasons, such as preparing data for
        # convolutional operations or recurrent networks.
        self.alphas_bar_prod_prev = F.pad(self.alphas_bar_prod[:-1], (1, 0), value=1.0)

        # For denoising preocess
        self.alphas_sqrt_recip = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # TODO
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_bar_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_bar_prod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # TODO

        # The last term for sqrt(beta)* z is zero
        self.posterior_variance = self.betas * (1. - self.alphas_bar_prod_prev) / (1. - self.alphas_bar_prod)


    # for samplig (denoising process)
    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, class_label=None, w=0.5):
        # TODO (2.2): implement the reverse diffusion process of the model for (noisy) samples x and timesteps t. Note that x and t both have a batch dimension

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean

        # TODO (2.2): The method should return the image at timestep t-1.
        betas_t = extract(self.betas, t, x.shape)
        one_minus_alphas_bar_sqrt_t = extract(
            self.one_minus_alphas_bar_sqrt, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.alphas_sqrt_recip, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean

        model_prediction = model(x, t)

        if class_label is not None:
            # model prediction with the conditional guidance
            model_prediction_guid = model(x, t, class_label)
            # combined model prediction
            model_prediction = (1 + w) * model_prediction_guid - w * model_prediction

        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model_prediction / one_minus_alphas_bar_sqrt_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:

            # Apply the conditional guidance according to Equation 11
            #conditional_mean = (1 + w) * model_mean - w * x

            # Algorithm 2 line 4:
            # return conditional_mean + torch.sqrt(posterior_variance_t) * noise
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3, class_labels=None):

        # TODO (2.2): Implement the full reverse diffusion loop from random noise to an image, iteratively ''reducing'' the noise in the generated image.
        device = next(model.parameters()).device

        # define the shape
        shape = (batch_size, channels, image_size, image_size)

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        class_labels = class_labels.to(device) if class_labels is not None else None

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, class_labels)

        # TODO (2.2): Return the generated images
        return img

    # forward diffusion (using the nice property)
    def q_sample(self, x_zero, t, noise=None):
        # TODO (2.2): Implement the forward diffusion process using the beta-schedule defined in the constructor; if noise is None, you will need to create a new noise vector, otherwise use the provided one.

        # generate random gaussian noise
        if noise is None:
            noise = torch.randn_like(x_zero)


        #It extracts values from the tensor self.alphas_bar_sqrt at the indices specified in the index tensor t.
        # The result is a tensor representing the square root of the product of alpha values up to
        # time t for each element in the batch.
        # Basically implementation of equation 5

        sqrt_alphas_prod_t = extract(self.alphas_bar_sqrt, t, x_zero.shape)
        sqrt_one_minus_alphas_prod_t = extract(
            self.one_minus_alphas_bar_sqrt, t, x_zero.shape
        )

        return sqrt_alphas_prod_t * x_zero + sqrt_one_minus_alphas_prod_t * noise

    def p_losses(self, denoise_model, x_zero, t, noise=None, class_labels=None, loss_type="l1"):
        # TODO (2.2): compute the input to the network using the forward diffusion process and predict the noise using the model; if noise is None, you will need to create a new noise vector, otherwise use the provided one.
        if noise is None:
            noise = torch.randn_like(x_zero)

        x_noisy = self.q_sample(x_zero=x_zero, t=t, noise=noise)
        model_prediction = denoise_model(x_noisy, t, class_labels)

        if loss_type == 'l1':
            # TODO (2.2): implement an L1 loss for this task
            loss = F.l1_loss(noise, model_prediction)
        elif loss_type == 'l2':
            # TODO (2.2): implement an L2 loss for this task
            loss = F.mse_loss(noise, model_prediction)
        else:
            raise NotImplementedError()

        return loss
