import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import numpy as np
from collections import OrderedDict
from copy import deepcopy

from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from torch.utils.data import Dataset


class DiffusionPriorUNet(nn.Module):

    def __init__(
            self, 
            embed_dim=1024, 
            cond_dim=42,
            hidden_dim=[1024, 512, 256, 128, 64],
            time_embed_dim=512,
            act_fn=nn.SiLU,
            dropout=0.0,
        ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim

        # 1. time embedding
        self.time_proj = Timesteps(time_embed_dim, True, 0)

        # 2. conditional embedding 
        # to 3.2, 3,3

        # 3. prior mlp

        # 3.1 input
        self.input_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim[0]),
            nn.LayerNorm(hidden_dim[0]),
            act_fn(),
        )

        # 3.2 hidden encoder
        self.num_layers = len(hidden_dim)
        self.encode_time_embedding = nn.ModuleList(
            [TimestepEmbedding(
                time_embed_dim,
                hidden_dim[i],
            ) for i in range(self.num_layers-1)]
        ) # d_0, ..., d_{n-1}
        self.encode_cond_embedding = nn.ModuleList(
            [nn.Linear(cond_dim, hidden_dim[i]) for i in range(self.num_layers-1)]
        )
        self.encode_layers = nn.ModuleList(
            [nn.Sequential(
                    nn.Linear(hidden_dim[i], hidden_dim[i+1]),
                    nn.LayerNorm(hidden_dim[i+1]),
                    act_fn(),
                    nn.Dropout(dropout),
                ) for i in range(self.num_layers-1)]
        )

        # 3.3 hidden decoder
        self.decode_time_embedding = nn.ModuleList(
            [TimestepEmbedding(
                time_embed_dim,
                hidden_dim[i],
            ) for i in range(self.num_layers-1,0,-1)]
        ) # d_{n}, ..., d_1
        self.decode_cond_embedding = nn.ModuleList(
            [nn.Linear(cond_dim, hidden_dim[i]) for i in range(self.num_layers-1,0,-1)]
        )
        self.decode_layers = nn.ModuleList(
            [nn.Sequential(
                    nn.Linear(hidden_dim[i], hidden_dim[i-1]),
                    nn.LayerNorm(hidden_dim[i-1]),
                    act_fn(),
                    nn.Dropout(dropout),
                ) for i in range(self.num_layers-1,0,-1)]
        )

        # 3.4 output
        self.output_layer = nn.Linear(hidden_dim[0], embed_dim)
        

    def forward(self, x, t, c=None):
        # x (batch_size, embed_dim)
        # t (batch_size, )
        # c (batch_size, cond_dim)

        # 1. time embedding
        t = self.time_proj(t) # (batch_size, time_embed_dim)

        # 2. conditional embedding 
        # to 3.2, 3.3

        # 3. prior mlp

        # 3.1 input
        x = self.input_layer(x) 
        # print("x.shape:", x.shape, "t:", t.shape, "c", c.shape)
        # 3.2 hidden encoder
        hidden_activations = []
        for i in range(self.num_layers-1):
            hidden_activations.append(x)
            t_emb = self.encode_time_embedding[i](t) 
            if c is not None:
                c_emb = self.encode_cond_embedding[i](c)
            else:
                # Adjust the zero tensor size to match the expected dimensions for the current layer
                c_emb = torch.zeros(x.size(0), self.hidden_dim[i], device=x.device)

            # c_emb = self.decode_cond_embedding[i](c) if c is not None else torch.zeros(x.size(0), self.hidden_dim, device=x.device)
            # print("x.shape:", x.shape, "t_emb.shape:", t_emb.shape, "c_emb.shape:", c_emb.shape)
            x = x + t_emb + c_emb
            x = self.encode_layers[i](x)
        
        # 3.3 hidden decoder
        for i in range(self.num_layers-1):
            t_emb = self.decode_time_embedding[i](t)
            if c is not None:
                c_emb = self.decode_cond_embedding[i](c)
            else:
                # Adjust the zero tensor size to match the expected dimensions for the current layer
                c_emb = torch.zeros_like(x)
            # c_emb = self.decode_cond_embedding[i](c) if c is not None else torch.zeros(x.size(0), self.hidden_dim, device=x.device)
            # print("x.shape:", x.shape, "t_emb.shape:", t_emb.shape, "c_emb.shape:", c_emb.shape)
            x = x + t_emb + c_emb
            x = self.decode_layers[i](x)
            x += hidden_activations[-1-i]

        # 3.4 output
        x = self.output_layer(x)

        return x

    
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def add_noise_with_snr(
    self,
    original_samples: torch.FloatTensor,
    noise: torch.FloatTensor,
    timesteps: torch.IntTensor,
) -> torch.FloatTensor:
    # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
    # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
    # for the subsequent add_noise calls
    self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
    alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
    timesteps = timesteps.to(original_samples.device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    snr = sqrt_alpha_prod ** 2 / sqrt_one_minus_alpha_prod ** 2
    return noisy_samples, snr


def Soft_Min_SNR(snr, gamma=4):
    # Soft-Min-SNR Loss Weighting
    # https://arxiv.org/pdf/2401.11605.pdf
    # sigsnrma: tensor(batch_size, )
    weight = 1 / (1 / snr + 1 / gamma)
    return weight

# diffusion pipe
class Pipe:
    
    def __init__(self, diffusion_prior=None, scheduler=None, device='cuda'):
        self.diffusion_prior = diffusion_prior.to(device)
        self.ema = deepcopy(self.diffusion_prior).to(device)  # Create an EMA of the model for use after training
        requires_grad(self.ema, False)
        update_ema(self.ema, self.diffusion_prior, decay=0)  # Ensure EMA is initialized with synced weights
        self.ema.eval()  # EMA model should always be in eval mode

        if scheduler is None:
            from diffusers.schedulers import DDPMScheduler
            self.scheduler = DDPMScheduler() 
            self.scheduler.add_noise_with_snr = add_noise_with_snr.__get__(self.scheduler)
        else:
            self.scheduler = scheduler
            
        self.device = device
        
    def train(self, dataloader, num_epochs=10, learning_rate=1e-4, uncondition_rate=0.1, SNR_weighted=False):

        device = self.device
        self.diffusion_prior.train()

        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(self.diffusion_prior.parameters(), lr=learning_rate)
        from diffusers.optimization import get_cosine_schedule_with_warmup
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=(len(dataloader) * num_epochs),
        )

        num_train_timesteps = self.scheduler.config.num_train_timesteps

        for epoch in range(num_epochs):
            loss_sum = 0
            for batch in dataloader:
                c_embeds = batch['c_embedding'].to(device)
                h_embeds = batch['h_embedding'].to(device)
                N = h_embeds.shape[0]

                # 1. randomly replecing c_embeds to None
                if torch.rand(1) < uncondition_rate:
                    c_embeds = None

                # 2. Generate noisy embeddings as input
                noise = torch.randn_like(h_embeds)

                # 3. sample timestep
                timesteps = torch.randint(0, num_train_timesteps, (N,), device=device)

                # 4. add noise to h_embedding
                perturbed_h_embeds, snr = self.scheduler.add_noise_with_snr(
                    h_embeds,
                    noise,
                    timesteps
                ) # (batch_size, embed_dim), (batch_size, )

                # 5. predict noise
                noise_pre = self.diffusion_prior(perturbed_h_embeds, timesteps, c_embeds)
                
                # 6. loss function weighted by sigma
                loss = criterion(noise_pre, noise) # (batch_size,)
                if SNR_weighted:
                    loss = (Soft_Min_SNR(snr) * loss).mean()
                else:
                    loss = loss.mean()
                            
                # 7. update parameters
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.diffusion_prior.parameters(), 1.0)

                lr_scheduler.step()
                optimizer.step()
                update_ema(self.ema, self.diffusion_prior)

                loss_sum += loss.item()

            loss_epoch = loss_sum / len(dataloader)
            print(f'epoch: {epoch}, loss: {loss_epoch}, lr: {optimizer.param_groups[0]["lr"]}')
            # lr_scheduler.step(loss)

    @torch.no_grad()
    def generate(
            self, 
            c_embeds=None, 
            num_inference_steps=50, 
            timesteps=None,
            guidance_scale=5.0,
            generator=None,
            shape=None,
            N=1,
            use_ema=False,
        ):
        # c_embeds (n_cond, )
        model = self.ema if use_ema else self.diffusion_prior
        model.eval()

        # 1. Prepare timesteps
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, self.device, timesteps)

        # 2. Prepare c_embeds
        if c_embeds is not None:
            c_embeds = c_embeds.to(self.device)
            c_embeds = c_embeds.repeat_interleave(N, dim=0) # (n_cond*N, )
            N = c_embeds.shape[0] # n_cond * N

        # 3. Prepare noise
        if shape is None:
            shape = (self.diffusion_prior.embed_dim, )
        h_t = torch.randn(N, *shape, generator=generator, device=self.device)
        # scale the initial noise by the standard deviation required by the scheduler
        h_t = h_t * self.scheduler.init_noise_sigma

        # 4. denoising loop
        for _, t in tqdm(enumerate(timesteps)):
            t_tensor = torch.ones(N, dtype=torch.float, device=self.device) * t
            # 4.1 noise prediction
            if guidance_scale == 0 or c_embeds is None:
                noise_pred = model(h_t, t_tensor)
            else:
                noise_pred_cond = model(h_t, t_tensor, c_embeds)
                noise_pred_uncond = model(h_t, t_tensor)
                # perform classifier-free guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # 4.2 compute the previous noisy sample h_t -> h_{t-1}
            h_t = self.scheduler.step(noise_pred, t, h_t, generator=generator).prev_sample
        
        return h_t