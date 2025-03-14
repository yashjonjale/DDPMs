import torch
import torch.utils
import torch.utils.data
from tqdm.auto import tqdm
from torch import nn
import argparse
import torch.nn.functional as F
import utils
import dataset
import os
import matplotlib.pyplot as plt
import math


# SinusoidalPositionEmbeddings
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim # embedd dimention 128 ... 

    def forward(self, time): # input of shape (64)
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class NoiseScheduler():
    """
    Noise scheduler for the DDPM model

    Args:
        num_timesteps: int, the number of timesteps
        type: str, the type of scheduler to use
        **kwargs: additional arguments for the scheduler

    This object sets up all the constants like alpha, beta, sigma, etc. required for the DDPM model
    
    """
    def __init__(self, num_timesteps=50, type="linear", **kwargs):

        self.num_timesteps = num_timesteps
        self.type = type

        if type == "linear":
            self.init_linear_schedule(**kwargs)
        else:
            raise NotImplementedError(f"{type} scheduler is not implemented") # change this if you implement additional schedulers


    def init_linear_schedule(self, beta_start, beta_end):
        """
        Precompute whatever quantities are required for training and sampling
        """

        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float32)

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0] , dtype=torch.float32), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def add_noise(self, x, t, noise):
        # Index the scaling factors for each sample in the batch
        # t: tensor of shape [batch_size] # [64] -> 
        # x is 64*2 
        # noise is 64*2
        # sqrt_alphas_cumprod is  of shape T 
        # sqrt_alphas is of shape 64 .. make is of shape 64 * 1 .. by unsqueeeze for bradcasting to 64 * 2
        sqrt_alphas = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sqrt_alphas * x + sqrt_one_minus * noise

    def __len__(self):
        return self.num_timesteps
    
class DDPM(nn.Module):
    def __init__(self, n_dim=3, n_steps=200):
        """
        Noise prediction network for the DDPM

        Args:
            n_dim: int, the dimensionality of the data
            n_steps: int, the number of steps in the diffusion process
        We have separate learnable modules for `time_embed` and `model`. `time_embed` can be learned or a fixed function as well

        """
        super().__init__()
        self.n_dim = n_dim
        self.n_steps = n_steps
        self.embed_dim = 128 # hyper parameter 

        # Sinusoidal embedding for timesteps
        self.time_embed = SinusoidalPositionEmbeddings(dim=self.embed_dim)
        # Optional projection layers to refine the embedding
        self.time_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

        # Noise prediction model (increased capacity)
        self.model = nn.Sequential(
            nn.Linear(n_dim + self.embed_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, n_dim)
        )
        

    def forward(self, x, t): #  inputs is of shape (64,2) , (64)
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
            t: torch.Tensor, the timestep tensor [batch_size]

        Returns:
            torch.Tensor, the predicted noise tensor [batch_size, n_dim]
        """
        # Get sinusoidal embeddings for timesteps
        time_embed = self.time_embed(t)  # [batch_size, embed_dim] # 64 * 128
        time_embed = self.time_proj(time_embed)  # [batch_size, embed_dim] # 64 * 128
        x_t = torch.cat([x, time_embed], dim=1)  # [batch_size, n_dim + embed_dim] # (64+2)*128
        return self.model(x_t) 

class ConditionalDDPM():
    pass
    
class ClassifierDDPM():
    """
    ClassifierDDPM implements a classification algorithm using the DDPM model
    """
    
    def __init__(self, model: ConditionalDDPM, noise_scheduler: NoiseScheduler):
        pass

    def __call__(self, x):
        pass

    def predict(self, x):
        pass

    def predict_proba(self, x):
        pass

def train(model, noise_scheduler, dataloader, optimizer, epochs, run_name):
    """
    Train the model and save the model and necessary plots

    Args:
        model: DDPM, model to train
        noise_scheduler: NoiseScheduler, scheduler for the noise
        dataloader: torch.utils.data.DataLoader, dataloader for the dataset
        optimizer: torch.optim.Optimizer, optimizer to use
        epochs: int, number of epochs to train the model
        run_name: str, path to save the model
    """
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for batch in dataloader: # batch ( x , y )
            x, y = batch # x.shape is 64*2 ... y is null
            t = torch.randint(0, noise_scheduler.num_timesteps, (x.shape[0],), device=x.device)
            # t.shape is 64 .. 
            noise = torch.randn_like(x)
            x_t = noise_scheduler.add_noise(x, t, noise) #
            pred_noise = model(x_t, t)  # inputs is of shape (64,2) , (64)
            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
    plt.plot(losses)
    print(" Loss graph saved ! " )
    plt.savefig(f'{run_name}/losses.png')
    plt.close()


def reverse_step(xt, eps, t, noise_scheduler):  # ( 1000*2 , 1000*2 , 1 )
    alpha_t = noise_scheduler.alphas[t] # alpha_t of shape 1
    alpha_cumprod_t = noise_scheduler.alphas_cumprod[t] # of shape 1
    beta_t = noise_scheduler.betas[t] # of shape 1
    
    mu = (xt - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * eps) / torch.sqrt(alpha_t)
    if t > 0:
        z = torch.randn_like(xt)
    else:
        z = 0
    return mu + torch.sqrt(beta_t) * z

@torch.no_grad()
def sample(model, n_samples, noise_scheduler, return_intermediate=False): # n_smaples = 1000 in our case 
    """
    Sample from the model
    
    Args:
        model: DDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        return_intermediate: bool
    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]

    If `return_intermediate` is `False`,
            torch.Tensor, samples from the model [n_samples, n_dim]
    Else
        the function returns all the intermediate steps in the diffusion process as well 
        Return: [[n_samples, n_dim]] x n_steps
        Optionally implement return_intermediate=True, will aid in visualizing the intermediate steps
    """   
    device = next(model.parameters()).device
    x = torch.randn(n_samples, model.n_dim, device=device) # 1000 * 2  .. this is X_T
    if return_intermediate:
        intermediates = [x]
    for t in reversed(range(noise_scheduler.num_timesteps)): # from T-1 to 0 .. 
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long) # full( shape , fill_value )
        pred_noise = model(x, t_tensor) # shape ( 1000*2  , 1000 ) # noise_pred of shape 1000*2
        x = reverse_step(x, pred_noise, t, noise_scheduler) # ( 1000*2 , 1000*2 , 1 )
        if return_intermediate:
            intermediates.append(x)
    return intermediates if return_intermediate else x
    

def sampleCFG(model, n_samples, noise_scheduler, guidance_scale, class_label):
    """
    Sample from the conditional model
    
    Args:
        model: ConditionalDDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        guidance_scale: float
        class_label: int

    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]
    """
    pass

def sampleSVDD(model, n_samples, noise_scheduler, reward_scale, reward_fn):
    """
    Sample from the SVDD model

    Args:
        model: ConditionalDDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        reward_scale: float
        reward_fn: callable, takes in a batch of samples torch.Tensor:[n_samples, n_dim] and returns torch.Tensor[n_samples]

    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]
    """
    pass
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'sample' , 'study'], default='sample')
    parser.add_argument("--n_steps", type=int, default=None) # this is T // number of diffusion steps 
    # we are supposed to check for T=10,50,100,150,200
    parser.add_argument("--lbeta", type=float, default=None) # 0.0001
    parser.add_argument("--ubeta", type=float, default=None) # 0.02
    parser.add_argument("--epochs", type=int, default=None) # 100 epoch 
    parser.add_argument("--n_samples", type=int, default=None) # number of samples we wanna generate .. say 1000
    parser.add_argument("--batch_size", type=int, default=None) # 64 
    parser.add_argument("--lr", type=float, default=None) # 0.001 ... for now 
    parser.add_argument("--dataset", type=str, default = None) # do for moon first [8000 , 2]
    parser.add_argument("--seed", type=int, default = 42)
    parser.add_argument("--n_dim", type=int, default = None) # dimention of data .. # ( 2 for moon and 64 for albestros )

    # take the args from the command line

    args = parser.parse_args()
    utils.seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_name = f'exps/ddpm_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}_{args.dataset}' # can include more hyperparams
    os.makedirs(run_name, exist_ok=True) # create a directory for plots and models 

    model = DDPM(n_dim=args.n_dim, n_steps=args.n_steps) # 2 , 200
    noise_scheduler = NoiseScheduler(num_timesteps=args.n_steps, beta_start=args.lbeta, beta_end=args.ubeta) # 200 , 0.0001 , 0.02
    model = model.to(device)

    if args.mode == 'train':
        epochs = args.epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        data_X, data_y = dataset.load_dataset(args.dataset)
        # can split the data into train and test -- for evaluation later
        data_X = data_X.to(device)
        data_y = data_y.to(device)
        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_X, data_y), batch_size=args.batch_size, shuffle=True)
        train(model, noise_scheduler, dataloader, optimizer, epochs, run_name)

        torch.save(model.state_dict(), f'{run_name}/model.pth') # ADDED THIS LINE

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(f'{run_name}/model.pth'))
        samples = sample(model, args.n_samples, noise_scheduler)  # n_samples is 1000
        torch.save(samples, f'{run_name}/samples_{args.seed}_{args.n_samples}.pth')

    elif args.mode == 'study':
        model.load_state_dict(torch.load(f'{run_name}/model.pth'))
        samples = sample(model, args.n_samples, noise_scheduler).cpu()
        torch.save(samples, f'{run_name}/samples_{args.seed}_{args.n_samples}.pth')
        print(f"Samples saved to {run_name}/samples_{args.seed}_{args.n_samples}.pth")

        # Load true data for evaluation
        data_X, _ = dataset.load_dataset(args.dataset)
        data_X = data_X.cpu()  # Move to CPU for evaluation

        # Convert samples and true data to required formats
        samples_np = samples.numpy()  # For get_emd
        data_X_np = data_X.numpy()   # For get_emd
        samples_torch = samples       # For get_nll (already torch tensor)
        data_X_torch = data_X         # For get_nll (already torch tensor)

        # Compute metrics
        emd = utils.get_emd(samples_np, data_X_np)
        nll = utils.get_nll(data_X_torch, samples_torch, temperature=1e-1)
        print(f"EMD: {emd:.4f}, NLL: {nll:.4f}")

        # Save metrics
        with open(f'{run_name}/metrics.txt', 'w') as f:
            f.write(f"EMD: {emd:.4f}\nNLL: {nll:.4f}\n")

        # Visualize samples vs true data
        plt.scatter(samples_np[:, 0], samples_np[:, 1], s=5, alpha=0.5, label='Generated')
        plt.scatter(data_X_np[:args.n_samples, 0], data_X_np[:args.n_samples, 1], s=5, alpha=0.5, label='True')
        plt.legend()
        plt.title(f'Samples (T={args.n_steps}, beta={args.lbeta}-{args.ubeta})')
        plt.savefig(f'{run_name}/samples_plot.png')
        plt.close()


    else:
        raise ValueError(f"Invalid mode {args.mode}")