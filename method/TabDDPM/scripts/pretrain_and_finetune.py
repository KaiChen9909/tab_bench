import torch 
import os 
import random
import math
import warnings
import sys
target_path="./"
sys.path.append(target_path)

import torch.nn as nn
from opacus import PrivacyEngine
from method.TabDDPM.model.modules import MLPDiffusion
from method.TabDDPM.model.diffusion import GaussianMultinomialDiffusion
from method.TabDDPM.data.dataset import *
from method.TabDDPM.scripts.noise import get_noise_multiplier
torch.set_default_dtype(torch.float32) 

def get_model(
        model_params
):
    return MLPDiffusion(**model_params)

# def update_ema(target_params, source_params, rate=0.999):
#     """
#     Update target parameters to be closer to those of source parameters using
#     an exponential moving average.
#     :param target_params: the target parameter sequence.
#     :param source_params: the source parameter sequence.
#     :param rate: the EMA rate (closer to 1 means slower).
#     """
#     for targ, src in zip(target_params, source_params):
#         targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)

class Trainer:
    def __init__( 
            self, diffusion, dataloader, lr, weight_decay, steps, 
            device=torch.device('cuda:0'), 
            dp_epsilon = None, dp_delta = None, rho_used = None,
            report_every=False
    ):
        self.diffusion = diffusion
        self.report_every = report_every #if dump loss every epoch
        self.ema_model = deepcopy(self.diffusion)
        for param in self.ema_model._denoise_fn.parameters():
            param.detach_()

        self.dataloader = dataloader
        self.n_epoch = steps
        self.init_lr = lr
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])
        self.log_every = 10
        self.print_every = 10
        self.ema_every = 10
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.rho_used = rho_used

        if self.dp_epsilon is not None:
            privacy_engine = PrivacyEngine(accountant='rdp')
            
            if self.rho_used > 0:
                account_history = (np.sqrt(1/(2*self.rho_used)), 1.0, 1)
                alpha_history = list(10000 + 10000 * np.arange(100)/100)
            else:
                account_history = None 
                alpha_history = None
            
            self.diffusion._denoise_fn, self.optimizer, self.dataloader = privacy_engine.make_private(
                module = self.diffusion._denoise_fn,
                optimizer = torch.optim.SGD(self.diffusion._denoise_fn.parameters(), lr=lr, weight_decay=weight_decay, momentum = 0.8),
                data_loader = dataloader,
                noise_multiplier = get_noise_multiplier(
                    target_epsilon = self.dp_epsilon,
                    target_delta = self.dp_delta,
                    sample_rate = 1/len(dataloader),
                    epochs=self.n_epoch,
                    accountant="rdp",
                    account_history = account_history,
                    alpha_history = alpha_history
                ),
                max_grad_norm = 1.0,
                batch_first = True,
                loss_reduction = "mean",
                noise_generator = None,
                grad_sample_mode = "hooks",
                poisson_sampling = True,
                clipping="flat"
            )
        else:
            self.optimizer = torch.optim.Adam(self.diffusion._denoise_fn.parameters(), lr=lr, weight_decay=weight_decay)

    def _anneal_lr(self, step):
        frac_done = step / self.n_epoch
        lr = self.init_lr * (1 - np.exp(frac_done - 1))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        
        with warnings.catch_warnings(): 
            warnings.simplefilter('ignore') #ignore some unnecessary warning output from opacus, see https://github.com/pytorch/opacus/issues/328
            self.optimizer.zero_grad()
            loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
            loss = loss_multi + loss_gauss
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.diffusion._denoise_fn.parameters(), 1.0)
            self.optimizer.step()

        del x, out_dict
        torch.cuda.empty_cache()
        return loss_multi, loss_gauss

    def run_loop(self):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        # torch.autograd.set_detect_anomaly(True) 

        while step < self.n_epoch:
            for x, out_dict in self.dataloader:
                out_dict = {'y': out_dict}
                batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

                curr_count += len(x)
                curr_loss_multi += batch_loss_multi.item() * len(x)
                curr_loss_gauss += batch_loss_gauss.item() * len(x)
                
            self._anneal_lr(step)

            # report loss
            if self.report_every:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (step + 1) % self.print_every == 0 or step <= 9:
                    print(f'Epoch {(step + 1)}/{self.n_epoch} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}')
                self.loss_history.loc[len(self.loss_history)] =[step + 1, mloss, gloss, mloss + gloss]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0
            else:
                if (step + 1) % self.log_every == 0 or step <= 9:
                    mloss = np.around(curr_loss_multi / curr_count, 4)
                    gloss = np.around(curr_loss_gauss / curr_count, 4)
                    if (step + 1) % self.print_every == 0 or step <= 9:
                        print(f'Epoch {(step + 1)}/{self.n_epoch} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}')
                    self.loss_history.loc[len(self.loss_history)] =[step + 1, mloss, gloss, mloss + gloss]
                    curr_count = 0
                    curr_loss_gauss = 0.0
                    curr_loss_multi = 0.0

            # update_ema(self.ema_model._denoise_fn.parameters(), self.diffusion._denoise_fn.parameters())

            step += 1



def pretrain(
        parent_dir,
        dataset = None,
        steps = 1000,
        lr = 0.002,
        weight_decay = 1e-4,
        batch_size = 1024,
        model_type = 'mlp',
        model_params = None,
        num_timesteps = 1000,
        gaussian_loss_type = 'mse',
        scheduler = 'cosine',
        T_dict = None,
        num_numerical_features = 0,
        device = torch.device('cuda:0'),
        seed = 42,
        report_every=False,
        change_val = False
): 
    random.seed(seed)
    pretrain_loader = prepare_torch_dataloader(dataset, split='pretrain', batch_size=batch_size)

    cat_size = np.array(dataset.get_category_sizes('train'))
    if len(cat_size) == 0 or T_dict['cat_encoding'] == 'one-hot':
        cat_size = np.array([0])
    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(cat_size) + num_numerical_features
    model_params['d_in'] = d_in

    print("model_params:", model_params)
    if model_type == 'mlp':
        model = get_model(model_params).to(device)
    else: 
        raise ValueError('Not a mlp model')

    diffusion = GaussianMultinomialDiffusion(
        num_classes=cat_size,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type= gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler= scheduler,
        device=device
    )
    diffusion = diffusion.to(device)
    diffusion.train() #set denoise_fn to train status
    trainer = Trainer(
        diffusion,
        pretrain_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        device=device,
        report_every = report_every
    )
    
    trainer.run_loop()

    #save pretrain model
    print('pretrain model path:', parent_dir)
    trainer.loss_history.to_csv(os.path.join(parent_dir, 'pretrain_loss.csv'), index=False) 
    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'pretrain_model.pt'))
    torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'pretrain_ema_model.pt'))

    del model, diffusion
    torch.cuda.empty_cache()


def finetune(
        parent_dir,
        dataset = None,
        steps = 1000,
        lr = 0.002,
        weight_decay = 1e-4,
        batch_size = 1024,
        model_type = 'mlp',
        model_params = None,
        model_path = None,
        num_timesteps = 1000,
        gaussian_loss_type = 'mse',
        scheduler = 'cosine',
        T_dict = None,
        num_numerical_features = 0,
        device = torch.device('cuda:0'),
        seed = 0,
        dp_epsilon = None,
        dp_delta = None,
        rho_used = None,
        report_every=False
): 
    random.seed(seed)
    train_loader = prepare_torch_dataloader(dataset, split='train', batch_size=batch_size)

    cat_size = np.array(dataset.get_category_sizes('train'))
    if len(cat_size) == 0 or T_dict['cat_encoding'] == 'one-hot':
        cat_size = np.array([0])

    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(cat_size) + num_numerical_features
    model_params['d_in'] = d_in
    
    if model_type == 'mlp':
        model = get_model(model_params).to(device)
    else: 
        raise ValueError('Not a mlp model')

    if model_path is not None:
        model.load_state_dict(
            torch.load(model_path, map_location="cpu")
        )

    diffusion = GaussianMultinomialDiffusion(
        num_classes=cat_size,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type= gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler= scheduler,
        device=device
    )
    diffusion = diffusion.to(device)
    diffusion.train()
    finetuner = Trainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        device=device,
        dp_epsilon = dp_epsilon,
        dp_delta = dp_delta,
        rho_used = rho_used,
        report_every = report_every
    )
    finetuner.run_loop()

    print('model path:', parent_dir)
    finetuner.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False) 
    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.pt'))
    # torch.save(finetuner.ema_model.state_dict(), os.path.join(parent_dir, 'ema_model.pt'))
    torch.cuda.empty_cache() 

    return diffusion

