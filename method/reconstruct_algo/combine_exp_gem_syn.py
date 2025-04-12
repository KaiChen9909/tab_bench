import time
import copy
import pickle
import argparse

import numpy as np
from torch import optim
import torch.nn.functional as F

import method.GEM.Util.util_general as util
from method.GEM.Util.util_gem import *
from method.GEM.Util.qm import QueryManager
from method.GEM.Util.gan.ctgan.models import Generator
from method.GEM.Util.gan.ctgan.transformer import DataTransformer
from method.GEM.mbi import Dataset, Domain
from method.privsyn.PrivSyn.privsyn import PrivSyn


def get_syndata_errors(gem, query_manager, num_samples, domain, real_answers, resample=False):
    fake_data = gem.generate_fake_data(gem.mean, gem.std, resample=resample)

    fake_answers = gem._get_fake_answers(fake_data, query_manager).cpu().numpy()
    idxs = [len(x) for x in real_answers]
    idxs = np.cumsum(idxs)
    idxs = np.concatenate([[0], idxs])
    idxs = np.vstack([idxs[:-1], idxs[1:]])
    x = []
    for i in range(idxs.shape[-1]):
        x.append(fake_answers[idxs[0, i]:idxs[1, i]])
    fake_answers = x
    _errors_distr = util.get_errors(real_answers, fake_answers)

    samples = []
    for i in range(num_samples):
        x = gem.get_onehot(fake_data).cpu()
        samples.append(x)
    x = torch.cat(samples, dim=0)
    df = gem.transformer.inverse_transform(x, None)
    data_synth = Dataset(df, domain)

    fake_answers = query_manager.get_answer(data_synth, concat=False)
    _errors = util.get_errors(real_answers, fake_answers)

    return _errors, _errors_distr

class GEM(object):
    def __init__(self, device='cuda:0', embedding_dim=128, gen_dim=(256, 256), batch_size=500, save_dir=None):
        self.device = torch.device(device)
        self.save_dir = save_dir

        self.embedding_dim = embedding_dim
        self.gen_dim = gen_dim

        self.batch_size = batch_size
        self.mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        self.std = self.mean + 1

        self.true_max_errors = []

    def save(self, path):
        with open(path, 'wb') as handle:
            pickle.dump(self.__dict__, handle)

    def load(self, path):
        with open(path, 'rb') as handle:
            tmp_dict = pickle.load(handle)
        self.__dict__.update(tmp_dict)

    def setup_data(self, train_data, discrete_columns=tuple(), domain=None, overrides=[]):
        extra_rows = get_missing_rows(train_data, discrete_columns, domain)
        if len(extra_rows) > 0:
            train_data = pd.concat([extra_rows, train_data]).reset_index(drop=True)

        if not hasattr(self, "transformer") or 'transformer' in overrides:
            self.transformer = DataTransformer()
            self.transformer.fit(train_data, discrete_columns)

        data_dim = self.transformer.output_dimensions
        if not hasattr(self, "generator") or 'generator' in overrides:
            self.generator = Generator(self.embedding_dim, self.gen_dim, data_dim).to(self.device)
            if self.batch_size == 1: # can't apply batch norm if batch_size = 1
                self.generator.eval()

    def _apply_activate(self, data, tau=0.2):
        data_t = []
        st = 0
        for item in self.transformer.output_info:
            ed = st + item[0]
            if item[1] == 'softmax':
                logits = data[:, st:ed]
                probs = logits.softmax(-1)
                data_t.append(probs)
            else:
                assert 0
            st = ed
        return torch.cat(data_t, dim=1)

    def get_onehot(self, data, how='sample'):
        data_t = []
        st = 0
        for item in self.transformer.output_info:
            ed = st + item[0]
            if item[1] == 'softmax':
                probs = data[:, st:ed]
                out = torch.zeros_like(probs)
                if how == 'sample':
                    idxs = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)
                elif how == 'argmax':
                    idxs = probs.argmax(-1)
                else:
                    assert 0
                out[torch.arange(out.shape[0]).to(self.device), idxs] = 1
                data_t.append(out)
            else:
                assert 0
            st = ed
        return torch.cat(data_t, dim=1)

    def generate_fake_data(self, mean, std, resample=False):
        if not hasattr(self, "fakez") or resample:
            self.fakez = torch.normal(mean=mean, std=std)
        fake = self.generator(self.fakez)
        fake_data = self._apply_activate(fake)
        return fake_data

    def _get_fake_answers(self, fake_data, qm):
        fake_answers = torch.zeros(qm.queries.shape[0]).to(self.device)
        for fake_data_chunk in torch.split(fake_data.detach(), 25):# 100  #TODO: make adaptive to fit GPU memory
            x = fake_data_chunk[:, qm.queries]
            # mask = qm.queries < 0 # TODO: mask out -1 queries for different k-ways
            x = x.prod(-1)
            x = x.sum(axis=0)
            fake_answers += x
        fake_answers /= fake_data.shape[0]
        return fake_answers

    def _get_past_errors(self, fake_data, queries):
        q_t_idxs = self.past_query_idxs.clone()
        fake_query_attr = fake_data[:, queries[q_t_idxs]]
        past_fake_answers = fake_query_attr.prod(-1).mean(axis=0)
        past_real_answers = self.past_measurements.clone()

        errors = past_real_answers - past_fake_answers
        errors = torch.clamp(errors.abs(), 0, np.infty)
        return errors, q_t_idxs

    def outside_fit(self, eps0, sensitivity, qm, real_answers,
            lr=1e-4, eta_min=1e-5, resample=False, ema_beta=0.5,
            max_idxs=100, max_iters=100, alpha=0.5, marginal_idx = None,
            verbose=False):

        if isinstance(qm, list):
            real_answers_list = []
            queries = []
            for i in range(len(qm)):
                real_answer = np.array(real_answers[i])
                real_answer += np.random.normal(loc=0, scale=sensitivity / eps0[i], size=real_answer.shape)

                real_answers_list.append(real_answers[i])
                queries += list(qm[i].queries)

            real_answers = np.concatenate(real_answers_list)
            real_answers = torch.tensor(real_answers).to(self.device)
            real_answers = torch.clamp(real_answers, 0, 1)
        else:
            real_answers = torch.tensor(real_answers).to(self.device)
            real_answers += np.random.normal(loc=0, scale=sensitivity / eps0)
            real_answers = torch.clamp(real_answers, 0, 1)

            queries = torch.tensor(qm.queries).to(self.device).long()

        # self.past_query_idxs = torch.tensor([])
        # self.past_measurements = torch.tensor([])
        # self.all_max_errors = []
        self.past_query_idxs = torch.tensor(np.arange(len(queries)))
        self.past_measurements = real_answers
        self.all_max_errors = []

        self.optimizerG = optim.Adam(self.generator.parameters(), lr=lr)
        # if eta_min is not None:
        #     self.schedulerG = optim.lr_scheduler.CosineAnnealingLR(self.optimizerG, 1, eta_min=eta_min)

        fake_data = self.generate_fake_data(self.mean, self.std, resample=resample)

        lr = None
        for param_group in self.optimizerG.param_groups:
            lr = param_group['lr']
        optimizer = optim.Adam(self.generator.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iters, eta_min=1e-8)

        step = 0
        while step < max_iters:
            optimizer.zero_grad()

            fake_query_attr = [fake_data[:, q] for q in queries]
            fake_answer = [attr.prod(-1).mean(axis=0) for attr in fake_query_attr]
            real_answer = [measurement.clone() for measurement in self.past_measurements]

            errors = sum((real_answer[i] - fake_answer[i]).abs() for i in range(len(real_answer)))
            loss = errors.mean()

            loss.backward()
            optimizer.step()
            scheduler.step()

            # generate new data for next iteration
            fake_data = self.generate_fake_data(self.mean, self.std, resample=resample)
            #  errors, q_t_idxs = self._get_past_errors(fake_data, queries)

            step += 1
            print(f'step {step+1}/{max_iters} finished', end = '\r')
        
    def syn(self, n_sample, preprocesser, parent_dir, resample=False):
        n_batch = int(np.ceil(n_sample/self.batch_size))

        print('Start synthesizing')
        fake_data = self.generate_fake_data(self.mean, self.std, resample=resample)

        samples = []
        for i in range(n_batch):
            x = self.get_onehot(fake_data).cpu()
            samples.append(x)
        x = torch.cat(samples, dim=0)
        df = self.transformer.inverse_transform(x, None)
        df = df.head(n_sample)

        preprocesser.reverse_data(df, parent_dir)
        print('Synthesizing finished')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help='queries', default='adult')
    parser.add_argument('--marginal', type=int, help='queries', default=3)
    parser.add_argument('--workload', type=int, help='queries', default=32)
    parser.add_argument('--workload_seed', type=int, default=0)
    parser.add_argument('--all_marginals', action='store_true') # unused
    # privacy params
    parser.add_argument('--epsilon', type=float, help='Privacy parameter', default=1.0)
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.5)
    # acs params
    parser.add_argument('--state', type=str, default=None)
    parser.add_argument('--dataset_pub', type=str, default=None)
    parser.add_argument('--state_pub', type=str, default=None)
    parser.add_argument('--reduce_attr', action='store_true')
    # adult params
    parser.add_argument('--adult_seed', type=int, default=None)
    # GEM params
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--syndata_size', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eta_min', type=float, default=None)
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--max_idxs', type=int, default=100)
    parser.add_argument('--resample', action='store_true')
    # misc params
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    print(args)
    return args

def add_default_params(args, df, domain):
    args.marginal = 2
    args.workload = 100000
    args.workload_seed = 0
    args.syndata_size = 1024
    # args.T = 2 * df.shape[1]
    args.alpha = 0.5
    args.dim = sum(domain.values())
    args.lr = 1e-3
    args.eta_min = None 
    if args.dataset in ['bank', 'higgs-small', 'loan']:
        args.max_iters = 100
    else: 
        args.max_iters = 50
    args.max_idxs = 100
    args.resample = False
    args.verbose = False 
    return args


def gem_syn_main(args, df, domain, rho, parent_dir, **kwargs):
    args = add_default_params(args, df, domain)
    proj = list(df.columns)
    domain = Domain(domain.keys(), domain.values())
    data = Dataset(df, domain)

    # workloads = randomKwayData(data, args.workload, args.marginal, seed=args.workload_seed)
    workloads1 = [(x,) for x in proj]
    workloads2 = PrivSyn.two_way_marginal_selection(data.df, data.domain.config, 0.1*rho, 0.8*rho)
    # workloads = workloads1 + workloads2

    N = data.df.shape[0]
    # domain_dtype = data.df.max().dtype

    gem = GEM(embedding_dim=args.dim, device=args.device, gen_dim=[args.dim * 2, args.dim * 2], batch_size=args.syndata_size, save_dir=parent_dir)

    query_manager1 = QueryManager(data.domain, workloads1)
    real_answers1 = query_manager1.get_answer(data, concat=False) # [[...], [...]]

    query_manager2 = QueryManager(data.domain, workloads2)
    real_answers2 = query_manager2.get_answer(data, concat=False) # [[...], [...]]

    eps0 = [np.sqrt((0.2 * rho)/len(workloads1)), np.sqrt((1.6 * rho)/len(workloads2))]

    gem.setup_data(data.df, proj, data.domain, overrides=['transformer'])

    # k_thresh = np.round(args.T * 0.5).astype(int)
    # k_thresh = np.maximum(1, k_thresh)
    print('one-way marginal training')
    gem.outside_fit(eps0=eps0[0], sensitivity=1 / N, lr=args.lr, eta_min=args.eta_min,
            qm=query_manager1, real_answers=np.concatenate(real_answers1),
            max_iters=10, alpha=args.alpha,
            # save_num=k_thresh, 
            verbose=args.verbose)
    
    print('two-way marginal training')
    gem.outside_fit(eps0=eps0, sensitivity=1 / N, lr=args.lr, eta_min=args.eta_min,
            qm=[query_manager1, query_manager2], real_answers=[np.concatenate(real_answers1), np.concatenate(real_answers2)],
            max_iters=args.max_iters, alpha=args.alpha,
            # save_num=k_thresh, 
            verbose=args.verbose)

    return {"gem_syn_generator": gem}