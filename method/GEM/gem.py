import time
import copy
import pickle
import argparse

import numpy as np
from torch import optim
import torch.nn.functional as F
import sklearn.preprocessing
import method.GEM.Util.util_general as util
from method.GEM.Util.util_gem import *
from method.GEM.Util.qm import QueryManager
from method.GEM.Util.gan.ctgan.models import Generator
from method.GEM.Util.gan.ctgan.transformer import DataTransformer
from method.GEM.mbi import Dataset, Domain
from evaluator.eval_tvd import tvd_divide


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
    def __init__(self, device='cuda:0', embedding_dim=128, gen_dim=(256, 256), batch_size=500, save_dir=None, preprocesser=None, args=None):
        self.device = torch.device(device)
        self.save_dir = save_dir

        self.embedding_dim = embedding_dim
        self.gen_dim = gen_dim

        self.batch_size = batch_size
        self.mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        self.std = self.mean + 1

        self.true_max_errors = []
        self.preprocesser = preprocesser
        self.args = args

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

    def fit(self, T, eps0, sensitivity, qm, real_answers,
            lr=1e-4, eta_min=1e-5, resample=False, ema_beta=0.5,
            max_idxs=100, max_iters=100, alpha=0.5,
            save_interval=10, save_num=50, verbose=False):

        real_answers = torch.tensor(real_answers).to(self.device)
        queries = torch.tensor(qm.queries).to(self.device).long()

        self.past_query_idxs = torch.tensor([])
        self.past_measurements = torch.tensor([])
        self.all_max_errors = []

        self.optimizerG = optim.Adam(self.generator.parameters(), lr=lr)
        if eta_min is not None:
            self.schedulerG = optim.lr_scheduler.CosineAnnealingLR(self.optimizerG, T, eta_min=eta_min)

        fake_data = self.generate_fake_data(self.mean, self.std, resample=resample)
        fake_answers = self._get_fake_answers(fake_data, qm)
        answer_diffs = real_answers - fake_answers

        if self.args.test:
            tvd_tracker = pd.DataFrame(columns=['T', 'num-num tvd', 'cat-cat tvd', 'num-cat tvd'])

        ema_error = None
        for t in tqdm(range(T)):
            # get max error query /w exponential mechanism (https://arxiv.org/pdf/2004.07223.pdf Lemma 3.2)
            score = answer_diffs.abs().cpu().numpy()
            score[self.past_query_idxs.cpu()] = -np.infty # to ensure we don't resample past queries (though unlikely)
            EM_dist_0 = np.exp(2 * alpha * eps0 * score / (2 * sensitivity), dtype=np.float128) 
            EM_dist = EM_dist_0 / EM_dist_0.sum()
            max_query_idx = util.sample(EM_dist)

            max_query_idx = torch.tensor([max_query_idx]).to(self.device)
            sampled_max_error = answer_diffs[max_query_idx].abs().item()

            # get noisy measurements
            real_answer = real_answers[max_query_idx]
            real_answer += np.random.normal(loc=0, scale=sensitivity / (eps0 * (1-alpha)))
            real_answer = torch.clamp(real_answer, 0, 1)

            # keep track of past queries
            if len(self.past_query_idxs) == 0:
                self.past_query_idxs = torch.cat([max_query_idx])
                self.past_measurements = torch.cat([real_answer])
            elif max_query_idx not in self.past_query_idxs:
                self.past_query_idxs = torch.cat((self.past_query_idxs, max_query_idx)).clone()
                self.past_measurements = torch.cat((self.past_measurements, real_answer)).clone()

            errors, q_t_idxs = self._get_past_errors(fake_data, queries)
            idx_max = errors.argmax().item()
            curr_max_error = errors[idx_max].item()
            self.all_max_errors.append(curr_max_error)

            if ema_error is None:
                ema_error = curr_max_error
            ema_error = ema_beta * ema_error + (1 - ema_beta) * curr_max_error
            threshold = 0.5 * ema_error

            lr = None
            for param_group in self.optimizerG.param_groups:
                lr = param_group['lr']
            optimizer = optim.Adam(self.generator.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iters, eta_min=1e-8)


            step = 0
            while step < max_iters:
                optimizer.zero_grad()

                idxs = torch.arange(q_t_idxs.shape[0], device = self.device)

                # above THRESHOLD
                mask = errors >= threshold
                idxs = idxs[mask]
                q_t_idxs = q_t_idxs[mask]
                errors = errors[mask]

                # get top MAX_IDXS
                max_errors_idxs = errors.argsort()[-max_idxs:]
                idxs = idxs[max_errors_idxs]
                q_t_idxs = q_t_idxs[max_errors_idxs]
                errors = errors[max_errors_idxs]

                if len(q_t_idxs) == 0: # no errors above threshold
                    break
                

                fake_query_attr = fake_data[:, queries[q_t_idxs]]
                fake_answer = fake_query_attr.prod(-1).mean(axis=0)
                real_answer = self.past_measurements[idxs].clone()

                errors = (real_answer - fake_answer).abs()
                loss = errors.mean()

                loss.backward()
                optimizer.step()
                scheduler.step()

                # generate new data for next iteration
                fake_data = self.generate_fake_data(self.mean, self.std, resample=resample)
                errors, q_t_idxs = self._get_past_errors(fake_data, queries)

                step += 1

            if hasattr(self, "schedulerG"):
                self.schedulerG.step()

            fake_answers = self._get_fake_answers(fake_data, qm)
            answer_diffs = real_answers - fake_answers
            true_max_error = answer_diffs.abs().max().item()
            # answer_diffs[self.past_query_idxs] = 0 # to ensure we don't resample past queries (though unlikely)

            self.true_max_errors.append(true_max_error)

            save_path = os.path.join(self.save_dir, 'epoch_{}.pkl'.format(t + 1))
            if ((t + 1) % save_interval == 0) or (t + 1 > T - save_num):
                self.save(save_path)

            if verbose and step > 0:
                print("Epoch {}:\tTrue Error: {:.4f}\tEM Error: {:.4f}\n"
                      "Iters: {}\tLoss: {:.4f}".format(
                    t, true_max_error, sampled_max_error, step, loss.item())) 
            
            if self.args.test:
                tvd1, tvd2,tvd3= self.tvd_summary_simple(fake_data)
                tvd_tracker.loc[len(tvd_tracker)] = [t+1,tvd1,tvd2,tvd3]

        if self.args.test:
            tvd_tracker.to_csv(os.path.join(self.save_dir, 'tvd_track.csv'))
        
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

        # df.iloc[:, self.num_idx] = self.num_encoder.inverse_transform(df.iloc[:, self.num_idx])

        preprocesser.reverse_data(df, parent_dir)
        print('Synthesizing finished')

    def tvd_summary(self, fake_data):
        with torch.no_grad():
            data = self.get_onehot(fake_data).detach().cpu()
            data = self.transformer.inverse_transform(data, None)

            x_num_fake, x_cat_fake, y_fake = self.preprocesser.reverse_data(data)
            
            x_num_real = None 
            x_cat_real = None
            if os.path.exists(f'data/{self.args.dataset}/X_num_test.npy'):
                x_num_real = np.load(f'data/{self.args.dataset}/X_num_test.npy', allow_pickle=True)
            if os.path.exists(f'data/{self.args.dataset}/X_cat_test.npy'):
                x_cat_real = np.load(f'data/{self.args.dataset}/X_cat_test.npy', allow_pickle=True)
            y_real = np.load(f'data/{self.args.dataset}/y_test.npy', allow_pickle=True)

            num_id = x_num_real.shape[1] if x_num_real is not None else 0
            cat_id = x_cat_real.shape[1] if x_cat_real is not None else 0

            tvd1 = tvd_divide(
                x_num_real, x_cat_real, y_real,
                x_num_fake, x_cat_fake, y_fake,
                dim=2, part = ['num-num']
            )['2way margin'][0]

            tvd2 = tvd_divide(
                x_num_real, x_cat_real, y_real,
                x_num_fake, x_cat_fake, y_fake,
                dim=2, part = ['cat-cat']
            )['2way margin'][0]

            tvd3 = tvd_divide(
                x_num_real, x_cat_real, y_real,
                x_num_fake, x_cat_fake, y_fake,
                dim=2, part = ['num-cat']
            )['2way margin'][0]

            tvd4 = tvd_divide(
                x_num_real, x_cat_real, y_real,
                x_num_fake, x_cat_fake, y_fake,
                dim=2, part = ['num-y']
            )['2way margin'][0]

            tvd5 = tvd_divide(
                x_num_real, x_cat_real, y_real,
                x_num_fake, x_cat_fake, y_fake,
                dim=2, part = ['cat-y']
            )['2way margin'][0]

            tvd = (num_id * tvd4 + cat_id * tvd5 + (num_id-1)*num_id/2 * tvd1 + num_id*cat_id * tvd3 + (cat_id-1)*cat_id/2 * tvd2) / ((num_id + cat_id) * (num_id + cat_id + 1)/2)

            return tvd1, tvd2, tvd3, tvd4, tvd5, tvd

    def tvd_summary_simple(self, fake_data):
        with torch.no_grad():
            data = self.get_onehot(fake_data).detach().cpu()
            data = self.transformer.inverse_transform(data, None)

            x_num_fake, x_cat_fake, y_fake = self.preprocesser.reverse_data(data)
            
            x_num_real = None 
            x_cat_real = None
            if os.path.exists(f'data/{self.args.dataset}/X_num_test.npy'):
                x_num_real = np.load(f'data/{self.args.dataset}/X_num_test.npy', allow_pickle=True)
            if os.path.exists(f'data/{self.args.dataset}/X_cat_test.npy'):
                x_cat_real = np.load(f'data/{self.args.dataset}/X_cat_test.npy', allow_pickle=True)
            y_real = np.load(f'data/{self.args.dataset}/y_test.npy', allow_pickle=True)

            num_id = x_num_real.shape[1] if x_num_real is not None else 0
            cat_id = x_cat_real.shape[1] if x_cat_real is not None else 0

            tvd1 = tvd_divide(
                x_num_real, x_cat_real, y_real,
                x_num_fake, x_cat_fake, y_fake,
                dim=2, part = ['num-num']
            )['2way margin'][0]

            tvd2 = tvd_divide(
                x_num_real, x_cat_real, y_real,
                x_num_fake, x_cat_fake, y_fake,
                dim=2, part = ['caty-caty']
            )['2way margin'][0]

            tvd3 = tvd_divide(
                x_num_real, x_cat_real, y_real,
                x_num_fake, x_cat_fake, y_fake,
                dim=2, part = ['num-caty']
            )['2way margin'][0]


            return tvd1, tvd2, tvd3


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
    args.T = 5 * df.shape[1]
    args.alpha = 0.5
    args.dim = sum(domain.values())
    args.lr = 1e-3
    args.eta_min = None 
    args.max_iters = 500
    args.max_idxs = 100
    args.resample = False
    args.verbose = False 
    return args


def gem_main(args, df, domain, rho, parent_dir, **kwargs):
    # args = get_args()

    # dataset_name = args.dataset
    # if args.dataset.startswith('acs_') and args.state is not None:
    #     dataset_name += '_{}'.format(args.state)
    # if args.dataset.startswith('adult') and args.adult_seed is not None:
    #     dataset_name += '_{}'.format(args.adult_seed)

    # results_dir ='results/{}'.format(dataset_name)
    # save_dir_query = 'save/qm/{}/{}_{}_{}/'.format(args.dataset, args.marginal, args.workload, args.workload_seed)
    # save_dir = 'save/gem/{}/{}_{}_{}/{}_{}_{}_{}/'.format(dataset_name,
    #                                                       args.marginal, args.workload, args.workload_seed,
    #                                                       args.epsilon, args.T, args.alpha, args.syndata_size)
    # if args.dataset_pub is not None:
    #     dataset_pub_name = args.dataset_pub
    #     if args.dataset_pub.startswith('acs_') and args.state_pub is not None:
    #         dataset_pub_name += '_{}'.format(args.state_pub)
    #     elif args.dataset_pub.startswith('adult') and args.adult_seed is not None:
    #         dataset_pub_name += '_{}'.format(args.adult_seed)
    #     if args.reduce_attr:
    #         dataset_pub_name += '_reduce_attr'
    #     save_dir = 'save/gem_pub/{}/{}_{}_{}/{}/{}_{}_{}_{}/'.format(dataset_name,
    #                                                                  args.marginal, args.workload, args.workload_seed,
    #                                                                  dataset_pub_name,
    #                                                                  args.epsilon, args.T, args.alpha, args.syndata_size)
    # for d in [results_dir, save_dir_query, save_dir]:
    #     if not os.path.exists(d):
    #         os.makedirs(d)

    # ### Setup Data ###
    # proj = get_proj(args.dataset)
    # if args.dataset.endswith('-small'):
    #     args.dataset = args.dataset[:-6]

    # filter_private, filter_pub = get_filters(args)

    # marginals = [args.marginal]
    # if args.all_marginals:
    #     marginals += list(np.arange(args.marginal)[1:][::-1])
    args = add_default_params(args, df, domain)
    proj = list(df.columns)
    domain = Domain(domain.keys(), domain.values())
    data = Dataset(df, domain)

    workloads = randomKwayData(data, args.workload, args.marginal, seed=args.workload_seed)
    print('Woekloads size:', len(workloads))

    N = data.df.shape[0]
    # domain_dtype = data.df.max().dtype

    query_manager = QueryManager(data.domain, workloads)
    real_answers = query_manager.get_answer(data, concat=False)

    ### Train generator ###
    # delta = 1.0 / N ** 2
    # eps0, rho = util.get_eps0_zCDP(args.epsilon, delta, args.T, alpha=args.alpha)
    eps0 = util.get_eps0_simple(rho, args.T, alpha=args.alpha)

    # result_cols = {'adult_seed': args.adult_seed,
    #                'marginal': args.marginal,
    #                'all_marginals': args.all_marginals,
    #                'num_workloads': len(workloads),
    #                'workload_seed': args.workload_seed,
    #                'num_queries': query_manager.num_queries,
    #                'dataset_pub': args.dataset_pub,
    #                'state_pub': args.state_pub,
    #                'priv_size': N,
    #                }
    # run_id = hash(time.time())
    gem = GEM(
        embedding_dim=args.dim, device=args.device, gen_dim=[args.dim * 2, args.dim * 2], batch_size=args.syndata_size, save_dir=parent_dir, 
        preprocesser=kwargs.get('preprocesser', None), args=args
    )

    # if args.dataset_pub is not None:
    #     dataset_pub_name = args.dataset_pub
    #     if args.dataset_pub.startswith('acs_') and args.state_pub is not None:
    #         dataset_pub_name += '_{}'.format(args.state_pub)
    #     elif args.dataset_pub.startswith('adult') and args.adult_seed is not None:
    #         dataset_pub_name += '_{}'.format(args.adult_seed)
    #     if args.reduce_attr:
    #         dataset_pub_name += '_reduce_attr'
    #     save_path_pub = 'save/gem_nondp/{}/{}_{}_{}/best.pkl'.format(
    #         dataset_pub_name, args.marginal, args.workload, args.workload_seed)
    #     gem_pub = copy.deepcopy(gem)
    #     gem_pub.load(save_path_pub)
    #     gem.generator = gem_pub.generator
    #     gem.fakez = gem_pub.fakez
    #     del gem_pub
    gem.setup_data(data.df, proj, data.domain, overrides=['transformer'])

    k_thresh = np.round(args.T * 0.5).astype(int)
    k_thresh = np.maximum(1, k_thresh)
    gem.fit(T=args.T, eps0=eps0, sensitivity=1 / N, lr=args.lr, eta_min=args.eta_min,
            qm=query_manager, real_answers=np.concatenate(real_answers),
            max_iters=args.max_iters, alpha=args.alpha,
            save_num=k_thresh, verbose=args.verbose)

    return {"gem_generator": gem}



    # metrics = ['max', 'mean', 'median', 'mean_squared', 'mean_workload_squared']
    # errors = {}
    # for metric in metrics:
    #     errors[metric] = []
    #     errors['distr_' + metric] = []

    # ### Evaluate ###
    # num_samples = 100000 // args.syndata_size
    # k_evals = []

    # k_evals.append('LAST')
    # _errors, _errors_distr = get_syndata_errors(gem, query_manager, num_samples, data.domain, real_answers, resample=args.resample)
    # for metric in metrics:
    #     errors[metric].append(_errors[metric])
    #     errors['distr_' + metric].append(_errors_distr[metric])

    # # ema weights of last k generators
    # for beta in [0.5, 0.9, 0.99]:
    #     k_evals.append('EMA_{}'.format(beta))
    #     weights = get_ema_weights(gem, args.T, k_thresh, beta, parent_dir)
    #     gem.generator.load_state_dict(weights)
    #     _errors, _errors_distr = get_syndata_errors(gem, query_manager, num_samples, data.domain, real_answers, resample=args.resample)
    #     for metric in metrics:
    #         errors[metric].append(_errors[metric])
    #         errors['distr_' + metric].append(_errors_distr[metric])

    # ### Save results ###
    # results = {'run_id': [run_id] * len(k_evals),
    #            'epsilon': [args.epsilon] * len(k_evals),
    #            'T': [args.T] * len(k_evals),
    #            'eps0': [eps0] * len(k_evals),
    #            'lr': [args.lr] * len(k_evals),
    #            'eta_min': [args.eta_min] * len(k_evals),
    #            'max_iters': [args.max_iters] * len(k_evals),
    #            'alpha': [args.alpha] * len(k_evals),
    #            'syndata_size': [args.syndata_size] * len(k_evals),
    #            'resample': args.resample,
    #            'last_k_iters': k_evals,
    #            'max_error': errors['max'],
    #            'mean_error': errors['mean'],
    #            'median_error': errors['median'],
    #            'mean_squared_error': errors['mean_squared'],
    #            'mean_workload_squared_error': errors['mean_workload_squared'],
    #            'distr_max_error': errors['distr_max'],
    #            'distr_mean_error': errors['distr_mean'],
    #            'distr_median_error': errors['distr_median'],
    #            'distr_mean_squared_error': errors['distr_mean_squared'],
    #            'distr_mean_workload_squared_error': errors['distr_mean_workload_squared'],
    #            }
    # df_results = pd.DataFrame.from_dict(results)
    # i = df_results.shape[1]
    # for key, val in result_cols.items():
    #     df_results[key] = val

    # # rearrange columns for better presentation
    # cols = list(df_results.columns[i:]) + list(df_results.columns[:i])
    # df_results = df_results[cols]
    # print(df_results[['last_k_iters', 'max_error']])

    # if args.dataset != 'adult':
    #     del df_results['adult_seed']
    # if args.state_pub is None:
    #     del df_results['state_pub']

    # if args.dataset_pub is None:
    #     del df_results['dataset_pub']
    #     results_path = os.path.join(parent_dir, 'gem.csv')
    # else: # using pretrained public generator
    #     if args.reduce_attr:
    #         results_path = os.path.join(parent_dir, 'gem_pub_reduced.csv')
    #     else:
    #         results_path = os.path.join(parent_dir, 'gem_pub.csv')
    # save_results(df_results, results_path=results_path)
