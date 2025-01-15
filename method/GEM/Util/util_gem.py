from tqdm import tqdm
from method.GEM.Util.data_pub_sampling import *
import torch

def soft_cross_entropy(input, target):
    logprobs = torch.log(input)
    xe = -(target * logprobs).sum(axis=-1).mean()
    return xe

def soft_cross_entropy_1d(input, target):
    input = torch.stack((input, 1-input), dim=-1)
    target = torch.stack((target, 1-target), dim=-1)
    return soft_cross_entropy(input, target)

# get rows that are missing to complete the support (for instantiating the transformer)
def get_missing_rows(train_data, discrete_columns, domain):
    missing_cols = {}
    for col in discrete_columns:
        domain_vals = set(np.arange(domain[col]))
        train_data_vals = set(train_data[col].unique())
        missing = list(domain_vals - train_data_vals)
        if len(missing) != 0:
            missing_cols[col] = missing

    extra_rows = []
    if len(missing_cols) != 0:
        num_extra_rows = max([len(x) for x in missing_cols.values()])
        extra_rows = train_data.loc[:num_extra_rows - 1].copy()
        for col, missing_vals in missing_cols.items():
            extra_rows.loc[:len(missing_vals) - 1, col] = missing_vals

    return extra_rows

def get_synth_data(model, size, dtype=np.int8):
    if size <= model.batch_size:
        num_gen = 1
        gen_size = size
    else:
        num_gen = np.ceil(size / model.batch_size).astype(int)
        gen_size = model.batch_size
    mean = torch.zeros(gen_size, model.embedding_dim, device=model.device)
    std = mean + 1

    df_synth = []
    for _ in range(num_gen):
        fake_data = model.generate_fake_data(mean, std)
        fake_data = fake_data.detach().cpu()
        fake_data = model.transformer.inverse_transform(fake_data, None)
        df_synth.append(fake_data)
    df_synth = pd.concat(df_synth).reset_index(drop=True)
    df_synth = df_synth.astype(dtype)

    return df_synth

def get_syndata_last_iters(model, size, T, last_iters=1, save_dir=None, dtype=np.int8):
    df_synth = []
    for i in tqdm(range(last_iters)):
        load_path = os.path.join(save_dir, 'epoch_{}.pkl'.format(T - i))
        model.load(load_path)

        df = get_synth_data(model, size, dtype=dtype)
        df['iter'] = i
        df_synth.append(df)

    df_synth = pd.concat(df_synth).reset_index(drop=True)
    return df_synth

def get_support_last_iters(df_synth, domain):
    data_synth = Dataset(df_synth, domain)
    data_support, A_init = get_A_init(data_synth, data_synth.df)

    df_support = data_support.df
    df_support.reset_index(inplace=True)
    df_synth.set_index('iter', inplace=True)
    cols = list(df_synth.columns.values)

    k = df_synth.index.unique().values.shape[0]
    A_iters = np.zeros((k, A_init.shape[-1]))
    for i in tqdm(range(k)):
        df = df_synth.loc[[i]]
        df_merge = pd.merge(df, df_support, left_on=cols, right_on=cols)
        assert (df.shape[0] == df_merge.shape[0])

        counts = df_merge.groupby('index').size()
        A = np.zeros(A_init.shape)
        A[counts.index.values] = counts.values
        assert (counts.sum() == df.shape[0])

        A_iters[i] = A
    del df_support['index']

    return data_support, A_iters

def get_avg_weights(model, T, last_iters, save_dir):
    weights = {}
    k_array = np.arange(last_iters)[::-1]
    for i in tqdm(k_array):
        load_path = os.path.join(save_dir, 'epoch_{}.pkl'.format(T - i))
        model.load(load_path)

        for k, m in model.generator.named_modules(): # temporary work around for mismatched pytorch versions
            m._non_persistent_buffers_set = set()

        w = model.generator.state_dict()
        for key in w.keys():
            if key not in weights.keys():
                weights[key] = w[key]
            else:
                weights[key] += w[key]
    for key in weights.keys():
        weights[key] = weights[key] / float(len(k_array))
    return weights

def get_ema_weights(model, T, last_iters, beta, save_dir):
    weights = {}
    k_array = np.arange(last_iters)[::-1]
    for i in tqdm(k_array):
        load_path = os.path.join(save_dir, 'epoch_{}.pkl'.format(T - i))
        model.load(load_path)

        for k, m in model.generator.named_modules(): # temporary work around for mismatched pytorch versions
            m._non_persistent_buffers_set = set()

        w = model.generator.state_dict()
        for key in w.keys():
            if key not in weights.keys():
                weights[key] = w[key]
            else:
                weights[key] = beta * weights[key] + (1 - beta) * w[key]
    return weights