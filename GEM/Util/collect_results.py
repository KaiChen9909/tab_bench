# import sys
# sys.path.append("../private-pgm/src")

# from mwem_cdp import *
# from GEM.Util.data_pub_sampling import *

# def get_args():
#     parser = argparse.ArgumentParser()

#     # privacy params
#     parser.add_argument('--marginal', type=int, help='queries', default=3)
#     parser.add_argument('--workload', type=int, help='queries', default=128)
#     parser.add_argument('--workload_seed', type=int, default=0)
#     parser.add_argument('--epsilon', type=float, nargs='+', default=[])
#     # public dataset params
#     parser.add_argument('--dataset_test', type=str, default=None)
#     parser.add_argument('--dataset_val', type=str, default=None)
#     parser.add_argument('--state', type=str, default=None)
#     parser.add_argument('--state_pub', type=str, nargs='+', default=[])
#     parser.add_argument('--pub_frac', type=float, default=1.0)
#     parser.add_argument('--frac_seed', type=int, default=0)
#     # misc params
#     parser.add_argument('--permute', action='store_true')
#     parser.add_argument('--error_col', type=str, default='max_error_avg')

#     args = parser.parse_args()

#     print(args)
#     return args

# def get_mwem_pub(args, dataset_name):
#     if dataset_name.startswith('acs_') and args.state is not None:
#         dataset_name += '_{}'.format(args.state)
#     results_dir ='results/{}'.format(dataset_name)

#     results_path = os.path.join(results_dir, 'mwem_cdp_pub.csv')
#     df = pd.read_csv(results_path)

#     masks = []
#     masks.append(df['marginal'] == args.marginal)
#     masks.append(df['num_workloads'] == args.workload)
#     masks.append(df['workload_seed'] == args.workload_seed)
#     masks.append(df['pub_frac'] == args.pub_frac)
#     masks.append(df['frac_seed'] == args.frac_seed)
#     masks.append(df['permute'] == args.permute)
#     masks.append(df['state_pub'].isin(args.state_pub))
#     masks.append(df['epsilon'].isin(args.epsilon))

#     for mask in masks:
#         df = df.loc[mask]

#     stds = df.groupby(['epsilon', 'state_pub', 'T'])[args.error_col].std().fillna(0).values
#     df = df.groupby(['epsilon', 'state_pub', 'T']).mean().reset_index()
#     df['std'] = stds

#     return df

# def get_hdmm(args):
#     dataset_name = args.dataset_test
#     if dataset_name.startswith('acs_') and args.state is not None:
#         dataset_name += '_{}'.format(args.state)
#     results_dir ='results/{}'.format(dataset_name)

#     results_path = os.path.join(results_dir, 'hdmm.csv')
#     df = pd.read_csv(results_path)

#     masks = []
#     masks.append(df['marginal'] == args.marginal)
#     masks.append(df['num_workloads'] == args.workload)
#     masks.append(df['workload_seed'] == args.workload_seed)
#     masks.append(df['epsilon'].isin(args.epsilon))

#     for mask in masks:
#         df = df.loc[mask]

#     stds = df.groupby(['epsilon'])['max_error'].std().fillna(0).values
#     df = df.groupby(['epsilon']).mean().reset_index()
#     df['std'] = stds

#     return df

# def get_dq(args):
#     dataset_name = args.dataset_test
#     if dataset_name.startswith('acs_') and args.state is not None:
#         dataset_name += '_{}'.format(args.state)
#     results_dir ='results/{}'.format(dataset_name)

#     results_path = os.path.join(results_dir, 'dq.csv')
#     df = pd.read_csv(results_path)

#     masks = []
#     masks.append(df['marginal'] == args.marginal)
#     masks.append(df['num_workloads'] == args.workload)
#     masks.append(df['workload_seed'] == args.workload_seed)
#     masks.append(df['epsilon'].isin(args.epsilon))

#     for mask in masks:
#         df = df.loc[mask]

#     stds = df.groupby(['epsilon', 'samples', 'mw'])['max_error'].std().fillna(0).values
#     df = df.groupby(['epsilon', 'samples', 'mw']).mean().reset_index()
#     df['std'] = stds
#     df = df.loc[df.groupby('epsilon')['max_error'].idxmin()]

#     return df

# def get_mwem(args):
#     dataset_name = args.dataset_test
#     if dataset_name.startswith('acs_') and args.state is not None:
#         dataset_name += '_{}'.format(args.state)
#     results_dir ='results/{}'.format(dataset_name)

#     results_path = os.path.join(results_dir, 'mwem_cdp.csv')
#     df = pd.read_csv(results_path)

#     masks = []
#     masks.append(df['marginal'] == args.marginal)
#     masks.append(df['num_workloads'] == args.workload)
#     masks.append(df['workload_seed'] == args.workload_seed)
#     masks.append(df['permute'] == args.permute)
#     masks.append(df['epsilon'].isin(args.epsilon))

#     for mask in masks:
#         df = df.loc[mask]

#     stds = df.groupby(['epsilon', 'T'])[args.error_col].std().fillna(0).values
#     df = df.groupby(['epsilon', 'T']).mean().reset_index()
#     df['std'] = stds
#     df = df.loc[df.groupby('epsilon')[args.error_col].idxmin()]

#     return df

# def get_latex(arr, textbf=False):
#     latex = '&'
#     for x in arr:
#         if textbf:
#             latex += ' $\\mathbf{{{:.4f}}}$ &'.format(x)
#         else:
#             latex += ' ${:.4f}$ &'.format(x)
#     return latex

# if __name__ == "__main__":
#     args = get_args()

#     df = get_mwem_pub(args, args.dataset_val)
#     df = df.loc[df.groupby('epsilon')[args.error_col].idxmin()]
#     df = df[['epsilon', 'state_pub', 'T', args.error_col]]
#     df_val = df
#     del df_val['max_error_avg']

#     df = get_mwem_pub(args, args.dataset_test)
#     col_join = list(df_val.columns.values)
#     df_mw_pub = pd.merge(df_val, df, left_on=col_join, right_on=col_join)
#     print('MW-PUB:')
#     print(df_mw_pub[['epsilon', 'state_pub', 'T', args.error_col, 'std']])
#     print(get_latex(df_mw_pub[args.error_col], textbf=True) + '\n')

#     df_hdmm = get_hdmm(args)
#     print('HDMM:')
#     print(df_hdmm[['epsilon',  'max_error', 'std']])
#     print(get_latex(df_hdmm['max_error']) + '\n')

#     df_dq = get_dq(args)
#     print('DQ:')
#     print(df_dq[['epsilon', 'samples', 'mw', 'max_error', 'std']])
#     print(get_latex(df_dq['max_error']) + '\n')

#     if args.dataset_test.endswith('-small'):
#         df_mwem = get_mwem(args)
#         print('MWEM:')
#         print(df_mwem[['epsilon', 'T', args.error_col, 'std']])
#         print(get_latex(df_mwem[args.error_col]) + '\n')





'''
python Util/collect_results.py --state PA \
--marginal 5 --workload 3003 --workload_seed 0 \
--epsilon 0.1 0.15 0.2 0.25 0.5 1.0 \
--dataset_test acs_2018_1yr-small --dataset_val acs_2014_1yr-small \
--state_pub PA OH NY CA \
--pub_frac 1 --frac_seed 0 --permute
'''

'''
python Util/collect_results.py --state PA \
--marginal 3 --workload 4096 --workload_seed 0 \
--epsilon 0.1 0.15 0.2 0.25 0.5 1.0 \
--dataset_test acs_2018_1yr --dataset_val acs_2014_1yr \
--state_pub PA OH NY CA \
--pub_frac 1 --frac_seed 0 --permute
'''

'''
python Util/collect_results.py --state GA \
--marginal 3 --workload 4096 --workload_seed 0 \
--epsilon 0.1 0.15 0.2 0.25 0.5 1.0 \
--dataset_test acs_2018_1yr --dataset_val acs_2014_1yr \
--state_pub GA LA NY CA \
--pub_frac 1 --frac_seed 0 --permute
'''