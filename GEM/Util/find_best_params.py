# import sys
# sys.path.append("../private-pgm/src")

# from mwem_cdp import *
# from GEM.Util.data_pub_sampling import *

# def get_args():
#     parser = argparse.ArgumentParser()

#     # privacy params
#     parser.add_argument('--dataset', type=str, help='queries', default='adult')
#     parser.add_argument('--marginal', type=int, help='queries', default=3)
#     parser.add_argument('--workload', type=int, help='queries', default=128)
#     parser.add_argument('--workload_seed', type=int, default=0)
#     # public dataset params
#     parser.add_argument('--fn', type=str, default='mwem_cdp_pub.csv')
#     parser.add_argument('--dataset_pub', type=str, default=None)
#     parser.add_argument('--pub_frac', type=float, default=1.0)
#     parser.add_argument('--frac_seed', type=int, default=0)
#     # acs params
#     parser.add_argument('--state', type=str, default=None)
#     parser.add_argument('--state_pub', type=str, default=None)
#     # misc params
#     parser.add_argument('--permute', action='store_true')
#     parser.add_argument('--error_col', type=str, default='max_error_avg')

#     args = parser.parse_args()
#     if args.dataset_pub is None:
#         args.dataset_pub = args.dataset

#     # validate params
#     if args.pub_frac == 1 and args.frac_seed != 0:
#         print("Only need to run frac_seed=0 for pub_frac=1.0")
#         exit()
#     if args.dataset.startswith('acs_'):
#         assert(args.state is not None)
#         assert(args.state_pub is not None)

#     print(args)
#     return args

# if __name__ == "__main__":
#     args = get_args()

#     dataset_name = args.dataset
#     if args.dataset.startswith('acs_') and args.state is not None:
#         dataset_name += '_{}'.format(args.state)
#     results_dir ='results/{}'.format(dataset_name)

#     results_path = os.path.join(results_dir, args.fn)
#     df = pd.read_csv(results_path)

#     masks = []
#     masks.append(df['marginal'] == args.marginal)
#     masks.append(df['num_workloads'] == args.workload)
#     masks.append(df['workload_seed'] == args.workload_seed)
#     masks.append(df['dataset_pub'] == args.dataset_pub)
#     masks.append(df['state_pub'] == args.state_pub)
#     masks.append(df['frac_seed'] == args.frac_seed)
#     masks.append(df['permute'] == args.permute)

#     for mask in masks:
#         df = df[mask]

#     df = df.groupby(['epsilon', 'T']).mean().reset_index()
#     # pdb.set_trace()
#     df = df.loc[df.groupby('epsilon')[args.error_col].idxmin()]
#     df = df[['epsilon', 'T', args.error_col]]
#     print(df)

#     for T in df['T'].values:
#         print(T)


# Pennsylvania small
'''
python Util/find_best_params.py \
--dataset acs_2014_1yr-small --state PA \
--marginal 5 --workload 3003 --workload_seed 0 \
--dataset_pub acs_2014_1yr --state_pub NY \
--pub_frac 1 --frac_seed 0 --permute \
--fn mwem_cdp_pub_support.csv --error_col max_error_last
'''

# Pennsylvania
'''
python Util/find_best_params.py \
--dataset acs_2014_1yr --state PA \
--marginal 3 --workload 4096 --workload_seed 0 \
--dataset_pub acs_2014_1yr --state_pub FL \
--pub_frac 1 --frac_seed 0 --permute
'''

# Georgia
'''
python Util/find_best_params.py \
--dataset acs_2014_1yr --state GA \
--marginal 3 --workload 4096 --workload_seed 0 \
--dataset_pub acs_2014_1yr --state_pub NC \
--pub_frac 1 --frac_seed 0 --permute
'''

# California
'''
python Util/find_best_params.py \
--dataset acs_2014_1yr --state CA \
--marginal 3 --workload 4096 --workload_seed 0 \
--dataset_pub acs_2010_1yr --state_pub CA \
--pub_frac 1 --frac_seed 0 --permute
'''

# New York
'''
python Util/find_best_params.py \
--dataset acs_2014_1yr --state NY \
--marginal 3 --workload 4096 --workload_seed 0 \
--dataset_pub acs_2014_1yr --state_pub IL \
--pub_frac 1 --frac_seed 0 --permute
'''

