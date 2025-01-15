# from mwem import *
# from GEM.Util.data_pub_sampling import *
# from GEM.Util.qm import QueryManager

# def get_args():
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--dataset', type=str, help='queries', default='acs_2018_1yr')
#     parser.add_argument('--state', type=str, default=None, required=True)
#     parser.add_argument('--workload', type=int, help='queries', default=64)
#     parser.add_argument('--workload_seed', type=int, default=0)
#     parser.add_argument('--marginal', type=int, help='queries', default=3)
#     args = parser.parse_args()

#     print(args)
#     return args

# args = get_args()

# dataset_name = args.dataset
# if args.dataset.startswith('acs_') and args.state is not None:
#     dataset_name += '_{}'.format(args.state)
# results_dir = 'results/{}'.format(dataset_name)
# if not os.path.exists(results_dir):
#     os.makedirs(results_dir)

# proj = get_proj(args.dataset)
# if args.dataset.endswith('-small'):
#     if args.dataset.startswith('acs'):
#         args.dataset = args.dataset[:-6]
#         args.dataset_pub = args.dataset_pub[:-6]

# filter = ('STATE', args.state)
# data, workloads = randomKway(args.dataset, args.workload, args.marginal, seed=args.workload_seed, proj=proj, filter=filter)
# query_manager = QueryManager(data.domain, workloads)
# N = data.df.shape[0]

# real_answers = query_manager.get_answer(data, debug=False)
# query_manager.setup_query_attr()

# pd_states = pd.read_csv('Datasets/{}.csv'.format(args.dataset))
# states = pd_states['STATE'].unique()
# del pd_states

# errors = []
# for state in states:
#     filter = ('STATE', state)
#     data_pub, _ = randomKway(args.dataset, args.workload, args.marginal, seed=args.workload_seed, proj=proj, filter=filter)
#     data_pub, A_init = get_pub_dataset(data_pub, 1.0, 0)
#     query_manager.setup_xy(data_pub)
#     fake_answers = query_manager.get_answer_weights(A_init)

#     error = np.abs(real_answers - fake_answers)
#     max_error = error.max()
#     avg_error = error.mean()
#     errors.append({'state': state, 'max_error': max_error, 'avg_error': avg_error})
#     print(state, max_error, avg_error)

# df = pd.DataFrame(errors).sort_values('max_error').reset_index(drop=True)
# df['workload'] = args.workload
# df['marginal'] = args.marginal
# df['workload_seed'] = args.workload_seed
# df = df[['workload', 'marginal', 'workload_seed', 'state', 'max_error', 'avg_error']]

# path = os.path.join(results_dir, 'state_init_errors.csv')
# save_results(df, results_path=path)