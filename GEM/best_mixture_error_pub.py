import sys
sys.path.append("../private-pgm/src")

import time

from best_mixture_error import *
from Util.data_pub_sampling import *
from Util.qm import QueryManager

import pdb

def get_args():
    parser = argparse.ArgumentParser()

    # privacy params
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--dataset', type=str, help='queries', default='adult')
    parser.add_argument('--marginal', type=int, help='queries', default=3)
    parser.add_argument('--workload', type=int, help='queries', default=128)
    parser.add_argument('--workload_seed', type=int, default=0)
    parser.add_argument('--early_stopping', type=int, default=50)
    # public dataset params
    parser.add_argument('--dataset_pub', type=str, default=None)
    parser.add_argument('--pub_frac', type=float, default=1.0)
    parser.add_argument('--frac_seed', type=int, default=0)
    # acs params
    parser.add_argument('--state', type=str, default=None)
    parser.add_argument('--state_pub', type=str, default=None)

    args = parser.parse_args()
    if args.dataset_pub is None:
        args.dataset_pub = args.dataset

    # validate params
    if args.dataset == 'adult':
        print("Since we did not run this on ADULT, we have not set up this file such that the public/private splits are consistent with the ones used in the paper for ADULT (bias experiments). " +
              "If you wish to get public splits that are consistent, please see pmw_pub_bias.py for how loading of (data_pub, A_init) should be done. "
              "Otherwise, comment out the following line (that exits the code).")
        exit()
    if args.pub_frac == 1 and args.frac_seed != 0:
        print("Only need to run frac_seed=0 for pub_frac=1.0")
        exit()
    if args.dataset.startswith('acs_'):
        assert(args.state is not None)
        assert(args.state_pub is not None)

    print(args)
    return args

if __name__ == "__main__":
    args = get_args()

    results_dir ='results/{}'.format(args.dataset)
    if args.dataset.startswith('acs_'):
        results_dir += '_{}'.format(args.state)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    save_dir_query = 'save/qm/{}/{}_{}_{}/'.format(args.dataset, args.marginal, args.workload, args.workload_seed)
    save_dir_xy = save_dir_query + 'mwem_pub/{}_{}_{}_{}'.format(args.dataset_pub, args.state_pub, args.pub_frac, args.frac_seed)
    for d in [save_dir_query, save_dir_xy]:
        if not os.path.exists(d):
            os.makedirs(d)

    proj = get_proj(args.dataset)
    if args.dataset.endswith('-small'):
        if args.dataset.startswith('acs'):
            args.dataset = args.dataset[:-6]
            args.dataset_pub = args.dataset_pub[:-6]

    filter_private, filter_pub = get_filters(args)

    data, workloads = randomKway(args.dataset, args.workload, args.marginal, seed=args.workload_seed, proj=proj, filter=filter_private)
    query_manager = QueryManager(data.domain, workloads)
    N = data.df.shape[0]

    data_pub, _ = randomKway(args.dataset_pub, args.workload, args.marginal, seed=args.workload_seed, proj=proj, filter=filter_pub)
    N_pub = int(args.pub_frac * data_pub.df.shape[0])
    data_pub, A_init = get_pub_dataset(data_pub, args.pub_frac, args.frac_seed)

    print('workload: ', len(workloads))
    print('num queries: ', query_manager.num_queries)
    print('A:', A_init.shape)

    real_answers = query_manager.get_answer(data)
    query_manager.setup_query_attr(save_dir=save_dir_query)
    query_manager.setup_xy(data_pub, save_dir=save_dir_xy)
    fake_answers = query_manager.get_answer_weights(A_init)
    init_error = np.abs(real_answers - fake_answers).max()

    result_cols = {'marginal': args.marginal,
                   'num_workloads': len(workloads),
                   'workload_seed': args.workload_seed,
                   'num_queries': query_manager.num_queries,
                   'dataset_pub': args.dataset_pub,
                   'state_pub': args.state_pub,
                   'pub_frac': args.pub_frac,
                   'frac_seed': args.frac_seed,
                   'priv_size': N,
                   'pub_size': N_pub,
                   }

    for _ in range(args.num_runs):
        run_id = hash(time.time())

        A_avg, A_last = generate_nondp(data_pub, real_answers, A_init, query_manager, early_stopping=args.early_stopping, return_last=True)

        fake_answers = query_manager.get_answer_weights(A_avg)
        max_error_avg = np.abs(real_answers - fake_answers).max()

        fake_answers = query_manager.get_answer_weights(A_last)
        max_error_last = np.abs(real_answers - fake_answers).max()

        results = {'run_id': [run_id],
                   'init_error': [init_error],
                   'max_error': [max_error_last],
                   }
        df_results = pd.DataFrame.from_dict(results)
        i = df_results.shape[1]

        for key, val in result_cols.items():
            df_results[key] = val

        # rearrange columns for better presentation
        cols = list(df_results.columns[i:]) + list(df_results.columns[:i])
        df_results = df_results[cols]

        # only need these cols for ACS experiments
        if args.state_pub is None:
            del df_results['dataset_pub']
            del df_results['state_pub']

        results_path = os.path.join(results_dir, 'mw_nondp_pub.csv')
        save_results(df_results, results_path=results_path)