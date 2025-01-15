import os 
import json

def make_exp_dir(args):
    if args.test:
        parent_dir = f'exp/{args.dataset}/{args.method}/{args.epsilon}_{args.num_preprocess}_{args.rare_threshold}_test'
        data_path = f'data/{args.dataset}/'
        os.makedirs(parent_dir, exist_ok=True)
    elif args.syn_test:
        parent_dir = f'exp/{args.dataset}/{args.method}/{args.epsilon}_{args.num_preprocess}_{args.rare_threshold}_syn'
        data_path = f'data/{args.dataset}/'
        os.makedirs(parent_dir, exist_ok=True)
    else:
        parent_dir = f'exp/{args.dataset}/{args.method}/{args.epsilon}_{args.num_preprocess}_{args.rare_threshold}'
        data_path = f'data/{args.dataset}/'
        os.makedirs(parent_dir, exist_ok=True)

    return parent_dir, data_path


def prepare_eval_config(args, parent_dir):
    with open(f'data/{args.dataset}/info.json', 'r') as file:
        data_info = json.load(file)

    config = {
        'parent_dir': parent_dir,
        'real_data_path': f'data/{args.dataset}/',
        'model_params':{'num_classes': data_info['n_classes']},
        'sample': {'seed': 0, 'sample_num': data_info['train_size']}
    }

    with open(os.path.join(parent_dir, 'eval_config.json'), 'w') as file:
        json.dump(config, file, indent=4)
    return config

def algo_method(args):
    if args.method == 'aim':
        from method.AIM.aim import aim_main 
        algo = aim_main
    elif args.method == 'merf':
        from method.DP_MERF.single_generator_priv_all import merf_main
        algo = merf_main
    elif args.method == 'llm':
        from method.LLM.run_llm import llm_main
        algo = llm_main 
    elif args.method == 'gsd':
        from method.private_gsd.run_gsd import gsd_main 
        algo = gsd_main 
    elif args.method == 'privsyn':
        from method.privsyn.run_privsyn import privsyn_main
        algo = privsyn_main 
    elif args.method == 'ddpm':
        from method.TabDDPM.run_ddpm import ddpm_main
        algo = ddpm_main
    elif args.method == 'mrf':
        from method.PrivMRF.mrf_main import mrf_main
        algo = mrf_main 
    elif args.method == 'rap':
        from method.RAP.main import rap_main 
        algo = rap_main
    elif args.method == 'gem':
        from method.GEM.gem import gem_main 
        algo = gem_main 
    elif args.method == 'gumbel_select':
        from method.reconstruct_algo.combine_exp_select import gumbel_select_main
        algo = gumbel_select_main
    elif args.method == 'privsyn_select':
        from method.reconstruct_algo.combine_exp_select import privsyn_select_main
        algo = privsyn_select_main 
    elif args.method == 'gsd_syn':
        from method.reconstruct_algo.combine_exp_gsd_syn import gsd_syn_main 
        algo = gsd_syn_main 
    elif args.method == 'rap_syn':
        from method.reconstruct_algo.combine_exp_rap_syn import rap_syn_main 
        algo = rap_syn_main    
    elif args.method == 'gem_syn':
        from method.reconstruct_algo.combine_exp_gem_syn import gem_syn_main 
        algo = gem_syn_main   
    
    return algo