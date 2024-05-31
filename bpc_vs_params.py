import copy
import os
import os.path as osp

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import matplotlib.pyplot as plt

from utils import get_yaml_dict
import numpy as np


def satisfies_conditions(root_a, root_b, catchall='any'):

    for key, value in root_a.items():
        if isinstance(value, dict):
            if not satisfies_conditions(root_a[key], root_b[key]):
                return False
        elif isinstance(root_b[key], list):
            if root_a[key] not in root_b[key]:
                return False
        else:
            if root_a[key] != root_b[key] and root_b[key] != catchall :
                return False

    return True


def get_leaf_value(root, addr_string):
    keys = addr_string.split('.')
    value = copy.deepcopy(root)
    while len(keys) > 0:
        value = value[keys.pop(0)]
    return value


@hydra.main(version_base=None, config_path="./visualization/", config_name="visualization_configs")
def main(cfg: DictConfig) -> None:
    args = OmegaConf.to_container(cfg, resolve=True)
    for key, value in args.items():
        print(key + " : " + str(value))

    groups = dict()
    
    for filename in os.listdir(args['visualization']['root'])  :
        if "DS_Store" in filename: ## for MAC DSstore 
            continue
        
        dct = torch.load(
            osp.join(args['visualization']['root'], filename, 'model_best.pth'), map_location=torch.device('cpu')
        )
        exp_cfg = get_yaml_dict(osp.join(args['visualization']['root'], filename, "configs.yaml"))
        
        # Configuring what to include in plot
        if not satisfies_conditions(exp_cfg, args, catchall='any'):
            print("  Skipping: {} | Epochs: {}".format(filename, dct['epoch']))
            continue

        bpc = dct['test_metrics']['bpc']
        params = dct['num_params']
        print("Exp: {} | Epochs: {}".format(filename, dct['epoch']))

        group_name = ", ".join([str(get_leaf_value(exp_cfg, attr_name)) for attr_name in args['visualization']['group_by']])
        if group_name not in groups:
            groups[group_name] = ([params], [bpc])
        else:
            groups[group_name][0].append(params)
            groups[group_name][1].append(bpc)
    
    for group_name, (params, bpc) in groups.items():
        indx = np.argsort(params)
        params_sorted = [params[i] for i in indx]
        bpc_sorted = [bpc[i] for i in indx]
        plt.plot(params_sorted, bpc_sorted, '-o', label=group_name)
 
        ### To save data: create a folder results and :
        # params = np.array(params_sorted)
        # bpc = np.array(bpc_sorted)
        # bpc = np.concatenate([params[:,np.newaxis], bpc[:,np.newaxis]], axis=1)
        # np.save(f'./results/bpc_vs_params_{group_name}.npy', bpc)

    plt.xlabel('Number of parameters')
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel('BPC')
    plt.legend()
    plt.show()
    plt.savefig(args['visualization']['output_filename'])


if __name__ == "__main__":
    main()