import numpy as np
import pandas as pd
from lib import *
import torch
import scipy 
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", help="provide desired dataset")
    parser.add_argument("--target", help="provide targeted class")
    parser.add_argument("--tokens", help="provide tokens")
    parser.add_argument("--model_info", help="provide model info")
    parser.add_argument("--mlm", help="provide mlm flag")
    parser.add_argument("--metric", help="provide metric")

    args = parser.parse_args()
    config = {}

    if bool(args.mlm):
      config['target'] = 'class_reduce'
    else:
      config['target'] = 'NoneClass' if args.target == 'None' else f'{args.target}Class'
    
    # update config
    config['dataset'] = args.dataset
    config['max_len'] = f'{args.tokens}tokens'
    config['model_info'] = args.model_info
    

    path = create_nested_directories(config)

    attentions_list = torch.load(path + 'attentions.pt')
    attributions_list = torch.load(path + 'attributions.pt')
    
    # Implementation of the second idea
    # I_{L,H} = E_x(corr(Attr(A_{L,H}), A_{L,H})

    if args.metric == 'cor':
        
        SpearMetric = cor(attentions_list, attributions_list)

        torch.save(SpearMetric, path + 'corr.pt')

        # Note the sign before parse absolute value
        SpearSign = torch.ones_like(SpearMetric)

        SpearSign[torch.where(SpearMetric < 0)] = -1
        SpearSign[torch.where(SpearMetric == 0)] = 0

        SpearMetric = torch.abs(SpearMetric)

        SpearMetric, SpearSign = SpearMetric.flatten(), SpearSign.flatten()

        params = [(0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        condition = lambda n, p: (n >= p[0]) & (n < p[1])

        result, indices = calculate_percentages(SpearMetric, condition, params)

        # Find which heads are strongly correlated 
        visualization = torch.zeros((12, 12)).detach()

        for item in indices.items():
            for i in item[1]:
                visualization[i // 12][i % 12] = f(item[0])

        torch.save(visualization, path + 'visualization.pt')

        params = [1, -1]
        condition = lambda n, p: (n == p)

        # Filename for storine the statistics
        filename = path + 'stats.txt'

        with open(filename, 'w') as file:
            for (r, p), ind in zip(result.items(), indices.items()):
                file.write(f"Percentage of correlation scores between {r[0]} and {r[1]}: {p:.2f}%\n")

                ind_result, _ = calculate_percentages(SpearSign[ind[1]], condition, params)

                for ind in ind_result.items():
                    file.write(f"\tFrom which the percentage of {'positive' if ind[0] == 1 else 'negative'} scores: {ind[1]:.2f}%\n")


        # # Compute the correlation - Outliers Removal
        # SpearMetricOutRemoval, SpearSignOutRemoval = corr_computation_outlier_rem(attentions_mean, attributions_mean)


        # params = [(0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        # condition = lambda n, p: (n >= p[0]) & (n < p[1])

        # result, indices = calculate_percentages(SpearMetricOutRemoval, condition, params)

        # # Find which heads are strongly correlated 
        # visualization = torch.zeros((12, 12)).detach()

        # for item in indices.items():
        #     for i in item[1]:
        #         visualization[i // 12][i % 12] = f(item[0])

        # torch.save(visualization, path + '-visualization_outrem.pt')

        # params = [1, -1]
        # condition = lambda n, p: (n == p)

        # # Filename for storine the statistics without outliers
        # filename = path + '-stats_outremoval.txt'

        # with open(filename, 'w') as file:
        #     for (r, p), ind in zip(result.items(), indices.items()):
        #         file.write(f"Percentage of correlation scores between {r[0]} and {r[1]}: {p:.2f}%\n")

        #         ind_result, _ = calculate_percentages(SpearSignOutRemoval[ind[1]], condition, params)

        #         for ind in ind_result.items():
        #             file.write(f"\tFrom which the percentage of {'positive' if ind[0] == 1 else 'negative'} scores: {ind[1]:.2f}%\n")
        
    else:

        # result = calculate_average_mutual_information_continuous(attentions_list, attributions_list)
        # torch.save(result, path + 'mutual.pt')
        
        # result = calculate_average_kl_divergence(attentions_list, attributions_list) #og
        # torch.save(result, path + 'kl_div.pt') #og
        result = kl_div(attributions_list, attentions_list) #reverse
        torch.save(result, path + 'kl_div_rev.pt') #reverse