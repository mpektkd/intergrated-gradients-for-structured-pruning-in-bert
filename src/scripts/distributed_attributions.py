import multiprocessing as mp
import numpy as np

import traceback

import logging 


from argparse import ArgumentParser

import pandas as pd
import torch.distributed as dist

from lib import *

import torch



from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP


import time


# #### Pruning ####
# logger = logging.getLogger(__name__)

# '''
#     Prunes heads of the model. 
#     heads_to_prune: dict of {layer_num: list of heads to prune in this layer} 
#     See base class PreTrainedModel
    
#     Μεσα καλειται το prune_heads() του BertModel το οποιο καλει το find_pruneable_heads_and_indices(), 
#     που υπολογιζει βασει των ηδη pruned heads τα indices. Υπαρχει μεταβλητη already_pruned_heads, η οποια
#     γινεται update μονη της. 
# '''

# def dist_main(rank, size, dataset, tokenizer):

#     size = WORLD_SIZE
#     processes = []

#     for rank in range(size):
#         p = Process(target=init_process, args=(rank, size, run, dataset, tokenizer))
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()



def init_process(rank, size, fn, queue=None, event=None, head_mask=None, inference=False, m_config=None, t_config=None, backend='gloo'):
    try:
        """ Initialize the distributed environment. """
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(m_config["port"])
        dist.init_process_group(backend, rank=rank, world_size=size)
        if rank == 0 and queue is not None:
            final_metric, preds, labels = fn(rank, size, head_mask, inference, m_config, t_config)

            queue.put(final_metric)
            queue.put(preds)
            queue.put(labels)
            queue.put(None)
            event.wait()
        else:
            fn(rank, size, head_mask, inference, m_config, t_config)
    except Exception as e:
        traceback.print_exc()
    finally:
        # Clean up the process group
        if dist.is_initialized():
            dist.destroy_process_group()

def run(rank, size, head_mask=None, inference=False, m_config=None, t_config=None):
    '''
    Distributed computation of the attributions and attentions for 
    the whole dataset.
    '''
    
    # Set seed for ensure reproducability for each distinct process
    # https://chatgpt.com/share/96a1aac3-bfb6-49e5-a2b2-1a8adc1898d9
    set_seed(m_config["seed"], rank)

    _distributed = size > 1
    model = m_config["model"]
    dataset = m_config["dataset"]
    igspp = m_config["igspp"]
    isp = m_config["isp"]

    batch_size = m_config["batch_size"]
    task = m_config["task"]

        ##############  Implementation with no batch scaling ################
    # sampler = DistributedEvalSampler(dataset, num_replicas=size, rank=rank, shuffle=False)   

    # collator = PruningCollator(cls_id, sep_id)
    # loader = DataLoader(
    #     dataset, 
    #     batch_size=1, 
    #     sampler=sampler, 
    #     # collate_fn=collator 
    #     collate_fn=collator # because the load_configuration() does the encoding on preprocessing
    # )
    ##########     ##########     ###################          ###########

    # implementation for bucketing and distributed sampling
    sampler = DistributedBatchSamplerSimilarLength(dataset, num_replicas=size, rank=rank, shuffle=False, batch_size=batch_size, task=task, seed=m_config["seed"])
    collator = PruningCollator(t_config)
    loader = DataLoader(
        dataset, 
        batch_sampler=sampler, 
        # collate_fn=collator 
        collate_fn=collator # because the load_configuration() does the encoding on preprocessing
    )

    model = model.cuda(rank)

    if _distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if head_mask is not None:
        head_mask = head_mask.cuda(rank)

    ## Στο igspp δεν περναω μασκα. Εχω ηδη prunaρει το μοντελο κι παιρνω το Inference του.
    interpret = igspp_get_interpretability_scores if igspp or isp else get_interpretability_scores
    predict = igspp_get_predictions if igspp  or isp else get_predictions

    if inference:
        total_attributions = []
    else:
        total_attributions = interpret(
            model,
            rank,
            loader,
            _distributed=_distributed,
            target=m_config["target"], 
            head_mask=head_mask
        )

    # total_attentions, pred_input_ids = get_predictions(
    total_attentions, preds, labels = predict(
        model,
        rank,
        loader, 
        _distributed=_distributed,
        head_mask=head_mask,
        inference=inference
    )

    # Because we have distributed processing, somehow i have to secure that the index of the attentions and 
    # the attributions are for the same sentence, because there is the possibility one process finish both
    # computations and the other only the 'attention' one, and then another again both, so the second will not
    # manage to send through pipe the corresponding attributions.
    # I decided to have only one pipe, and send the data after i have collected in an object like 'list'.
    # Finally, the indices of torch.distributed.gather_object() gather_list corresponds
    # to rank of the process, i.e. the GPU. So, i can send the computations seperately. If it runs slowly,
    # i will do it.

    metric = []

    if not inference:
        if igspp or isp:
            metric = igspp_cor(total_attentions, total_attributions, head_mask)    
        else:
            mi = b_mi
            corr = cor 

            metric = mi(total_attentions, total_attributions) if m_config["criterion"] == "mi" else corr(total_attentions, total_attributions)


    pipe_data = [metric, preds, labels]

    if rank == 0:
        gathered_data = [[] for _ in range(size)]
        torch.distributed.gather_object(pipe_data, object_gather_list=gathered_data)

        final_metric = []
        if _distributed:
            list_0, list_1 = gathered_data[0], gathered_data[1]
            
            if not inference:
                final_metric = torch.stack((list_0[0], list_1[0])).mean(dim=0).clone().detach()

            preds = torch.cat((list_0[1], list_1[1]), dim=0)
            labels = torch.cat((list_0[2], list_1[2]), dim=0)

        else:
            list_0 = gathered_data[0]
            
            if not inference:
                final_metric = list_0[0]

            preds = list_0[1]
            labels = list_0[2]

        del total_attributions, total_attentions
        torch.cuda.empty_cache()

        return final_metric, preds, labels
    else:
        torch.distributed.gather_object(pipe_data)
        del total_attributions, total_attentions
        torch.cuda.empty_cache()
        return

def dist_attribution(m_config, t_config, world_size, head_mask=None, inference=False):

    WORLD_SIZE = world_size
    
    # define variable for time measurement
    start_time = time.time()
    
    size = WORLD_SIZE

    wait = mp.Event()
    try:
        queue = mp.Queue()    

        processes = []
        for rank in range(size):
            p = mp.Process(target=init_process, args=(rank, size, run, queue if rank == 0 else None, wait, head_mask, inference, m_config, t_config))
            p.start()
            processes.append(p)

        result_arrays = []
        nb_ended_workers = 0
        while nb_ended_workers != 1:
            worker_result = queue.get()
            if worker_result is None:
                nb_ended_workers += 1
            else:
                result_arrays.append(worker_result)

        wait.set()
        
        for p in processes:
            p.join()

        print(f"--- {time.time() - start_time} seconds ---")
        return result_arrays

    except Exception as e:
        traceback.print_exc()
    finally:
        pass

if __name__ == '__main__':
    dist_attribution()