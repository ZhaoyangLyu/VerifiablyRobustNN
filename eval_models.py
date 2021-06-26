import numpy as np
import torch

from bound_layers import BoundSequential
from config import load_config, get_path, config_modelloader, config_dataloader

import json
import argparse
import pdb
import time
import os

from custom_attacks import eval_pgd_restart_v2


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name=''):
        self.reset()
        # name is the name of the quantity that we want to record, used as tag in tensorboard
        self.name = name
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.total_time = 0
        self.avg_time = 0
    def update(self, val, n=1, time = None, summary_writer=None, global_step=None):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if not time is None:
            self.total_time = self.total_time + time
            self.avg_time = self.total_time / self.count
        if not summary_writer is None:
            # record the val in tensorboard
            summary_writer.add_scalar(self.name, val, global_step=global_step)

def generate_C(data, labels, num_class=10):
    c = torch.eye(num_class).type_as(data)[labels].unsqueeze(1) - torch.eye(num_class).type_as(data).unsqueeze(0) 
    # remove specifications to self
    I = (~(labels.data.unsqueeze(1) == torch.arange(num_class).type_as(labels.data).unsqueeze(0)))
    c = (c[I].view(data.size(0),num_class-1,num_class))
    return c

def get_robust_error(ub, lb, labels, apply_C=True):
    # ub and lb are of shape (batch, num_class) or (batch, num_class-1)
    if apply_C:
        lb_min = lb.min(dim=1)[0] # of shape batch
        # pdb.set_trace()
        error = (lb_min<0).float().mean()
    else:
        ub_ori = ub.detach().clone()

        batch = lb.shape[0]
        idx = torch.arange(batch).to(ub.device)
        true_lb = lb[idx, labels] # of shape (batch)

        ub_ori[idx, labels] = ub_ori.min() - 1e8
        other_ub_max = ub_ori.max(dim=1)[0] # of shape (batch)

        error = (other_ub_max > true_lb).float().mean()
    return error


def test_robust_error(model, model_name, labels, x, x_U, x_L, C, eps, norm, batch, apply_C, print_info, last_epoch,i,
                    ibp_error, ibp_crown_error, lbp_error, lbp_crown_error, crown_error, bound_type=['ibp', 'crown-ibp', 'lbp', 'crown-lbp', 'crown']):
    if 'ibp' in bound_type:
        start = time.time()
        h_U_int, h_L_int, losses, unstable, dead, alive = model.interval_range(
            norm=norm, x_U=x_U, x_L=x_L, eps=eps, C=C)
        error = get_robust_error(h_U_int, h_L_int, labels, apply_C=apply_C)
        ibp_time = time.time() - start
        ibp_error.update(error, batch, time=ibp_time)
    
    if 'crown-ibp' in bound_type:
        start = time.time()
        crown_ibp_ub, ibp_ubias, crown_ibp_lb, ibp_lbias = model(norm=norm, x_U=x_U, x_L=x_L, 
                        eps=eps, C=C, upper=True, lower=True, method_opt="backward_range")
        error = get_robust_error(crown_ibp_ub, crown_ibp_lb, labels, apply_C=apply_C)
        crown_ibp_time = time.time() - start
        ibp_crown_error.update(error, batch, time = crown_ibp_time+ibp_time)

    if 'lbp' in bound_type:
        start = time.time()
        h_U_linear,h_L_linear = model.linear_range(x_U=x_U, x_L=x_L,
                x0=x, norm=norm, eps=eps, C=C)
        error = get_robust_error(h_U_linear, h_L_linear, labels, apply_C=apply_C)
        lbp_time = time.time() - start
        lbp_error.update(error, batch, time=lbp_time)

    if 'crown-lbp' in bound_type:
        start = time.time()
        crown_lbp_ub, lbp_ubias, crown_lbp_lb, lbp_lbias = model(norm=norm, x_U=x_U, x_L=x_L, 
                        eps=eps, C=C, upper=True, lower=True, method_opt="backward_range")
        error = get_robust_error(crown_lbp_ub, crown_lbp_lb, labels, apply_C=apply_C)
        crown_lbp_time = time.time() - start
        lbp_crown_error.update(error, batch, time=crown_lbp_time+lbp_time)

    if 'crown' in bound_type:
        start = time.time()
        crown_ub, _, crown_lb, _ = model(norm=norm, x_U=x_U, x_L=x_L, eps=eps, 
                    C=C, upper=True, lower=True, method_opt="full_backward_range")
        error = get_robust_error(crown_ub, crown_lb, labels, apply_C=apply_C)
        crown_time = time.time() - start
        crown_error.update(error, batch, time=crown_time)

    if print_info and not last_epoch:
        print('%s Progress %d / %d' % (model_name, i, length))
        if 'ibp' in bound_type:
            print('IBP      :\t LB %3.4f \t| UB %.4f\t Verified error: %.4f\t Avg time: %.6f' % (h_L_int.mean(), h_U_int.mean(), ibp_error.avg*100, ibp_error.avg_time))
        if 'crown-ibp' in bound_type:
            print('CROWN-IBP:\t LB %3.4f \t| UB %.4f\t Verified error: %.4f\t Avg time: %.6f' % (crown_ibp_lb.mean(), crown_ibp_ub.mean(), ibp_crown_error.avg*100, ibp_crown_error.avg_time))
        if 'lbp' in bound_type:
            print('LBP      :\t LB %3.4f \t| UB %.4f\t Verified error: %.4f\t Avg time: %.6f' % (h_L_linear.mean(), h_U_linear.mean(), lbp_error.avg*100, lbp_error.avg_time))
        if 'crown-lbp' in bound_type:
            print('CROWN-LBP:\t LB %3.4f \t| UB %.4f\t Verified error: %.4f\t Avg time: %.6f' % (crown_lbp_lb.mean(), crown_lbp_ub.mean(), lbp_crown_error.avg*100, lbp_crown_error.avg_time))
        if 'crown' in bound_type:
            print('CROWN    :\t LB %3.4f \t| UB %.4f\t Verified error: %.4f\t Avg time: %.6f' % (crown_lb.mean(), crown_ub.mean(), crown_error.avg*100, crown_error.avg_time))

    if last_epoch:
        print('%s Final Result' % model_name)
        if 'ibp' in bound_type:
            print('IBP      :\t Verified error: %.4f\t Avg time: %.6f' % (ibp_error.avg*100, ibp_error.avg_time))
        if 'crown-ibp' in bound_type:
            print('CROWN-IBP:\t Verified error: %.4f\t Avg time: %.6f' % (ibp_crown_error.avg*100, ibp_crown_error.avg_time))
        if 'lbp' in bound_type:
            print('LBP      :\t Verified error: %.4f\t Avg time: %.6f' % (lbp_error.avg*100, lbp_error.avg_time))
        if 'crown-lbp' in bound_type:
            print('CROWN-LBP:\t Verified error: %.4f\t Avg time: %.6f' % (lbp_crown_error.avg*100, lbp_crown_error.avg_time))
        if 'crown' in bound_type:
            print('CROWN    :\t Verified error: %.4f\t Avg time: %.6f' % (crown_error.avg*100, crown_error.avg_time))

    return ibp_error, ibp_crown_error, lbp_error, lbp_crown_error, crown_error

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.json', 
                        help='JSON file for configuration')
    parser.add_argument('-e', '--epsilon', type=float, default=0.1,
                        help='Evaluate the model at radius of epsilon. Only specify epsilon for MNIST dataset.')
    parser.add_argument('-d', '--device', type=int, default=0,
                        help='The gpu index to perform computation on')
    parser.add_argument('-b', '--batch_size', type=int, default=128,
                        help='The batch_size for evaluation. This is only valid for pgd attack evaluations. We always use batchsize 1 when computing verified errors')
    parser.add_argument('--skip_pgd_attack', action='store_true', 
                        help='Whether to skip the pgd attack evaluation. If true, we directly compute verified errors. (default: False)')
    parser.add_argument('--pgd_attack_step', type=int, default=200,
                        help='Number of pgd attack steps. (default: 200)')
    args = parser.parse_args()

    with torch.no_grad():
        with open(args.config) as f:
            data = f.read()
        config = json.loads(data)
        print('The configuration is:')
        print(json.dumps(config, indent=4))

        assert config['dataset'] in ['mnist' , 'cifar']
        assert len(config['models']) == 1 

        eval_config = config["eval_params"]
        train_config = config["training_params"]

        device = torch.device('cuda:%d' % args.device)
        eval_config["loader_params"]['test_batch_size'] = args.batch_size
        
        models_path = config['models_path']
        models_name = 'network'

        # set up model file name, epsilon, and input shape
        if config['dataset'] == 'mnist':
            assert args.epsilon in eval_config['epsilon']
            eval_config['epsilon'] = args.epsilon
            input_shape = [2,1,28,28]
            is_cifar_dataset = False
            models_path = os.path.join(models_path, config['models'][0]['model_id']+'_best.pth_test_eps_%.5f' % args.epsilon)
        elif config['dataset'] == 'cifar':
            assert len(eval_config['epsilon']) == 0
            eval_config['epsilon'] = eval_config['epsilon'][0]
            input_shape = [2,3,32,32]
            is_cifar_dataset = True
            raise Exception('model name construction has not been implemented for cifar dataset')

        # set up slope for leaky relu and param relu activation
        eval_config['method_params']['bound_opts'] = {}
        eval_config['method_params']['bound_opts']['activation'] = train_config['method_params']['bound_opts']['activation']
        if 'neg_slope' in train_config['method_params']['bound_opts'].keys():
            eval_config['method_params']['bound_opts']['neg_slope'] = train_config['method_params']['bound_opts']['neg_slope']
        slope_schedule_setting = train_config.get('slope_schedule', {'slope_schedule':False})
        if slope_schedule_setting['slope_schedule']:
            eval_config['method_params']['bound_opts']['neg_slope'] = slope_schedule_setting['end_slope']
        
        # setup the tight strategy to compute bounding lines
        # relu: zero-lb, leaky_relu:zero-lb, leaky_relu_step or param_leaky_relu_step: neg-slope-lb
        # this is the tight strategy to choose bounding lines descrobed in the paper
        if eval_config["method_params"]["bound_opts"]['activation'] in ['relu', 'leaky_relu']:
            eval_config['method_params']['bound_opts']['zero-lb']=True
        elif eval_config["method_params"]["bound_opts"]['activation'] in ['leaky_relu_step', 'param_leaky_relu_step']:
            eval_config['method_params']['bound_opts']['neg-slope-lb']=True
        else:
            raise Exception('The activation %s is not supported' % eval_config["method_params"]["bound_opts"]['activation'])

        # build model and load model state dict
        models, _ = config_modelloader(config, load_pretrain = False)
        models = models[0]
        models = BoundSequential.convert(models, eval_config["method_params"]["bound_opts"]).to(device)
        param_pending_init = 'param' in eval_config["method_params"]["bound_opts"]['activation']
        if param_pending_init:
            input_data = torch.rand(*input_shape).to(device)
            _ = models(input_data, method_opt='forward')
        models.load_state_dict(torch.load(models_path, map_location=device)['state_dict'])
        print('Model state dict loaded from file', models_path)
        models.update_parameter()

        _, loader = config_dataloader(config, **eval_config["loader_params"])

        # test model clean acc
        verbose = False
        model_acc = AverageMeter()
        for idx, (data, labels) in enumerate(loader):
            data, labels = data.to(device), labels.to(device)
            batch = data.shape[0]
            out = models(data, method_opt="forward")
            pred = torch.argmax(out, dim=1)
            acc = (pred == labels).float().mean()
            model_acc.update(acc, batch)
            if verbose:
                print('Progress: %.2f [%d / %d]' % (idx/len(loader), idx, len(loader)))
                print('%s accuracy on clean data is %.2f, err is %.2f' % (models_name, acc*100, 100-acc*100))
        print('%s accuracy on clean test data is %.2f, err is %.2f' % (models_name, 
                model_acc.avg*100, 100-model_acc.avg*100))


        # setup data statistics
        # this part is used for both pgd attack and computing verified errors 
        std = torch.tensor(loader.std).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)
        mean = torch.tensor(loader.mean).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)
        data_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1)).to(device)
        data_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1)).to(device)

        if not args.skip_pgd_attack:
            custom_clamp = is_cifar_dataset # only for cifar dataset
            # because for cifar dataset, we normalize the input by its channelwise mean and std
            # but for mnist dataset, we directly use the original input which range from 0 to 1 
            test_step_size = eval_config['epsilon'] / 10. if is_cifar_dataset else 0.01#
            test_num_steps = args.pgd_attack_step
            print('Begin evaluating the model at epsilon %.5f using pgd attack of %d steps (step size %.5f) with 10 random starts'
                    % (eval_config['epsilon'], test_num_steps, test_step_size))
            pdg_error = eval_pgd_restart_v2(models, device, loader, 'PGD',
                                test_epsilon=eval_config['epsilon'], test_num_steps=test_num_steps,
                                test_step_size=test_step_size, num_classes=10,
                                data_max=data_max, data_min=data_min, mean=mean, std=std,
                                restarts=10, custom_clamp=custom_clamp)


        # begin compute verified errors
        num_class = 10
        norm = np.inf
        apply_C = True
        report_interval = 10

        # The bound computation methods to use
        # only include crown bounds for very small networks because crown is very memory consuming 
        bound_type=  ['ibp', 'crown-ibp', 'lbp', 'crown-lbp']#, 'crown']

        # reset loader batch size to 1
        eval_config["loader_params"]['test_batch_size'] = 1
        _, loader = config_dataloader(config, **eval_config["loader_params"])
        length = len(loader)


        eps = eval_config['epsilon']
        ibp_error = AverageMeter()
        ibp_crown_error = AverageMeter()
        lbp_error = AverageMeter()
        lbp_crown_error = AverageMeter()
        crown_error = AverageMeter()

        print('Begin computing verified errors for the model at epsilon %.5f' % eps)
        for i, (data, labels) in enumerate(loader):
            data, labels = data.to(device), labels.to(device)
            batch = data.shape[0]
            
            x_U = torch.min(data + (eps / std), data_max)
            x_L = torch.max(data - (eps / std), data_min)

            if apply_C:
                C = generate_C(data, labels, num_class=num_class)
            else:
                C = torch.eye(num_class).unsqueeze(0).to(device)

            print_info = (i % report_interval == 0 or i==length-1)
            last_epoch = i==length-1
            ibp_error, ibp_crown_error, lbp_error, lbp_crown_error, crown_error = test_robust_error(
                            models, models_name, labels, data, x_U, x_L, C, eps, norm, batch, apply_C, print_info, last_epoch,i,
                            ibp_error, ibp_crown_error, lbp_error, lbp_crown_error, crown_error, bound_type=bound_type)
        
        if not args.skip_pgd_attack:
            print('PGD attack error is %.4f' % pdg_error)