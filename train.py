## Copyright (C) 2019, Huan Zhang <huan@huan-zhang.com>
##                     Hongge Chen <chenhg@mit.edu>
##                     Chaowei Xiao <xiaocw@umich.edu>
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
##
import sys
import os
import copy
import torch
from torch.nn import Sequential, Linear, ReLU, CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from datasets import loaders
from bound_layers import BoundSequential, BoundLinear, BoundConv2d, BoundDataParallel, BoundReLU, BoundLeakyReLU
from bound_param_ramp import BoundLeakyReLUStep
import torch.optim as optim
# from gpu_profile import gpu_profile
import time
from datetime import datetime
# from convex_adversarial import DualNetwork
from eps_scheduler import EpsilonScheduler
from config import load_config, get_path, config_modelloader, config_dataloader, update_dict
from argparser import argparser
import json

from utils.anneal_weight_bound import anneal_weight

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
    def update(self, val, n=1, summary_writer=None, global_step=None):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if not summary_writer is None:
            # record the val in tensorboard
            summary_writer.add_scalar(self.name, val, global_step=global_step)


class Logger(object):
    def __init__(self, log_file = None):
        self.log_file = log_file

    def log(self, *args, **kwargs):
        print(*args, **kwargs)
        if self.log_file:
            print(*args, **kwargs, file = self.log_file)
            self.log_file.flush()

def record_net_status(net, writer, global_step, disable_multi_gpu):
    # record neurons status (dead, alive, unstable) in each layer of the network
    with torch.no_grad():
        if writer is None:
            return 0
        layer = 0
        loss = 0
        if isinstance(net, BoundDataParallel):
            # net._replicas is a list, each element is a model on the corresponding device
            # we only record net status for one model
            if not disable_multi_gpu:
                net = net._replicas[-1]
            else:
                net = net.module
        for module in net:
            if isinstance(module, BoundReLU) or isinstance(module, BoundLeakyReLU):
                layer = layer+1
                lower_mean = module.lower_l.mean().item()
                upper_mean = module.upper_u.mean().item()
                writer.add_scalar('Bound/Lower Bound/Layer %d' % layer, 
                                    lower_mean, global_step)
                writer.add_scalar('Bound/Upper Bound/Layer %d' % layer, 
                                    upper_mean, global_step)
                writer.add_scalar('Bound/Bound Gap/Layer %d' % layer, 
                                    upper_mean-lower_mean, global_step)

                loss = loss + upper_mean - lower_mean
                
                writer.add_scalar('Neuron Status/Alive Percent/Layer %d' % layer, module.alive.item(), global_step)
                writer.add_scalar('Neuron Status/Dead Percent/Layer %d' % layer, module.dead.item(), global_step)
                writer.add_scalar('Neuron Status/Unstable Percent/Layer %d' % layer, module.unstable.item(), global_step)

                unstable = (module.lower_l<0) * (module.upper_u>0)
                u_l = (module.upper_u[unstable] / (-module.lower_l[unstable])).mean()
                writer.add_scalar('Neuron Status/Unstable u_l/Layer %d' % layer, u_l.item(), global_step)
            elif isinstance(module, BoundLeakyReLUStep):
                layer = layer+1
                lower_mean = module.lower_l.mean().item()
                upper_mean = module.upper_u.mean().item()
                writer.add_scalar('Bound/Lower Bound/Layer %d' % layer, 
                                    lower_mean, global_step)
                writer.add_scalar('Bound/Upper Bound/Layer %d' % layer, 
                                    upper_mean, global_step)
                writer.add_scalar('Bound/Bound Gap/Layer %d' % layer, 
                                    upper_mean-lower_mean, global_step)

                loss = loss + upper_mean - lower_mean
                
                for key in module.neuron_status:
                    writer.add_scalar('Neuron Status/%s Percent/Layer %d' % (key, layer), module.neuron_status[key].item(), global_step)
    return loss

def record_mean_activation(net, dataloader, device, include_minus_values):
    # record the mean activation in each layer for all samples in the training set
    # this mean activation will be used as the initial value of the bending point r in the ParamRamp activation
    with torch.no_grad():
        if isinstance(net, BoundDataParallel):
            net = net.module
        for m in net:
            if isinstance(m, BoundLeakyReLUStep):
                m.record_mean_activation = True
                m.include_minus_values = include_minus_values
                m.mean_act = 0
                m.num_examples = 0

        for data, _ in dataloader:
            data = data.to(device)
            _ = net(data, method_opt='forward')

        net.use_mean_act_as_param()

        for m in net:
            if isinstance(m, BoundLeakyReLUStep):
                m.record_mean_activation = False
                
        net.reset_ignore_right_step(False)
    return 0

def Train(model, t, loader, eps_scheduler, max_eps, norm, logger, verbose, train, opt, method, 
            disable_multi_gpu = False, target_eps=None, tensorboard_writer=None, after_crown_or_lbp_settings={}, **kwargs):
    # if train=True, use training mode
    # if train=False, use test mode, no back prop
    compute_ibp_only = False

    num_class = 10
    losses = AverageMeter('Loss/Total Loss')
    l1_losses = AverageMeter('Loss/L1 Loss')
    errors = AverageMeter('Error/Clean Error')
    robust_errors = AverageMeter('Error/Robust Error')
    regular_ce_losses = AverageMeter('Loss/Regular CE Loss')
    robust_ce_losses = AverageMeter('Loss/Robust Loss')
    relu_activities = AverageMeter()
    bound_bias = AverageMeter()
    bound_diff = AverageMeter()
    unstable_neurons = AverageMeter()
    dead_neurons = AverageMeter()
    alive_neurons = AverageMeter()
    batch_time = AverageMeter()
    batch_multiplier = kwargs.get("batch_multiplier", 1)  
    kappa = 1
    beta = 1
    if train:
        model.train() 
    else:
        model.eval()

    # pregenerate the array for specifications, will be used for scatter
    sa = np.zeros((num_class, num_class - 1), dtype = np.int32)
    for i in range(sa.shape[0]):
        for j in range(sa.shape[1]):
            if j < i:
                sa[i][j] = j
            else:
                sa[i][j] = j + 1
    sa = torch.LongTensor(sa) 

    batch_size = loader.batch_size * batch_multiplier
    if batch_multiplier > 1 and train:
        logger.log('Warning: Large batch training. The equivalent batch size is {} * {} = {}.'.format(batch_multiplier, loader.batch_size, batch_size))
    # per-channel std and mean
    std = torch.tensor(loader.std).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    mean = torch.tensor(loader.mean).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
 
    model_range = 0.0
    # t is the epoch index 
    end_eps = eps_scheduler.get_eps(t+1, 0)
    if end_eps < np.finfo(np.float32).tiny:
        logger.log('eps {} close to 0, using natural training'.format(end_eps))
        method = "natural"

    need_to_replicate = True # indicate whether need to (replicate) update the model on multi gpus during the forward pass
    for i, (data, labels) in enumerate(loader): 
        start = time.time()
        eps = eps_scheduler.get_eps(t, int(i//batch_multiplier)) 
        if (not train) and (not target_eps is None):
            # during the evaluation phase, we use the smaller value in [eps, target_eps] for evaluation
            ori_eps = eps
            # eps is constant during the evaluation phase, because the eps_scheduler we send is a constant scheduler in evalution phase
            eps = min(eps, target_eps) 
        global_step = eps_scheduler.get_global_step(t, int(i//batch_multiplier))
        if not tensorboard_writer is None:
            tensorboard_writer.add_scalar('Training Schedule/Eps', eps, global_step)
            tensorboard_writer.add_scalar('Training Schedule/Epoch', t, global_step)

        if train and i % batch_multiplier == 0:   
            opt.zero_grad()
        # generate specifications
        c = torch.eye(num_class).type_as(data)[labels].unsqueeze(1) - torch.eye(num_class).type_as(data).unsqueeze(0) 
        # remove specifications to self
        I = (~(labels.data.unsqueeze(1) == torch.arange(num_class).type_as(labels.data).unsqueeze(0)))
        c = (c[I].view(data.size(0),num_class-1,num_class))
        # scatter matrix to avoid compute margin to self
        sa_labels = sa[labels]
        # storing computed lower bounds after scatter
        lb_s = torch.zeros(data.size(0), num_class)
        ub_s = torch.zeros(data.size(0), num_class)

        # Assume unnormalized data is from range 0 - 1
        if kwargs["bounded_input"]:
            if norm != np.inf:
                raise ValueError("bounded input only makes sense for Linf perturbation. "
                                 "Please set the bounded_input option to false.")
            data_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
            data_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))
            data_ub = torch.min(data + (eps / std), data_max)
            data_lb = torch.max(data - (eps / std), data_min)
        else:
            if norm == np.inf:
                data_ub = data + (eps / std)
                data_lb = data - (eps / std)
            else:
                # For other norms, eps will be used instead.
                data_ub = data_lb = data

        # move data to the corresponding device
        device = list(model.parameters())[0].device
        data = data.to(device)
        data_ub = data_ub.to(device)
        data_lb = data_lb.to(device)
        labels = labels.to(device)
        c = c.to(device)
        sa_labels = sa_labels.to(device)
        lb_s = lb_s.to(device)
        ub_s = ub_s.to(device)
        # convert epsilon to a tensor
        eps_tensor = data.new(1)
        eps_tensor[0] = eps

        # compute model output
        output = model(data, method_opt="forward", disable_multi_gpu = (method == "natural") or disable_multi_gpu, 
                        need_to_replicate=need_to_replicate)
        need_to_replicate = False # only need to replicate at the beginning or opt.step() is called

        regular_ce = CrossEntropyLoss()(output, labels)
        regular_ce_losses.update(regular_ce.cpu().detach().numpy(), data.size(0), 
                        summary_writer=tensorboard_writer, global_step=global_step)
        errors.update(torch.sum(torch.argmax(output, dim=1)!=labels).cpu().detach().numpy()/data.size(0), 
                        data.size(0), summary_writer=tensorboard_writer, global_step=global_step)
        # get range statistic
        model_range = output.max().detach().cpu().item() - output.min().detach().cpu().item()
        
        
        
        # compute model output bounds
        if verbose or method != "natural":
            if kwargs["bound_type"] == "interval":
                ub, lb, relu_activity, unstable, dead, alive = model(norm=norm, x_U=data_ub, x_L=data_lb, 
                                                                eps=eps, C=c, method_opt="interval_range")
            elif kwargs['bound_type'] == 'lbp':
                # the closed form bounds will be computed by x_U and x_L if they are not None and norm=np.inf
                # x0 and eps will not be used in this case
                ub, lb = model(x_U=data_ub, x_L=data_lb, x0=data, norm=norm, eps=eps, C=c, method_opt="lbp")
                unstable = dead = alive = relu_activity = torch.tensor([0])
            elif kwargs["bound_type"] == "crown-full":
                _, _, lb, _ = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, upper=False, lower=True, 
                                    method_opt="full_backward_range")
                unstable = dead = alive = relu_activity = torch.tensor([0])
            elif "lbp-interval" in kwargs["bound_type"] and 'crown-lbp' not in kwargs["bound_type"]:
                _, ilb, relu_activity, unstable, dead, alive = model(norm=norm, x_U=data_ub, x_L=data_lb, 
                                                            eps=eps, C=c, method_opt="interval_range")
                # we design several different combination schemes of interval(ibp) bound and lbp bound for training
                if 'max' in kwargs["bound_type"] or (not train):
                    # choose max between interval and lbp as lower bound
                    _, llb = model(x_U=data_ub, x_L=data_lb, x0=data, norm=norm, eps=eps, C=c, method_opt="lbp")
                    diff = (llb - ilb).sum().item()
                    bound_diff.update(diff / data.size(0), data.size(0))
                    lb = torch.max(ilb, llb)
                elif 'anneal' in kwargs["bound_type"]:
                    _, llb = model(x_U=data_ub, x_L=data_lb, x0=data, norm=norm, eps=eps, C=c, method_opt="lbp")
                    lb, weight = anneal_weight([ilb, llb], min_weight=0.05, temp=1)
                    llb_weight = weight[:,:,1].mean().item()
                    bound_diff.update(llb_weight, data.size(0))
                elif 'mean' in kwargs["bound_type"]:
                    # use mean of interval and lbp as lower bound
                    _, llb = model(x_U=data_ub, x_L=data_lb, x0=data, norm=norm, eps=eps, C=c, method_opt="lbp")
                    diff = (llb - ilb).sum().item()
                    bound_diff.update(diff / data.size(0), data.size(0))
                    lb = (ilb+ llb)/2
                else:
                    # choose a convex combination of interval and lbp as lower bound
                    # as eps increase from 0 to target eps in the training process
                    # lb transits from lbp to interval lower bound
                    lbp_final_beta = kwargs['final-beta'] # default is 0
                    if train or target_eps is None:
                        beta = (max_eps - eps * (1.0 - lbp_final_beta)) / max_eps
                    else:
                        beta = (max_eps - ori_eps * (1.0 - lbp_final_beta)) / max_eps
                    # beta start from 1, end with 0 during training
                    tensorboard_writer.add_scalar('Training Schedule/beta', beta, global_step)
                    if beta < 1e-5:
                        lb = ilb
                    else:
                        _, llb = model(x_U=data_ub, x_L=data_lb, x0=data, norm=norm, eps=eps, C=c, method_opt="lbp")
                        diff = (llb - ilb).sum().item()
                        bound_diff.update(diff / data.size(0), data.size(0))
                        # lb = torch.max(lb, clb)
                        lb = llb * beta + ilb * (1 - beta)
                
            elif "crown-interval" in kwargs["bound_type"] or "crown-lbp-interval" in kwargs["bound_type"]:
                # in this approach, we first compute final layer bound and intermidiate layer bounds
                # using IBP. Then we use CROWN to compute the final layer bounds given bounds of intermediate layers
                # the final layer bound is then given by a convex combination of the ibp bound and crown bound
                # Enable multi-GPU only for the computationally expensive CROWN-IBP bounds, 
                # not for regular forward propagation and IBP because the communication overhead can outweigh benefits, giving little speedup. 
                
                if 'convex' in kwargs["bound_type"]:
                    crown_final_beta = kwargs['final-beta']
                    if train or target_eps is None:
                        beta = (max_eps - eps * (1.0 - crown_final_beta)) / max_eps
                    else:
                        # in this case, eps have been reset to min(eps, target_eps)
                        beta = (max_eps - ori_eps * (1.0 - crown_final_beta)) / max_eps

                if 'convex' in kwargs["bound_type"] and beta < 1e-5:# and train:
                    # in this case we only need interval bound, bound computation can be conducted on one gpu
                    # we don't need to compute lbp bound in this case
                    ub, ilb, relu_activity, unstable, dead, alive = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, 
                            C=c, method_opt="interval_range", disable_multi_gpu = not after_crown_or_lbp_settings['multi_gpu'])
                    disable_multi_gpu = not after_crown_or_lbp_settings['multi_gpu']
                    compute_ibp_only = True 
                    # indicate that we have pass the phase where we use convex combination of crown and ibp or lbp and ibp
                    # we only compute ibp in later iters
                else:
                    # use multigpu to compute IBP bounds
                    ub, ilb, relu_activity, unstable, dead, alive = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, 
                                                                    method_opt="interval_range")
                    if "crown-lbp-interval" in kwargs["bound_type"]:
                        # use LBP to compute bounds for intermediate layers 
                        if 'detach' in kwargs["bound_type"]: # crown-lbp-intervaL-detach
                            with torch.no_grad():
                                _, _ = model(x_U=data_ub, x_L=data_lb, x0=data, norm=norm, eps=eps, C=c, method_opt="lbp")
                        else: # crown-lbp-intervaL
                            _, _ = model(x_U=data_ub, x_L=data_lb, x0=data, norm=norm, eps=eps, C=c, method_opt="lbp")
                
                if 'max' in kwargs["bound_type"]:# or (not train):
                    # get the CROWN bound using interval bounds 
                    _, _, clb, bias = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="backward_range")
                    bound_bias.update(bias.sum() / data.size(0))
                    # how much better is crown-ibp better than ibp?
                    diff = (clb - ilb).sum().item()
                    bound_diff.update(diff / data.size(0), data.size(0))
                    if not tensorboard_writer is None:
                        tensorboard_writer.add_scalar('Bound Comparison/crown - interval', bound_diff.val, global_step)
                    lb = torch.max(ilb, clb)
                elif 'anneal' in kwargs["bound_type"]:
                    _, _, clb, bias = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="backward_range")
                    bound_bias.update(bias.sum() / data.size(0))
                    lb, weight = anneal_weight([ilb, clb], min_weight=0.05, temp=1)
                    clb_weight = weight[:,:,1].mean().item()
                    bound_diff.update(clb_weight, data.size(0))
                else: # convex combination between interval and crown-ibp or crown-lbp
                    if not tensorboard_writer is None:
                        tensorboard_writer.add_scalar('Training Schedule/beta', beta, global_step)
                    if beta < 1e-5:
                        lb = ilb
                    else:
                        if kwargs["runnerup_only"]:
                            # regenerate a smaller c, with just the runner-up prediction
                            # mask ground truthlabel output, select the second largest class
                            # print(output)
                            # torch.set_printoptions(threshold=5000)
                            masked_output = output.detach().scatter(1, labels.unsqueeze(-1), -100)
                            # print(masked_output)
                            # location of the runner up prediction
                            runner_up = masked_output.max(1)[1]
                            # print(runner_up)
                            # print(labels)
                            # get margin from the groud-truth to runner-up only
                            runnerup_c = torch.eye(num_class).type_as(data)[labels]
                            # print(runnerup_c)
                            # set the runner up location to -
                            runnerup_c.scatter_(1, runner_up.unsqueeze(-1), -1)
                            runnerup_c = runnerup_c.unsqueeze(1).detach()
                            # print(runnerup_c)
                            # get the bound for runnerup_c
                            _, _, clb, bias = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="backward_range")
                            clb = clb.expand(clb.size(0), num_class - 1)
                        else:
                            # get the CROWN bound using interval bounds 
                            _, _, clb, bias = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="backward_range")
                            bound_bias.update(bias.sum() / data.size(0))
                        # how much better is crown-ibp better than ibp?
                        diff = (clb - ilb).sum().item()
                        bound_diff.update(diff / data.size(0), data.size(0))
                        if not tensorboard_writer is None:
                            tensorboard_writer.add_scalar('Bound Comparison/crown - interval', bound_diff.val, global_step)
                        # lb = torch.max(lb, clb)
                        lb = clb * beta + ilb * (1 - beta)
            else:
                raise RuntimeError("Unknown bound_type " + kwargs["bound_type"]) 
            # lb is of shape (batch, num_class-1)
            # lb_s is of shape (batch, num_class) and initially set to 0
            
            # record bounds and neuron status after computing bounds
            bound_gap_loss = record_net_status(model, tensorboard_writer, global_step, disable_multi_gpu)
            if not tensorboard_writer is None:
                tensorboard_writer.add_scalar('Bound/Lower Bound/Final output', lb.mean().item(), global_step)
                tensorboard_writer.add_scalar('Loss/Bound Gap Loss', bound_gap_loss, global_step)

            margin_loss = False
            margin = 1

            if margin_loss:
                # lb_min = lb.min(dim=1)[0] # of shape (batch)
                # if lb > margin, we don't need to futher maximize it
                lb_clamp = torch.clamp(lb, max=margin)
                robust_ce = -lb_clamp.mean()
            else:
                lb = lb_s.scatter(1, sa_labels, lb)
                robust_ce = CrossEntropyLoss()(-lb, labels)

            if kwargs["bound_type"] != "convex-adv":
                relu_activities.update(relu_activity.sum().detach().cpu().item() / data.size(0), data.size(0))
                unstable_neurons.update(unstable.sum().detach().cpu().item() / data.size(0), data.size(0))
                dead_neurons.update(dead.sum().detach().cpu().item() / data.size(0), data.size(0))
                alive_neurons.update(alive.sum().detach().cpu().item() / data.size(0), data.size(0))
        
        
        if method == "robust":
            loss = robust_ce
        elif method == "robust_activity":
            loss = robust_ce + kwargs["activity_reg"] * relu_activity.sum()
        elif method == "natural":
            loss = regular_ce
        elif method == "robust_natural":
            natural_final_factor = kwargs["final-kappa"]
            if train or target_eps is None:
                kappa = (max_eps - eps * (1.0 - natural_final_factor)) / max_eps
            else:
                kappa = (max_eps - ori_eps * (1.0 - natural_final_factor)) / max_eps
            loss = (1-kappa) * robust_ce + kappa * regular_ce
            if not tensorboard_writer is None:
                tensorboard_writer.add_scalar('Training Schedule/kappa', kappa, global_step)
        elif method == 'robust_natural_bound-gap':
            natural_final_factor = kwargs["final-kappa"]
            if train or target_eps is None:
                kappa = (max_eps - eps * (1.0 - natural_final_factor)) / max_eps
            else:
                kappa = (max_eps - ori_eps * (1.0 - natural_final_factor)) / max_eps
            loss = (1-kappa) * robust_ce + kappa * regular_ce
            loss = loss + bound_gap_loss
            if not tensorboard_writer is None:
                tensorboard_writer.add_scalar('Training Schedule/kappa', kappa, global_step)
        else:
            raise ValueError("Unknown method " + method)

        # l1 loss regularization term, not used by default
        if train and kwargs["l1_reg"] > np.finfo(np.float32).tiny:
            reg = kwargs["l1_reg"]
            l1_loss = 0.0
            for name, param in model.named_parameters():
                if 'bias' not in name:
                    l1_loss = l1_loss + torch.sum(torch.abs(param))
            l1_loss = reg * l1_loss
            loss = loss + l1_loss
            l1_losses.update(l1_loss.cpu().detach().numpy(), data.size(0), 
                    summary_writer=tensorboard_writer, global_step=global_step)
        if train:
            loss.backward()
            if (i+1) % batch_multiplier == 0 or i == len(loader) - 1:
                opt.step()
                need_to_replicate = True # need to update the models on multi gpus after opt.step()

            # we should always update parameter no matter whether we opt.step()
            # if we don't update parameter, loss.backward will fail during the second call
            # if isinstance(model, BoundDataParallel):
            #     if model.module.contain_parameterized_act:
            #         model.update_parameter()
            # else:
            #     if model.contain_parameterized_act:
            #         # update useful values in parameterized activation function modules
            #         # since their parameters has changed by opt.step()
            #         model.update_parameter()
            #         # we need to handle the case where we use multi gpus

        losses.update(loss.cpu().detach().numpy(), data.size(0), 
                    summary_writer=tensorboard_writer, global_step=global_step)

        if verbose or method != "natural":
            robust_ce_losses.update(robust_ce.cpu().detach().numpy(), data.size(0), 
                            summary_writer=tensorboard_writer, global_step=global_step)
            # robust_ce_losses.update(robust_ce, data.size(0))
            robust_errors.update(torch.sum((lb<0).any(dim=1)).cpu().detach().numpy() / data.size(0), data.size(0), 
                            summary_writer=tensorboard_writer, global_step=global_step)

        batch_time.update(time.time() - start)
        if i % 50 == 0 and train:
            logger.log(  '[{:2d}:{:4d}]: eps {:4f}  '
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Total Loss {loss.val:.4f} ({loss.avg:.4f})  '
                    'L1 Loss {l1_loss.val:.4f} ({l1_loss.avg:.4f})  '
                    'CE {regular_ce_loss.val:.4f} ({regular_ce_loss.avg:.4f})  '
                    'RCE {robust_ce_loss.val:.4f} ({robust_ce_loss.avg:.4f})  '
                    'Err {errors.val:.4f} ({errors.avg:.4f})  '
                    'Rob Err {robust_errors.val:.4f} ({robust_errors.avg:.4f})  '
                    'Uns {unstable.val:.1f} ({unstable.avg:.1f})  '
                    'Dead {dead.val:.1f} ({dead.avg:.1f})  '
                    'Alive {alive.val:.1f} ({alive.avg:.1f})  '
                    'Tightness {tight.val:.5f} ({tight.avg:.5f})  '
                    'Bias {bias.val:.5f} ({bias.avg:.5f})  '
                    'Diff {diff.val:.5f} ({diff.avg:.5f})  '
                    'R {model_range:.3f}  '
                    'beta {beta:.3f} ({beta:.3f})  '
                    'kappa {kappa:.3f} ({kappa:.3f})  '.format(
                    t, i, eps, batch_time=batch_time,
                    loss=losses, errors=errors, robust_errors = robust_errors, l1_loss = l1_losses,
                    regular_ce_loss = regular_ce_losses, robust_ce_loss = robust_ce_losses, 
                    unstable = unstable_neurons, dead = dead_neurons, alive = alive_neurons,
                    tight = relu_activities, bias = bound_bias, diff = bound_diff,
                    model_range = model_range, 
                    beta=beta, kappa = kappa))
    
                    
    logger.log(  '[FINAL RESULT epoch:{:2d} eps:{:.4f}]: '
        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
        'Total Loss {loss.val:.4f} ({loss.avg:.4f})  '
        'L1 Loss {l1_loss.val:.4f} ({l1_loss.avg:.4f})  '
        'CE {regular_ce_loss.val:.4f} ({regular_ce_loss.avg:.4f})  '
        'RCE {robust_ce_loss.val:.4f} ({robust_ce_loss.avg:.4f})  '
        'Uns {unstable.val:.3f} ({unstable.avg:.3f})  '
        'Dead {dead.val:.1f} ({dead.avg:.1f})  '
        'Alive {alive.val:.1f} ({alive.avg:.1f})  '
        'Tight {tight.val:.5f} ({tight.avg:.5f})  '
        'Bias {bias.val:.5f} ({bias.avg:.5f})  '
        'Diff {diff.val:.5f} ({diff.avg:.5f})  '
        'Err {errors.val:.4f} ({errors.avg:.4f})  '
        'Rob Err {robust_errors.val:.4f} ({robust_errors.avg:.4f})  '
        'R {model_range:.3f}  '
        'beta {beta:.3f} ({beta:.3f})  '
        'kappa {kappa:.3f} ({kappa:.3f})  \n'.format(
        t, eps, batch_time=batch_time,
        loss=losses, errors=errors, robust_errors = robust_errors, l1_loss = l1_losses,
        regular_ce_loss = regular_ce_losses, robust_ce_loss = robust_ce_losses, 
        unstable = unstable_neurons, dead = dead_neurons, alive = alive_neurons,
        tight = relu_activities, bias = bound_bias, diff = bound_diff,
        model_range = model_range, 
        kappa = kappa, beta=beta))
    for i, l in enumerate(model if isinstance(model, BoundSequential) else model.module):
        if isinstance(l, BoundLinear) or isinstance(l, BoundConv2d):
            # compute l-infty induced norm of the weight
            norm = l.weight.data.detach().view(l.weight.size(0), -1).abs().sum(1).max().cpu()
            logger.log('layer {} norm {}'.format(i, norm))
            if not tensorboard_writer is None:
                tensorboard_writer.add_scalar('L-infty Induced Weight Norm/Module ID: %d' % i, norm, global_step)
                if not l.weight.grad is None:
                    tensorboard_writer.add_scalar('Gradient/L2 Norm/Module ID: %d' % i, l.weight.grad.norm().item(), global_step)
                    tensorboard_writer.add_scalar('Gradient/Mean/Module ID: %d' % i, l.weight.grad.mean().item(), global_step)
                    tensorboard_writer.add_scalar('Gradient/Zero Element Percent/Module ID: %d' % i, 
                                                    (l.weight.grad==0).float().mean().item(), global_step)
    
    if method == "natural":
        return errors.avg, errors.avg, disable_multi_gpu, compute_ibp_only
    else:
        return robust_errors.avg, errors.avg, disable_multi_gpu, compute_ibp_only

from shutil import copyfile
import os
import pdb
def main(args):
    config = load_config(args)
    global_train_config = config["training_params"]
    target_eps = config['eval_params']['epsilon'] 
    # target_eps is a list of eps that we want to evalute the trained model at 
    eps_len = len(target_eps)
    models, model_names = config_modelloader(config) 
    tensorboard_log_path = os.path.join(config["path_prefix"], config["models_path"], 'tensorboard_log')
    tensorboard_writer = SummaryWriter(log_dir=tensorboard_log_path)

    os.makedirs(os.path.join(config["path_prefix"], config["models_path"]), exist_ok=True)
    config_path = os.path.join(config["path_prefix"], config["models_path"], os.path.split(args.config)[-1])
    copyfile(args.config, config_path)

    # models is a list of models to be trained
    # every time we train one model, we train next model after training for the current model is finished 
    for model, model_id, model_config in zip(models, model_names, config["models"]):
        # make a copy of global training config, and update per-model config
        train_config = copy.deepcopy(global_train_config)
        if "training_params" in model_config:
            train_config = update_dict(train_config, model_config["training_params"])
        
        cuda_idx = int(train_config['device'])
        if cuda_idx<0:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:%d' % cuda_idx)

        model = BoundSequential.convert(model, train_config["method_params"]["bound_opts"])
        
        # read training parameters from config file
        epochs = train_config["epochs"]
        lr = train_config["lr"]
        weight_decay = train_config["weight_decay"]
        starting_epsilon = train_config["starting_epsilon"]
        end_epsilon = train_config["epsilon"]
        schedule_length = train_config["schedule_length"]
        schedule_start = train_config["schedule_start"]
        optimizer = train_config["optimizer"]
        method = train_config["method"]
        verbose = train_config["verbose"]
        lr_decay_step = train_config["lr_decay_step"]
        lr_decay_milestones = train_config["lr_decay_milestones"]
        lr_decay_factor = train_config["lr_decay_factor"]
        multi_gpu = train_config["multi_gpu"]
        # parameters specific to a training method
        method_param = train_config["method_params"]
        norm = float(train_config["norm"])
        train_data, test_data = config_dataloader(config, **train_config["loader_params"])

        after_crown_or_lbp_settings = train_config.get("after_crown_or_lbp_settings", None)

        if model.contain_pending_init_parameters:
            # let the intermediate activation layers know the size of the input and then initialize their parameter
            temp_data, _ = iter(train_data).next()
            _ = model(temp_data, method_opt="forward")

        
        # replace ParamRamp with LeakyReLU in the first several epochs
        # if we ignore the right part of ParamRamp, it becomes LeakyReLU
        if train_config['step_activation_params']['use_mean_act_as_param']:
            if (train_config['method_params']['bound_opts']['activation'] == 'param_leaky_relu_step' or 
                train_config['method_params']['bound_opts']['activation'] == 'param_slope_leaky_relu_step'):
                model.reset_ignore_right_step(True)
                print("\nWe ignore the right step of the param_leaky_relu_step function initially\n")
            else:
                print('Bound opts:', train_config['method_params']['bound_opts']['activation'])
                raise Exception('Only param_leaky_relu_step supports use mean act as param now')

        # build optimizer
        if optimizer == "adam":
            opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == "sgd":
            opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
        else:
            raise ValueError("Unknown optimizer")
       
        # set up eps scheduler
        batch_multiplier = train_config["method_params"].get("batch_multiplier", 1)
        batch_size = train_data.batch_size * batch_multiplier  
        num_steps_per_epoch = int(np.ceil(1.0 * len(train_data.dataset) / batch_size))
        epsilon_scheduler = EpsilonScheduler(train_config.get("schedule_type", "linear"), schedule_start * num_steps_per_epoch, ((schedule_start + schedule_length) - 1) * num_steps_per_epoch, starting_epsilon, end_epsilon, num_steps_per_epoch)
        max_eps = end_epsilon
        
        # set up slope scheduler for LeakyRelu and ParamRamp
        slope_schedule_setting = train_config.get('slope_schedule', {"slope_schedule":False})
        if slope_schedule_setting['slope_schedule']:
            assert slope_schedule_setting['start_slope'] == train_config['method_params']['bound_opts']['neg_slope']
            if (train_config['method_params']['bound_opts']['activation'] == 'param_leaky_relu_step' or 
                train_config['method_params']['bound_opts']['activation'] == 'leaky_relu_step' or 
                train_config['method_params']['bound_opts']['activation'] == 'leaky_relu'):
                slope_scheduler = EpsilonScheduler(slope_schedule_setting.get("schedule_type", "linear"), 
                                    slope_schedule_setting['schedule_start'], 
                                    slope_schedule_setting['schedule_start'] + slope_schedule_setting['schedule_length'] - 1, 
                                    -slope_schedule_setting['start_slope'], -slope_schedule_setting['end_slope'], 1)
                print('\nUse scheduled slope\n')
                # we use minus values of start_slope and end_slope becuase this scheduler requires end_slope>=start_slope
            else:
                print('The activation function is', train_config['method_params']['bound_opts']['activation'])
                raise Exception('Scheduled slope only support leaky_relu, param_leaky_relu_step or leaky_relu_step')
        
        # set up lr scheduler
        if lr_decay_step:
            # Use StepLR. Decay by lr_decay_factor every lr_decay_step.
            lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=lr_decay_step, gamma=lr_decay_factor)
            lr_decay_milestones = None
        elif lr_decay_milestones:
            # Decay learning rate by lr_decay_factor at a few milestones.
            lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=lr_decay_milestones, gamma=lr_decay_factor)
        else:
            raise ValueError("one of lr_decay_step and lr_decay_milestones must be not empty.")
        
        # print training info and model info
        model_name = get_path(config, model_id, "model", load = False)
        best_model_name = get_path(config, model_id, "best_model", load = False) 
        model_log = get_path(config, model_id, "train_log")
        logger = Logger(open(model_log, "w"))
        logger.log(model_name)
        logger.log("Command line:", " ".join(sys.argv[:]))
        logger.log("training configurations:", json.dumps(train_config, indent=4))
        logger.log("Tensorboard log path:", tensorboard_log_path)
        logger.log("Model structure:")
        logger.log(str(model))
        logger.log("data std:", train_data.std)
        best_err = [np.inf]*eps_len
        recorded_clean_err = [np.inf]*eps_len
        timer = 0.0
        
        # set up DataParallel Model
        if multi_gpu:
            device_ids = train_config['device_ids']
            if isinstance(device_ids, str):
                device_ids = [int(dd) for dd in device_ids.strip('[]').split(',')]
            logger.log("\nUsing multiple GPUs (device_ids: %s) for computing bounds\n" % str(device_ids))
            # pdb.set_trace()
            model = BoundDataParallel(model, device_ids=device_ids)
            device = torch.device('cuda:%d' % device_ids[0]) 
            # at this moment, the model has not been replicated on multi gpus.
            # it will be replicated during the first forward pass
        model.to(device)
        
        
        disable_multi_gpu = False
        # indiate whether need to disbale_multi_gpu, when we use convex combination of crown-lbp and interval
        # in the late training phase, we only need to compute interval bound, at this case we can disable multi gpu

        compute_ibp_only = False
        previous_compute_ibp_only = False
        for t in range(epochs):
            epoch_start_eps = epsilon_scheduler.get_eps(t, 0)
            epoch_end_eps = epsilon_scheduler.get_eps(t+1, 0)
            logger.log("Epoch {}, learning rate {}, epsilon {:.6g} - {:.6g}".format(t, lr_scheduler.get_lr(), epoch_start_eps, epoch_end_eps))
            tensorboard_writer.add_scalar('Training Schedule/Learning Rate', lr_scheduler.get_lr()[-1], global_step=t)

            # record mean activation and use it as the inital value for the bending point r in ParamRamp at the specified epoch 
            if (train_config['step_activation_params']['use_mean_act_as_param'] and 
                t == train_config['step_activation_params']['record_mean_act_epoch']):
                
                logger.log("\nRecord Mean Act as inital Param at epoch %d\n" % t)
                if config['dataset'] == 'mnist':
                    clean_train_data = train_data
                else:
                    temp_train_config = copy.deepcopy(train_config)
                    temp_train_config["loader_params"]['train_random_transform'] = False
                    clean_train_data, _ = config_dataloader(config, **temp_train_config["loader_params"])
                record_mean_activation(model, clean_train_data, device, train_config['step_activation_params']['include_minus_values'])
                logger.log('\nWe restore the right step part of ParamRamp function\n')
            
            # in the late training phase of CROWN-IBP, we do not need to compute crown bounds anymore
            # we only need to compute IBP bounds.
            # In this case, we may not want to use multigpu, or we may want to use less gpus
            # becuase IBP bound computation is not that computational intensive or memory consuming
            # multigpu training may not be faster than single gpu training because the communication overhead  
            if (not previous_compute_ibp_only) and compute_ibp_only:
                if not method_param['batch_multiplier'] == after_crown_or_lbp_settings['batch_multiplier']:
                    method_param['batch_multiplier'] = after_crown_or_lbp_settings['batch_multiplier']
                    logger.log("\nbatch_multiplier has been reset to %d\n" % after_crown_or_lbp_settings['batch_multiplier'])
                # if after_crown_or_lbp_settings['multi_gpu'] == False
                # the model need not to be rebuild
                # the computations will be performed on device_ids[0]
                if after_crown_or_lbp_settings['multi_gpu']:
                    new_device_ids = after_crown_or_lbp_settings['device_ids']
                    if isinstance(new_device_ids, str):
                        if new_device_ids=='same':
                            new_device_ids = device_ids
                        else:
                            new_device_ids = [int(dd) for dd in new_device_ids.strip('[]').split(',')]
                    if not new_device_ids == device_ids:
                        model = BoundDataParallel(model.module, device_ids = after_crown_or_lbp_settings['device_ids'])
                        logger.log("\ndevice_ids has been reset to %s\n" % after_crown_or_lbp_settings['device_ids'])
                if not train_config["loader_params"]['batch_size'] == after_crown_or_lbp_settings['batch_size']:
                    train_config["loader_params"]['batch_size'] = after_crown_or_lbp_settings['batch_size']
                    train_data, _ = config_dataloader(config, **train_config["loader_params"])
                    logger.log("\nTraining batch_size has been reset to %d\n" % after_crown_or_lbp_settings['batch_size'])

            previous_compute_ibp_only = compute_ibp_only

            # obtain and record current step slope value
            if slope_schedule_setting['slope_schedule']:
                current_slope = -slope_scheduler.get_eps(t, 0)
                model.update_slope(current_slope)
                tensorboard_writer.add_scalar('Training Schedule/Slope', current_slope, global_step=t)

            start_time = time.time() 
            robust_err_temp,clean_err_temp,disable_multi_gpu, compute_ibp_only = Train(model, t, train_data, epsilon_scheduler, 
                                    max_eps, norm, logger, verbose, True, opt, method, 
                                    disable_multi_gpu=disable_multi_gpu, tensorboard_writer=tensorboard_writer, 
                                    after_crown_or_lbp_settings=after_crown_or_lbp_settings, **method_param)
            tensorboard_writer.add_scalar('Epoch Error/Train Robust Error', robust_err_temp, global_step=t)
            tensorboard_writer.add_scalar('Epoch Error/Train Error', clean_err_temp, global_step=t)
            tensorboard_writer.add_scalar('Epoch Error/epoch_end_eps', epoch_end_eps, global_step=t)

            # update learning rate
            if lr_decay_step:
                # Use stepLR. Note that we manually set up epoch number here, so the +1 offset.
                lr_scheduler.step(epoch=max(t - (schedule_start + schedule_length - 1) + 1, 0))
            elif lr_decay_milestones:
                # Use MultiStepLR with milestones.
                lr_scheduler.step()
            # log training time
            epoch_time = time.time() - start_time
            timer += epoch_time
            logger.log('Epoch time: {:.4f}, Total time: {:.4f}'.format(epoch_time, timer))
            logger.log("Evaluating...")

            # evaluate the trained model at multiple eps values
            with torch.no_grad():
                # if there is value in target_eps that is larger than epoch_end_eps
                # we evalute the model at epoch_end_eps for this target_eps value
                there_is_eps_geq_end_eps = False
                for this_eps in target_eps:
                    if this_eps >= epoch_end_eps:
                        there_is_eps_geq_end_eps = True
                        
                target_eps = [epoch_end_eps] + target_eps
                err = [0] * (eps_len+1)
                clean_err = [0] * (eps_len+1)
                for eps_idx, this_target_eps in enumerate(target_eps):
                    if eps_idx == 0:
                        if there_is_eps_geq_end_eps:
                            # compute err and clean_err under epoch_end_eps
                            err_temp, clean_err_temp,_,_ = Train(model, t, test_data, EpsilonScheduler("linear", 0, 0, epoch_end_eps, epoch_end_eps, 1), 
                                                max_eps, norm, logger, verbose, False, None, method, disable_multi_gpu=disable_multi_gpu, 
                                                after_crown_or_lbp_settings=after_crown_or_lbp_settings, target_eps = this_target_eps,**method_param)
                        else:
                            err_temp = 0
                            clean_err_temp = 0
                        err[eps_idx] = err_temp
                        clean_err[eps_idx] = clean_err_temp
                    elif this_target_eps < epoch_end_eps:
                        # compute err and clean_err under this_target_eps
                        err_temp, clean_err_temp,_,_ = Train(model, t, test_data, EpsilonScheduler("linear", 0, 0, epoch_end_eps, epoch_end_eps, 1), 
                                                max_eps, norm, logger, verbose, False, None, method, disable_multi_gpu=disable_multi_gpu, 
                                                after_crown_or_lbp_settings=after_crown_or_lbp_settings, target_eps = this_target_eps,**method_param)
                        err[eps_idx] = err_temp
                        clean_err[eps_idx] = clean_err_temp
                    else: # eps_idx>0 and this_target_eps >= epoch_end_eps
                        # this target_eps >= epoch_end_eps, since we take eps=min(target_eps, epoch_end_eps)
                        # the evaluation result will be the same as this target_eps = epoch_end_eps
                        err[eps_idx] = err[0]
                        clean_err[eps_idx] = clean_err[0]
                
                target_eps = target_eps[1:]
                err = err[1:]
                clean_err = clean_err[1:]

                for eps_idx, this_target_eps in enumerate(target_eps):
                    tensorboard_writer.add_scalar('Epoch Error/Test Robust Error (target eps %.6f)' % this_target_eps, 
                                        err[eps_idx], global_step=t)
                    tensorboard_writer.add_scalar('Epoch Error/Test Error (target eps %.6f)' % this_target_eps, 
                                        clean_err[eps_idx], global_step=t)




            logger.log('saving to', model_name)
            torch.save({
                    'state_dict' : model.module.state_dict() if multi_gpu else model.state_dict(), 
                    'epoch' : t,
                    }, model_name)

            # save the best model after we reached the schedule
            if t >= (schedule_start + schedule_length):
                for err_idx, this_err in enumerate(err):
                    if this_err <= best_err[err_idx]:
                        best_err[err_idx] = this_err
                        recorded_clean_err[err_idx] = clean_err[err_idx]
                        this_best_model_name = best_model_name + '_test_eps_%.5f' % target_eps[err_idx]
                        logger.log('Saving best model {} with error {}'.format(this_best_model_name, best_err[err_idx]))
                        torch.save({
                                'state_dict' : model.module.state_dict() if multi_gpu else model.state_dict(), 
                                'robust_err' : err,
                                'clean_err' : clean_err,
                                'epoch' : t,
                                }, this_best_model_name)

        logger.log('Total Time: {:.4f}'.format(timer))
        logger.log('Model {} best err {}, clean err {}'.format(model_id, best_err, recorded_clean_err))

if __name__ == "__main__":
    args = argparser()
    main(args)
