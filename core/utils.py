import os
import pdb
import torch
import numpy as np
import torch.nn as nn
import random
import string
from pathlib import Path
from core.models import MLP, ResNet18

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def log_comet_metric(exp, name, val, step):
    exp.log_metric(name=name, value=val, step=step)

def get_random_string(length):
    # Random string with the combination of lower and upper case
    letters = string.ascii_letters
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def save_model(model, path):
    torch.save(model.cpu(), path)

def load_model(path):
    model = torch.load(path)
    return model

def save_task_model_by_policy(model, task, policy, exp_dir):
    # task = 0 is the initilization
    if task == 0 or policy == 'init':
        save_model(model, '{}/init.pth'.format(exp_dir))

    # the first task model is the same for all
    if task == 1:
        save_model(model, '{}/t_{}_seq.pth'.format(exp_dir, task))
        save_model(model, '{}/t_{}_lmc.pth'.format(exp_dir, task))
        save_model(model, '{}/t_{}_mtl.pth'.format(exp_dir, task))
    else:
        save_model(model, '{}/t_{}_{}.pth'.format(exp_dir, task, policy))


def load_task_model_by_policy(task, policy, exp_dir):
    if task == 0 or policy == 'init':
        return load_model('{}/init.pth'.format(exp_dir))
    return load_model('{}/t_{}_{}.pth'.format(exp_dir, task, policy))

def flatten_grads(m, numpy_output=False, bias=True, only_linear=False):
    total_grads = []
    for name, param in m.named_parameters():
        if only_linear:
            if (bias or not 'bias' in name) and 'linear' in name:
                total_grads.append(param.grad.detach().view(-1))
        else:
            if (bias or not 'bias' in name) and not 'bn' in name and not 'IC' in name:
                try:
                    total_grads.append(param.grad.detach().view(-1))
                except AttributeError:
                    pass
                    #print('no_grad', name)
    total_grads = torch.cat(total_grads)
    if numpy_output:
        return total_grads.cpu().detach().numpy()
    return total_grads


def compute_and_flatten_example_grads(m, criterion, data, target, task_id):
    _eg = []
    criterion2 = nn.CrossEntropyLoss(reduction='none').to(DEVICE)
    m.eval()
    m.zero_grad()
    pred = m(data, task_id)
    loss = criterion2(pred, target)
    for idx in range(len(data)):
        loss[idx].backward(retain_graph=True)
        _g = flatten_grads(m, numpy_output=True)
        _eg.append(torch.Tensor(_g))
        m.zero_grad()
    return torch.stack(_eg)


def flatten_example_grads(m, numpy_output=False):
    total_grads = []
    for param in m.parameters():
            total_grads.append(param.grad1.view(param.grad1.size()[0], -1))
    total_grads = torch.cat(total_grads, 1)
    if numpy_output:
        return total_grads.cpu().detach().numpy()
    return total_grads


def flatten_params(m, numpy_output=True):
    total_params = []
    for param in m.parameters():
            total_params.append(param.view(-1))
    total_params = torch.cat(total_params)
    if numpy_output:
        return total_params.cpu().detach().numpy()
    return total_params

def assign_weights(m, weights):
    state_dict = m.state_dict(keep_vars=True)
    index = 0
    with torch.no_grad():
        for param in state_dict.keys():
            if 'running_mean' in param or 'running_var' in param or 'num_batches_tracked' in param:
                continue
            # print(param, index)
            param_count = state_dict[param].numel()
            param_shape = state_dict[param].shape
            state_dict[param] =  nn.Parameter(torch.from_numpy(weights[index:index+param_count].reshape(param_shape)))
            index += param_count
    m.load_state_dict(state_dict)
    return m

def assign_grads(m, grads):
    state_dict = m.state_dict(keep_vars=True)
    index = 0
    for param in state_dict.keys():
        if 'running_mean' in param or 'running_var' in param or 'num_batches_tracked' in param:
                continue
        param_count = state_dict[param].numel()
        param_shape = state_dict[param].shape
        state_dict[param].grad =  grads[index:index+param_count].view(param_shape).clone()
        index += param_count
    m.load_state_dict(state_dict)
    return m

def accum_grads(m, grads):
    state_dict = m.state_dict(keep_vars=True)
    index = 0
    for param in state_dict.keys():
        if 'running_mean' in param or 'running_var' in param or 'num_batches_tracked' in param:
                continue
        param_count = state_dict[param].numel()
        param_shape = state_dict[param].shape
        state_dict[param].grad +=  grads[index:index+param_count].view(param_shape).clone()
        index += param_count
    m.load_state_dict(state_dict)
    return m


def get_norm_distance(m1, m2):
    m1 = flatten_params(m1, numpy_output=False)
    m2 = flatten_params(m2, numpy_output=False)
    return torch.norm(m1-m2, 2).item()


def get_cosine_similarity(m1, m2):
    m1 = flatten_params(m1)
    m2 = flatten_params(m2)
    cosine = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    return cosine(m1, m2)


def save_np_arrays(data, path):
    np.savez(file=path, **data)

def setup_experiment(experiment, config):
    Path(config['exp_dir']).mkdir(parents=True, exist_ok=True)
    if 'mixture' in config['dataset']:
        init_model = ResNet18(nclasses=83, config=config)
    else:
        init_model = ResNet18(config=config) if 'cifar' in config['dataset'] else MLP(config)
    save_model(init_model, '{}/init.pth'.format(config['exp_dir']))
    experiment.log_parameters(config)


class ContinualMeter:
    def __init__(self, name, n_tasks):
        self.name = name
        self.data = np.zeros((n_tasks, n_tasks))

    def update(self, current_task, target_task, metric):
        self.data[current_task-1][target_task-1] = round(metric, 3)

    def save(self, config):
        path = '{}/{}.csv'.format(config['exp_dir'], self.name)
        np.savetxt(path, self.data, delimiter=",")
