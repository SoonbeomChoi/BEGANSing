import os 
import torch 
import torch.nn as nn

def set_device(x, device, use_cpu=False):
    multi_gpu = False
    if len(device) > 1:
        multi_gpu = True 

    # When input is tensor 
    if isinstance(x, torch.Tensor): 
        if use_cpu:
            x = x.cpu()
        else:
            x = x.cuda(device[0])
     # When input is model
    elif isinstance(x, nn.Module): 
        if use_cpu:
            x.cpu()
        else:
            torch.cuda.set_device(device[0])
            if multi_gpu:
                x = nn.DataParallel(x, device_ids=device).cuda()
            else: 
                x.cuda(device[0])
    # When input is tuple 
    elif type(x) is tuple or type(x) is list:
        x = list(x)
        for i in range(len(x)):
            x[i] = set_device(x[i], device, use_cpu)
        x = tuple(x) 

    return x 

def from_parallel(state_dict):
    from_parallel = False
    for key, _ in state_dict.items():
        if key.find('module.') != -1:
            from_parallel = True
            break 

    return from_parallel

def unwrap_parallel(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value

    return new_state_dict

def load_weights(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
    if from_parallel: 
        state_dict = unwrap_parallel(state_dict)

    return state_dict

def save_checkpoint(checkpoint_path, model, optimizer=None, learning_rate=None, iteration=None, verbose=False):
    checkpoint = {'state_dict': model.state_dict()}
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if learning_rate is not None:
        checkpoint['learning_rate'] = learning_rate
    if iteration is not None:
        checkpoint['iteration'] = iteration
    
    torch.save(checkpoint, checkpoint_path)

    if verbose: 
        print("Saving checkpoint to %s" % (checkpoint_path))

def load_checkpoint(checkpoint_path, model, optimizer=None, verbose=False):
    assert os.path.isfile(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model'].state_dict()
    else:
        raise AssertionError("No model weight found in checkpoint, %s" % (checkpoint_path))

    if from_parallel: 
        state_dict = unwrap_parallel(state_dict)

    model.load_state_dict(state_dict)

    objects = [model]
    if 'optimizer' in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        objects.append(optimizer)
    if 'learning_rate' in checkpoint:
        learning_rate = checkpoint['learning_rate']
        objects.append(learning_rate)
    if 'iteration' in checkpoint:
        iteration = checkpoint['iteration']
        objects.append(iteration)

    if verbose:
        print("Loaded checkpoint from %s" % (checkpoint_path))

    if len(objects) == 1:
        objects = objects[0]

    return objects