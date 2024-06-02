import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import torch.utils.data as data
import numpy as np
import random
import time
import copy
from tqdm import tqdm

try: import torch_xla.core.xla_model as xm
except: pass

try: from submodlib.functions.facilityLocation import FacilityLocationFunction
except: pass

######################
# Friendly Noise Utils
######################

def generate_friendly_noise(
        model,
        trainloader,
        device,
        device_type,
        friendly_epochs,
        mu,
        friendly_lr,
        friendly_momentum=0.9,
        nesterov=True,
        friendly_steps=None,
        clamp_min=-32/255,
        clamp_max=32/255,
        return_preds=False,
        loss_fn='KL',
        model_train = False,
        img_dim = 32
        ):


    if loss_fn == 'MSE':
        criterion = torch.nn.MSELoss()
    elif loss_fn == 'KL':
        criterion = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    else:
        raise ValueError("No such loss fn")

    # Deep copy model and set all parameters to no grad for tpu
    model = copy.deepcopy(model)
    # Turn off grad for TPU stuff 
    for param in model.parameters():
        param.grad = None
        param.requires_grad = False
    
    # Set model to train mode for TPU
    if model_train:
        model.train()

    # Get all but first transform
    transform = trainloader.dataset.transform
    trainloader.dataset.transform = transforms.Compose([transforms.ToTensor()])

    dataset_size = len(trainloader.dataset)
    friendly_noise = torch.zeros((dataset_size, 3, img_dim, img_dim))
    if return_preds:
        preds = torch.zeros((dataset_size, 10))

    if xm.is_master_ordinal(): 
        pbar = tqdm(total=len(trainloader),desc='Friendly Noise Gen')

    for batch_idx, (inputs, target, idx, p) in enumerate(trainloader):

        init = (torch.rand(*(inputs.shape)) - 0.5) * 2 * 8/255
        eps = torch.autograd.Variable(init.to(device), requires_grad=True)

        optimizer = torch.optim.SGD([eps], lr=friendly_lr, momentum=friendly_momentum, nesterov=nesterov)

        if friendly_steps is None:
            friendly_steps = [friendly_epochs // 2, friendly_epochs // 4 * 3]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, friendly_steps)

        images_normalized = torch.stack([transform(x) for x in inputs], dim=0).to(device)

        with torch.no_grad():
            output_original = model(images_normalized)
        if loss_fn == 'KL':
            output_original = F.log_softmax(output_original, dim=1).detach()
        else:
            output_original = output_original.detach()

        for epoch in range(friendly_epochs):
            eps_clamp = torch.clamp(eps, clamp_min, clamp_max)
            perturbed = torch.clamp(inputs + eps_clamp, 0, 1)
            perturbed_normalized = torch.stack(
                [transform(p) for p in perturbed], dim=0)
            output_perturb = model(perturbed_normalized)
            if loss_fn == 'KL':
                output_perturb = F.log_softmax(output_perturb, dim=1)

            emp_risk, constraint = friendly_loss(output_perturb, output_original, eps_clamp, criterion)
            loss = emp_risk - mu * constraint

            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            if device_type =='xla': xm.mark_step()

        if device_type =='xla': xm.mark_step()
            
        friendly_noise[idx] = eps.cpu().detach()
        if return_preds:
           preds[idx] = output_original.cpu()

        if xm.is_master_ordinal():
            pbar.update(1)
        
    friendly_noise = torch.clamp(friendly_noise, clamp_min, clamp_max)

    trainloader.dataset.transform = transform

    if return_preds:
        return friendly_noise, preds
    else:
        return friendly_noise


def friendly_loss(output, target, eps, criterion):
    emp_risk = criterion(output, target)
    constraint = torch.mean(torch.square(eps))
    return emp_risk, constraint

#################
# Epic Utils
#################
    
def run_epic(args, target_net, base_train_poisoned_loader, epoch, device, times_selected):

    N = len(base_train_poisoned_loader.dataset)
    B = int(args.epic_subset_size * N)

    if args.verbose: print(f'Identifying small clusters at epoch {epoch}...B={B}, N={N}')

    subset = get_subset(args, target_net, base_train_poisoned_loader, B, epoch, N, base_train_poisoned_loader.dataset.indices, device, args.device_type)
    keep = np.where(times_selected[subset] == epoch)[0]
    subset = subset[keep]
    if len(subset) == 0:
        print(f"Epoch {epoch + 1}, 0 Subset, Using Random Subset!!!")
        subset = get_random_subset(B, N)

    train_data_poisoned_subset = Subset_Dataset(base_train_poisoned_loader, subset)

    return torch.utils.data.DataLoader(train_data_poisoned_subset, batch_size=args.batch_size, shuffle=True, num_workers=4)

# EPIC Dataset Subset Loader
class Subset_Dataset(data.Dataset):
    def __init__(
        self, loader, subset_indices,transform=None
    ):
        self.input_list = []
        self.label_list = []
        self.p_list = []
        self.index_list = []

        self.transform = transform
        
        for idx, (input, target, index, p) in enumerate(loader):
            idxs = np.where(np.isin(index, subset_indices))[0]
            self.input_list.extend([input[i] for i in idxs])
            self.label_list.extend([target[i] for i in idxs])
            self.p_list.extend([p[i] for i in idxs])
            self.index_list.extend([index[i] for i in idxs])

    def __getitem__(self, index):
        if self.transform is not None:
            input = self.transform(self.input_list[index])
        else:
            input = self.input_list[index]
        return input, self.label_list[index], self.index_list[index], self.p_list[index]
    
    def __len__(self):
        return len(self.input_list)

def get_subset(args, model, trainloader, num_sampled, epoch, N,indices,device,device_type):

    num_classes = model.linear.weight.shape[0]

    grad_preds = []
    labels = []
    conf_all = np.zeros(N)
    conf_true = np.zeros(N)

    with torch.no_grad():
        for ite, (inputs, targets, index, p) in enumerate(trainloader):
            model.eval()
            targets = targets.long()

            inputs = inputs.to(device)

            embed = model(inputs, penu=True)
            confs = torch.softmax(torch.matmul(embed, model.linear.weight.T), dim=1).cpu().detach()
            conf_all = np.amax(confs.numpy(), axis=1)
            conf_true = confs[range(len(targets)), targets].numpy()
            embed = embed.cpu().detach()
            g0 = confs - torch.eye(num_classes)[targets.long()]
            grad_preds.append(g0.cpu().detach().numpy())
        
            targets = targets.numpy()
            labels.append(targets)

            if device_type == 'xla': xm.mark_step()

        if device_type == 'xla': xm.mark_step()
        
        labels = np.concatenate(labels)
        subset, subset_weights, _, _, cluster_ = get_coreset(np.concatenate(grad_preds), labels, len(labels), num_sampled, num_classes, equal_num=args.epic_equal_num, optimizer=args.epic_greedy, metric=args.epic_metric)

    cluster = -np.ones(N, dtype=int)
    cluster = cluster_

    keep_indices = np.where(subset_weights > args.epic_cluster_thresh)
    if epoch >= args.epic_drop_after:
        keep_indices = np.where(np.isin(cluster, keep_indices))[0]
        subset = keep_indices
    else:
        subset = np.arange(N)

    return subset


def faciliy_location_order(c, X, y, metric, num_per_class, weights=None, optimizer="LazyGreedy"):
    class_indices = np.where(y == c)[0]
    X = X[class_indices]
    N = X.shape[0]

    start = time.time()
    obj = FacilityLocationFunction(n=len(X), data=X, metric=metric, mode='dense')
    S_time = time.time() - start

    start = time.time()
    greedyList = obj.maximize(
        budget=num_per_class,
        optimizer=optimizer,
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        verbose=False,
    )
    order = list(map(lambda x: x[0], greedyList))
    sz = list(map(lambda x: x[1], greedyList))
    greedy_time = time.time() - start

    S = obj.sijs
    order = np.asarray(order, dtype=np.int64)
    sz = np.zeros(num_per_class, dtype=np.float64)
    cluster = -np.ones(N)

    for i in range(N):
        if np.max(S[i, order]) <= 0:
            continue
        cluster[i] = np.argmax(S[i, order])
        if weights is None:
            sz[np.argmax(S[i, order])] += 1
        else:
            sz[np.argmax(S[i, order])] += weights[i]
    sz[np.where(sz==0)] = 1

    cluster[cluster>=0] += c * num_per_class

    return class_indices[order], sz, greedy_time, S_time, cluster


def get_orders_and_weights(B, X, metric, y=None, weights=None, equal_num=False, num_classes=10, optimizer="LazyGreedy"):
    '''
    Ags
    - X: np.array, shape [N, d]
    - B: int, number of points to select
    - metric: str, one of ['cosine', 'euclidean'], for similarity
    - y: np.array, shape [N], integer class labels for C classes
      - if given, chooses B / C points per class, B must be divisible by C
    - outdir: str, path to output directory, must already exist

    Returns
    - order_mg/_sz: np.array, shape [B], type int64
      - *_mg: order points by their marginal gain in FL objective (largest gain first)
      - *_sz: order points by their cluster size (largest size first)
    - weights_mg/_sz: np.array, shape [B], type float32, sums to 1
    '''
    N = X.shape[0]
    if y is None:
        y = np.zeros(N, dtype=np.int32)  # assign every point to the same class
    if num_classes is not None:
        classes = np.arange(num_classes)
    else:
        classes = np.unique(y)
    C = len(classes)  # number of classes

    if equal_num:
        class_nums = [sum(y == c) for c in classes]
        num_per_class = int(np.ceil(B / C)) * np.ones(len(classes), dtype=np.int32)
        minority = class_nums < np.ceil(B / C)
        if sum(minority) > 0:
            extra = sum([max(0, np.ceil(B / C) - class_nums[c]) for c in classes])
            for c in classes[~minority]:
                num_per_class[c] += int(np.ceil(extra / sum(minority)))
    else:
        num_per_class = np.int32(np.ceil(np.divide([sum(y == i) for i in classes], N) * B))
        total = np.sum(num_per_class)
        diff = total - B
        chosen = set()
        for i in range(diff):
            j = np.random.randint(C)
            while j in chosen or num_per_class[j] <= 0:
                j = np.random.randint(C)
            num_per_class[j] -= 1
            chosen.add(j)

    order_mg_all, cluster_sizes_all, greedy_times, similarity_times, cluster_all = zip(*map(
        lambda c: faciliy_location_order(c, X, y, metric, num_per_class[c], weights, optimizer=optimizer), classes))

    order_mg = np.concatenate(order_mg_all).astype(np.int32)
    weights_mg = np.concatenate(cluster_sizes_all).astype(np.float32)
    class_indices = [np.where(y == c)[0] for c in classes]
    class_indices = np.concatenate(class_indices).astype(np.int32)
    class_indices = np.argsort(class_indices)
    cluster_mg = np.concatenate(cluster_all).astype(np.int32)[class_indices]
    assert len(order_mg) == len(weights_mg)

    ordering_time = np.max(greedy_times)
    similarity_time = np.max(similarity_times)

    order_sz = []
    weights_sz = []
    vals = order_mg, weights_mg, order_sz, weights_sz, ordering_time, similarity_time, cluster_mg
    return vals

def get_coreset(gradient_est, 
                labels, 
                N, 
                B, 
                num_classes, 
                equal_num=True,
                optimizer="LazyGreedy",
                metric='euclidean'):
    '''
    Arguments:
        gradient_est: Gradient estimate
            numpy array - (N,p) 
        labels: labels of corresponding grad ests
            numpy array - (N,)
        B: subset size to select
            int
        num_classes:
            int
        normalize_weights: Whether to normalize coreset weights based on N and B
            bool
        gamma_coreset:
            float
        smtk:
            bool
        st_grd:
            bool

    Returns 
    (1) coreset indices (2) coreset weights (3) ordering time (4) similarity time
    '''
    try:
        subset, subset_weights, _, _, ordering_time, similarity_time, cluster = get_orders_and_weights(
            B, 
            gradient_est, 
            metric, 
            y=labels, 
            equal_num=equal_num, 
            num_classes=num_classes,
            optimizer=optimizer)
    except ValueError as e:
        print(e)
        print(f"WARNING: ValueError from coreset selection, choosing random subset for this epoch")
        subset, subset_weights = get_random_subset(B, N)
        ordering_time = 0
        similarity_time = 0

    if len(subset) != B:
        print(f"!!WARNING!! Selected subset of size {len(subset)} instead of {B}")
    # print(f'FL time: {ordering_time:.3f}, Sim time: {similarity_time:.3f}')

    return subset, subset_weights, ordering_time, similarity_time, cluster


def get_random_subset(B, N):
    # print(f'Selecting {B} element from the random subset of size: {N}')
    order = np.arange(0, N)
    np.random.shuffle(order)
    subset = order[:B]

    return subset