import torch as t
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from torchnet import meter
import numpy as np
from sklearn.metrics import roc_auc_score
import random

from data.common_dataloader import CommonDataloader

import models
from models.nafnet import NAFNet1
from config import opt

import os
from tqdm import tqdm
from itertools import chain
from PIL import Image
import sys
import time

# load old model weights into new model (only for layers with common name)
def load_model_weights(old_model,new_model):
    if opt.use_gpu:
        new_model.cuda()
    pretrained_dict = old_model.state_dict()
    substitute_dict = new_model.state_dict()
    pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in substitute_dict}
    substitute_dict.update(pretrained_dict)
    new_model.load_state_dict(substitute_dict)
    return new_model

def setup_seed(seed):
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    t.backends.cudnn.benchmark = False
    t.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


# hook for monitoring layer outputs
def get_features_hook(self, input, output):
    print("hook",output.data)

class CombineTrainer():
    def __init__(self, opt):
        self.description     = "train 1 net"
        self.train_data_root = opt.train_data_root
        self.model_name      = opt.model_name
        self.dataset         = opt.dataset

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.constant(m.weight, 1e-2)
            nn.init.xavier_normal(m.weight)
            nn.init.constant(m.bias,0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal(m.weight, mode="fan_out")
            # nn.init.constant(m.weight, 1e-3)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight, 2e-1)
            nn.init.constant(m.bias, 0)

    def print_grad(self, model):
        for name, parms in model.named_parameters():
            print('-->name:', name, '-->grad_required:', parms.requires_grad, \
                  ' -->grad_value:', parms.grad)

    def train(self):
        # setup_seed(opt.seed)
        # 1 model
        model = eval(self.model_name)(opt)
        loaded_model = eval(opt.loaded_model_name)(opt)
        # handle = model.head.register_forward_hook(get_features_hook)
        self.print_parameters(model)
        if self.model_name != 'F3Net':
            model.apply(self.weight_init)

        if opt.load_model and os.path.exists(opt.load_model_path):
            loaded_model.load(opt.load_model_path)
            model = load_model_weights(loaded_model, model)
        if opt.use_gpu:
            loaded_model.cuda()
            model.cuda()
        
        model.train()

        # 2 data
        train_data = CommonDataloader(self.train_data_root, self.dataset, noise=opt.train_noise, train=True)
        val_data   = CommonDataloader(self.train_data_root, self.dataset, noise=opt.test_noise, train=False)
        train_dataloader = DataLoader(
            train_data,
            opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers)
        val_dataloader = DataLoader(
            val_data,
            opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers)
        
        # 3 loss and optimizer
        criterion = t.nn.CrossEntropyLoss()
        mse_func = t.nn.MSELoss()
        learning_rate = opt.lr
        if opt.optimizer == 'adam':
            optimizer = t.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=opt.weight_decay
            )
        else:
            optimizer = t.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                weight_decay=opt.weight_decay,
                momentum=0.9
            )

        # train
        previous_loss = 1000
        for epoch in range(opt.max_epoch):
            start_time = time.perf_counter()
            current_loss = 0
            # self.val(model,val_dataloader)
            for ii, (lr,hr,label), in enumerate(train_dataloader):
                lr_input = Variable(lr, requires_grad=True)
                hr_target = Variable(hr)
                label_target = Variable(label)
                if opt.use_gpu:
                    lr_input = lr_input.cuda()
                    hr_target = hr_target.cuda()
                    label_target = label_target.cuda()
                optimizer.zero_grad()
                # score = model(lr_input)
                if self.model_name == 'NAFNet1Mid':
                    score, feature = nn.parallel.data_parallel(model, lr_input)
                    end_loss = criterion(score, label_target)
                    mid_loss = mse_func(feature, hr_target)
                    loss = end_loss + opt.mid_loss_weight * mid_loss
                else:
                    score = nn.parallel.data_parallel(model, lr_input)
                    loss = criterion(score, label_target)
                current_loss += loss.item()
                loss.backward()
                optimizer.step()
            end_time = time.perf_counter()
            current_loss = current_loss/(ii+1)
            print("[Epoch %03d] loss %f | "%(epoch,current_loss),end='')
            if opt.save_model:
                model.save(opt.save_model_path)
            # learning rate decay is set for SGD optimizer
            if opt.optimizer == 'sgd':
                if current_loss > previous_loss:
                    learning_rate = learning_rate*opt.lr_decay
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate
                previous_loss = current_loss
                print("lr %f | "%learning_rate, end='')
            self.val(model,val_dataloader)

            print('time %.2fs'%(end_time-start_time))
        # handle.remove()

    def val(self,my_model,dataloader):
        my_model.eval()
        confusion_matrix = meter.ConfusionMeter(2)
        score_all = []
        label_all = []
        with t.no_grad():
            for ii,(data, _, label) in enumerate(dataloader):
                val_input = Variable(data)
                val_label = Variable(label.type(t.LongTensor))
                if opt.use_gpu:
                    val_input = val_input.cuda()
                    val_label = val_label.cuda()
                if self.model_name == 'NAFNet1Mid':
                    score, _ = my_model(val_input)
                else:
                    score = my_model(val_input)
                confusion_matrix.add(score.data, label.long())
                score_all.extend(score[:,1].detach().cpu().numpy())
                label_all.extend(label)
        my_model.train()
        cm_value = confusion_matrix.value()
        accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) /\
                    (cm_value.sum())
        auc = roc_auc_score(label_all, score_all)
        print(f"val acc.{accuracy} | AUC {auc} | ", end='')
        return confusion_matrix, accuracy

    def print_parameters(self,model):
        pid = os.getpid()
        total_num = sum(i.numel() for i in model.parameters())
        trainable_num = sum(i.numel() for i in model.parameters() if i.requires_grad)

        print("=========================================")
        print("PID:",pid)
    
        print("\nNum of parameters:%i"%(total_num))
        print("Num of trainable parameters:%i"%(trainable_num))

        print("\nLoad model:",opt.load_model)
        print("Load model path:",opt.load_model_path)
        print("Loaded model name:",opt.loaded_model_name)
        print("Save model:",opt.save_model)
        print("Save model path:",opt.save_model_path)
        print("model_name:",opt.model_name)
        print("dataset:",opt.dataset)
        print("train noise:",opt.train_noise)
        print("test noise:",opt.test_noise)
        print("seed:",opt.seed)
        print("input noise scale:",opt.noise_scale)
        print("optimizer:",opt.optimizer)
        print("learning rate:",opt.lr)

        print("\nHyper-parameter:")
        print("batch_size:",opt.batch_size)
        print("n_nablock:",opt.n_nablock)
        print("in_channels:",opt.in_channels)
        print("hid_dim:",opt.hid_dim)
        print("n_heads:",opt.n_heads)
        print("n_layers:",opt.n_layers)
        print("growth_rate:",opt.growth_rate)
        print("dropout:",opt.dropout)

        print("\nConfiguration:\nGPU-ID:%s"%opt.gpu_id)

        print("=========================================")

