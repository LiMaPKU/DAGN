from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam
from torch.nn.parallel import DataParallel  # , DistributedDataParallel
from torch.autograd import Variable
from torch.nn.functional import l1_loss

from models.select_network import define_G, define_D, define_P
from models.model_base import ModelBase
from models.loss_ssim import SSIMLoss

from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip

import numpy as np
from IPython import embed
class ModelDFGN(ModelBase):
    """Train with pixel loss"""
    def __init__(self, opt):
        super(ModelDFGN, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.netG = define_G(opt).to(self.device)
        self.netG = DataParallel(self.netG)
        # mali

        self.netP = define_P(opt)
        if self.netP is not None:
            self.netP = DataParallel(self.netP.to(self.device))


    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.opt_train = self.opt['train']    # training option
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN

        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log
    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)

    # ----------------------------------------
    # save model
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_lossfn_type == 'ssim':
            self.G_lossfn = SSIMLoss().to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']

        # mali define perceptual loss
        if self.netP is not None:
            self.P_lossfn = nn.L1Loss().to(self.device)
        else:
            self.P_lossfn = None
    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0)
    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        self.opt_train['G_scheduler_milestones'],
                                                        self.opt_train['G_scheduler_gamma']
                                                        ))
    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data):
        self.H = data['H'].to(self.device)
        self.L = data['L'].to(self.device)
        if self.opt['se_predict'] == 'qf':
            self.qf_gt = data['qf'].to(self.device).squeeze()
        elif self.opt['se_predict'] == 'l1':
            self.qf_gt = Variable(torch.sum(torch.abs(self.H - self.L), dim=(1,2,3)).float())
            self.qf_gt = self.qf_gt/(self.opt['datasets']['train']['H_size']**2)
        else:
            raise NotImplementedError('se predict type: {} is not found.'.format(self.opt['se_predict']))
    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):

        self.E, _, _, _, _, _ = self.netG(self.L, self.H)

        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']
        G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
        self.QF_lossfn_weight = self.opt_train['QF_lossfn_weight']

        # mali: perceptual loss
        P_loss = torch.zeros([1,]).to(self.device)
        if self.P_lossfn is not None:
            self.P_lossfn_weight = self.opt_train['P_lossfn_weight']
            self.P_lossfn_indexs = self.opt_train['P_lossfn_indexs']
            out_p_e = self.netP(self.E)
            out_p_h = self.netP(self.H)
            for index in self.P_lossfn_indexs:
                P_loss += self.P_lossfn_weight * self.P_lossfn(out_p_e[index], out_p_h[index])
        else:
            pass

        loss = G_loss + P_loss

        self.G_optimizer.zero_grad()
        loss.backward()
        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)
        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict['G_loss'] = G_loss.item()

        if self.P_lossfn is not None:
            self.log_dict['P_loss'] = P_loss.item()
    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            #            self.E, self.QF = self.netG(self.L, torch.tensor([0.1]).reshape(1,1))
            self.E, _,_, _, _, _ = self.netG(self.L, self.H)

        self.netG.train()

    # ----------------------------------------
    # test / inference x8
    # ----------------------------------------
    def testx8(self):
        self.netG.eval()
        with torch.no_grad():
            self.E = test_mode(self.netG, self.L, mode=3, sf=self.opt['scale'], modulo=1)
        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        #        out_dict['QF'] = self.QF.mean(-1).mean(-1).detach()[0].float().cpu() # qf table

        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_O=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_O:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
