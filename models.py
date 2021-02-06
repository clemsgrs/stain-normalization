import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.autograd import Variable
import itertools

from utils import ImagePool, tensor2im
import networks as networks


class ModelBackbone():
    
    def __init__(self, p):

        self.p = p
        self.gpu_ids = p.gpu_ids
        self.isTrain = p.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(p.checkpoints_dir, p.name)

    def name(self):
        return 'BaseModel'

    def set_input(self, input):
        self.input = input

    # helper saving function that can be used by subclasses
    def save_model(self, model, model_label, epoch_label, gpu_ids):
        save_filename = f'{epoch_label}_net_{model_label}.pth'
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(model.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            model.cuda(gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_model(self, model, model_label, epoch_label):
        save_filename = f'{epoch_label}_net_{model_label}.pth'
        save_path = os.path.join(self.save_dir, save_filename)
        model.load_state_dict(torch.load(save_path))

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        if (lr <= 0):
            print(f'Learning rate = {lr:.7f}')
            print('EXITING TRAINING BECAUSE LR IS <0')
            sys.exit()
        print(f'Learning rate = {lr:.7f}')


class CycleGAN(ModelBackbone):
    
    def __init__(self, p):

        super(CycleGAN, self).__init__(p)
        nb = p.batchSize
        size = p.cropSize
        
        # load/define models
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.netG_A = networks.define_G(p.input_nc, p.output_nc, p.ngf, p.which_model_netG, p.norm, not p.no_dropout, p.init_type, self.gpu_ids)
        self.netG_B = networks.define_G(p.output_nc, p.input_nc, p.ngf, p.which_model_netG, p.norm, not p.no_dropout, p.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = p.no_lsgan
            self.netD_A = networks.define_D(p.output_nc, p.ndf, p.which_model_netD, p.n_layers_D, p.norm, use_sigmoid, p.init_type, self.gpu_ids)
            self.netD_B = networks.define_D(p.input_nc, p.ndf, p.which_model_netD, p.n_layers_D, p.norm, use_sigmoid, p.init_type, self.gpu_ids)
        
        if not self.isTrain or p.continue_train:
            which_epoch = p.which_epoch
            self.load_model(self.netG_A, 'G_A', which_epoch)
            self.load_model(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_model(self.netD_A, 'D_A', which_epoch)
                self.load_model(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = p.lr
            self.fake_A_pool = ImagePool(p.pool_size)
            self.fake_B_pool = ImagePool(p.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not p.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # gradient loss as per aicha
            if p.aicha_loss:
                self.criterionAicha = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=p.lr, betas=(p.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=p.lr, betas=(p.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=p.lr, betas=(p.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, p))

        # print('---------- Networks initialized -------------')
        # networks.print_network(self.netG_A)
        # networks.print_network(self.netG_B)
        # if self.isTrain:
        #     networks.print_network(self.netD_A)
        #     networks.print_network(self.netD_B)
        # print('-----------------------------------------------')

    def name(self):
        return 'CycleGAN'

    def set_input(self, inp):
        AtoB = self.p.which_direction == 'AtoB'
        input_A = inp['A' if AtoB else 'B']
        input_B = inp['B' if AtoB else 'A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0])
            input_B = input_B.cuda(self.gpu_ids[0])
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = inp['A_path' if AtoB else 'B_path']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        with torch.no_grad():
            real_A = Variable(self.input_A)
            fake_B = self.netG_A(real_A)
            self.rec_A = self.netG_B(fake_B).data
            self.fake_B = fake_B.data

            real_B = Variable(self.input_B)
            fake_A = self.netG_B(real_B)
            self.rec_B = self.netG_A(fake_A).data
            self.fake_A = fake_A.data

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        self.loss_D_A = loss_D_A.item()

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        self.loss_D_B = loss_D_B.item()

    def backward_G(self):
        lambda_idt = self.p.identity
        lambda_A = self.p.lambda_A
        lambda_B = self.p.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            idt_A = self.netG_A(self.real_B)
            loss_idt_A = self.criterionIdt(idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            idt_B = self.netG_B(self.real_A)
            loss_idt_B = self.criterionIdt(idt_B, self.real_A) * lambda_A * lambda_idt
            self.idt_A = idt_A.data
            self.idt_B = idt_B.data
            self.loss_idt_A = loss_idt_A.item()
            self.loss_idt_B = loss_idt_B.item()
        
        else:
            loss_idt_A = 0
            loss_idt_B = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        fake_B = self.netG_A(self.real_A)
        pred_fake = self.netD_A(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, True)

        # GAN loss D_B(G_B(B))
        fake_A = self.netG_B(self.real_B)
        pred_fake = self.netD_B(fake_A)
        loss_G_B = self.criterionGAN(pred_fake, True)

        # Forward cycle loss
        rec_A = self.netG_B(fake_B)
        loss_cycle_A = self.criterionCycle(rec_A, self.real_A) * lambda_A
        # print("loss_cycle_A  ",loss_cycle_A.grad)

        # Backward cycle loss
        rec_B = self.netG_A(fake_A)
        loss_cycle_B = self.criterionCycle(rec_B, self.real_B) * lambda_B

        # aicha forward
        if self.p.aicha_loss:
            print("I AM AICHCA LOSS")
            # w1 = Variable(torch.Tensor([1.0, 2.0, 3.0]), requires_grad=True)
            #
            # print(w1.gra)
            #
            # sys.exit()
            #
            # realA_img = tensor2im(self.real_A.data)
            # recA_img = tensor2im(rec_A.data)
            # print(realA_img.shape)
            # print("---------")
            # gradient_input = self.get_gradient(recA_img)
            #
            # # print(gradient_input)
            #
            # wx = Variable(torch.from_numpy(np.multiply(gradient_input, realA_img)), requires_grad=True)
            # wg = Variable(torch.from_numpy(np.multiply(gradient_input, recA_img)), requires_grad=True)
            # loss_Aicha_A = self.criterionAicha(wx, wg) * self.p.lambda_Aicha

            # aicha backwards

            realB_img = tensor2im(self.real_B.data)
            recB_img = tensor2im(rec_B.data)
            plt.imshow(realB_img)
            plt.imshow(recB_img)
            plt.show()

            sys.exit()

            gradient_input = np.gradient(realB_img)
            # print("gradient_input", gradient_input.size)
            gradient_input = torch.from_numpy(gradient_input)
            sys.exit()
            grad_times_real = torch.mul(gradient_input, self.real_B)

            # wx = Variable(torch.from_numpy(), requires_grad=True)
            wx = grad_times_real
            print('grad_times_real', grad_times_real)
            sys.exit()
            wg = Variable(torch.from_numpy(np.multiply(gradient_input, recB_img)), requires_grad=True)
            loss_Aicha_B = self.criterionAicha(wx, wg) * self.p.lambda_Aicha

            # aicha combined loss

            loss_G = loss_G_A + loss_G_B + loss_Aicha_A + loss_Aicha_B + loss_idt_A + loss_idt_B
        else:
            # combined loss
            loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B

        loss_G.backward()

        self.fake_B = fake_B.data
        self.fake_A = fake_A.data
        self.rec_A = rec_A.data
        self.rec_B = rec_B.data

        self.loss_G_A = loss_G_A.item()
        self.loss_G_B = loss_G_B.item()

        if self.p.aicha_loss:
            # AichaLoss
            self.loss_Aicha_A = loss_Aicha_A.item()
            self.loss_Aicha_B = loss_Aicha_B.item()
        else:
            # cycleLoss
            self.loss_cycle_A = loss_cycle_A.item()
            self.loss_cycle_B = loss_cycle_B.item()

    # def get_gradient(self, img):
    #     # print("get_gradient ",img)
    #     # return np.gradient(np.array(img))
    #     return np.gradient(img)

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A), ('Cyc_A', self.loss_cycle_A),
                                  ('D_B', self.loss_D_B), ('G_B', self.loss_G_B), ('Cyc_B', self.loss_cycle_B)])
        # aicha
        # ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A), ('Aicha_A', self.loss_cycle_A),
        #                           ('D_B', self.loss_D_B), ('G_B', self.loss_G_B), ('Aicha_B', self.loss_cycle_B)])

        if self.p.identity > 0.0:
            ret_errors['idt_A'] = self.loss_idt_A
            ret_errors['idt_B'] = self.loss_idt_B
        
        return ret_errors

    def get_current_visuals(self):
        real_A = tensor2im(self.input_A)
        fake_B = tensor2im(self.fake_B)
        rec_A = tensor2im(self.rec_A)
        real_B = tensor2im(self.input_B)
        fake_A = tensor2im(self.fake_A)
        rec_B = tensor2im(self.rec_B)
        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                   ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])
        if self.isTrain and self.p.identity > 0.0:
            ret_visuals['idt_A'] = tensor2im(self.idt_A)
            ret_visuals['idt_B'] = tensor2im(self.idt_B)
        
        return ret_visuals

    def save(self, label):
        self.save_model(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_model(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_model(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_model(self.netD_B, 'D_B', label, self.gpu_ids)


class TestModel(ModelBackbone):

    def __init__(self, p):

        super(TestModel, self).__init__(p)
        assert(not p.isTrain)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        which_epoch = p.which_epoch
        self.load_model(self.netG, 'G', which_epoch)
        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')
    
    def name(self):
        return 'TestModel'

    def set_input(self, inp):
        # we need to use single_dataset mode
        input_A = inp['A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0])
        self.input_A = input_A
        self.image_paths = inp['A_path']

    def test(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_A = tensor2im(self.real_A.data)
        fake_B = tensor2im(self.fake_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])


def create_model(p):
    
    model = None
    print(p.model)
    
    if p.model == 'cycle_gan':
        assert(p.dataset_mode == 'unaligned')
        model = CycleGAN(p)
    elif p.model == 'test':
        assert(p.dataset_mode == 'single')
        model = TestModel(p)
    else:
        raise ValueError(f'Model {p.model} not recognized')

    print(f'model {model.name()} was created')
    
    return model