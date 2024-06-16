# -*- coding: utf-8 -*-
from __future__ import print_function
import random
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import MNIST

from IPython.display import Image
from utility.dip_utility  import *
from utility.ddpm_utility import *

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ddpm_store_path = "ddpm_mnist.pt"

def train_dip():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark =True
    sigma = 25
    sigma_ = sigma/255.

    transform_dip = Compose([
        transforms.Resize((32, 32))]
    )
    dataset = MNIST("./datasets", download=True, train=True,transform=transform_dip)

    img_pil = crop_image(dataset[0][0], d=1)
    img_np = pil_to_np(img_pil)
    img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
    input_depth = 1
    pad = 'reflection'
    OPT_OVER = 'net' # 'net,input
    INPUT = 'noise' # 'meshgrid'
    pad = 'reflection'

    reg_noise_std = 1./30. # set to 1./20. for sigma=50
    LR = 0.01

    OPTIMIZER='adam' # 'LBFGS'
    show_every = 100
    exp_weight=0.99

    num_iter = 200
    net = skip(
                input_depth, 1,
                num_channels_down = [8, 16, 32],
                num_channels_up   = [8, 16, 32],
                num_channels_skip = [0, 4, 4],
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

    net = net.type(torch.cuda.FloatTensor)

    
    # Loss
    mse = torch.nn.MSELoss().type(torch.cuda.FloatTensor)

    img_noisy_torch = np_to_torch(img_noisy_np).type(torch.cuda.FloatTensor)


    # setting global value
    global i, out_avg, psrn_noisy_last, last_net, net_input
    net_input = get_noise(input_depth, 'noise', (img_pil.size[1], img_pil.size[0])).type(torch.cuda.FloatTensor).detach()
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    out_avg = None
    last_net = None
    psrn_noisy_last = 0
    i = 0
    def closure():
        reg_noise_std = 1./30. # set to 1./20. for sigma=50
        global i, out_avg, psrn_noisy_last, last_net, net_input
        PLOT = False
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)

        # Smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        total_loss = mse(out, img_noisy_torch)
        total_loss.backward()

        psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0])
        psrn_gt    = compare_psnr(img_np, out.detach().cpu().numpy()[0])
        psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0])

        # Note that we do not have GT for the "snail" example
        # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
        print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
        if  PLOT and i % show_every == 0:
            out_np = torch_to_np(out)
            plot_image_grid([np.clip(out_np, 0, 1),
                            np.clip(torch_to_np(out_avg), 0, 1)], factor=figsize, nrow=1)



        # Backtracking
        if i % show_every:
            if psrn_noisy - psrn_noisy_last < -5:
                print('Falling back to previous checkpoint.')

                for new_param, net_param in zip(last_net, net.parameters()):
                    net_param.data.copy_(new_param.cuda())

                return total_loss*0
            else:
                last_net = [x.detach().cpu() for x in net.parameters()]
                psrn_noisy_last = psrn_noisy

        i += 1

        return total_loss


    # start training  dip
    p = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, p, closure, LR, num_iter)

    # show picture
    out_np = torch_to_np(net(net_input))
    q = plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13)
    return net

def ddpm():
    
    # load data
    transform = Compose([
        ToTensor(),
        Lambda(lambda x: (x - 0.5) * 2)]
    )
    dataset = MNIST("./datasets", download=True, train=True, transform=transform)
    loader = DataLoader(dataset, 128, shuffle=True)
    
    # Defining model
    
    ddpm = MyDDPM(MyUNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)

    # Training
    n_epochs = 20
    lr = 0.001
    training_loop(ddpm, loader, n_epochs, optim=Adam(ddpm.parameters(), lr), device=device, store_path=ddpm_store_path)

def generate(net):
    
    best_model = MyDDPM(MyUNet(), n_steps=n_steps, device=device)
    best_model.load_state_dict(torch.load(ddpm_store_path, map_location=device))
    best_model.eval()

    print("Generating new images")
    generated = generate_new_images(
            best_model,
            n_samples=1,
            device=device,
            gif_name=f"final.gif",
            dip_model=net,times=2,steps=500
        )
    # show_images(generated, "Final result")
    from PIL import Image
    from IPython.display import display
    x = Image.open('final.gif')
    display(x)
    print('if you want to see the gif, you need to open it by yourself!, the gif file is final.gif')
    
if __name__ == "__main__":
    print("Start training DIP")
    net = train_dip()
    print("Start training DDPM")
    ddpm()
    print("Generating new images")
    generate(net)