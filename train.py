import torch
import torchvision
from torch import nn
from torchvision import models
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
from PIL import Image
from dataset.dataset import NYUUWDataset
from tqdm import tqdm
import random
from torchvision import models
import numpy as np
from models.networks import Classifier, UNetEncoder, UNetDecoder
import click
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class ColorLoss(torch.nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, output, target):
        # Get per-channel differences
        delta_r = output[:, 0, :, :] - target[:, 0, :, :]
        delta_g = output[:, 1, :, :] - target[:, 1, :, :]
        delta_b = output[:, 2, :, :] - target[:, 2, :, :]

        # Get mean level of red
        mean_red = (target[:, 0, :, :] + output[:, 0, :, :]) / 2

        # Compute color loss
        color_loss = torch.sqrt(
            ((512 + mean_red) / 256) * torch.pow(delta_r, 2) +
            4 * torch.pow(delta_g, 2) +
            ((767 - mean_red) / 256) * torch.pow(delta_b, 2)
        )

        return torch.mean(color_loss)


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, x, y):
        x_grad = self.get_gradients(x)
        y_grad = self.get_gradients(y)
        return F.l1_loss(x_grad, y_grad)

    def get_gradients(self, img):
        # Calculate gradients of img
        img = img.float()
        gx = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
        gy = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
        gradients = torch.mean(gx) + torch.mean(gy)
        return gradients


def to_img(x):
    """
        Convert the tanh (-1 to 1) ranged tensor to image (0 to 1) tensor
    """

    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 256, 256)

    return x


def set_requires_grad(nets, requires_grad=False):
    """
        Make parameters of the given network not trainable
    """

    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

    return requires_grad


def compute_val_metrics(fE, fI, fN, dataloader, no_adv_loss, domain_drive):
    """
        Compute SSIM, PSNR scores for the validation set
    """

    fE.eval()
    fI.eval()
    fN.eval()

    mse_scores = []
    ssim_scores = []
    psnr_scores = []
    corr = 0

    criterion_MSE = nn.MSELoss().cuda()

    for idx, data in tqdm(enumerate(dataloader)):
        uw_img, cl_img, water_type, _, _ = data
        uw_img = Variable(uw_img).cuda()
        cl_img = Variable(cl_img, requires_grad=False).cuda()

        fE_out, enc_outs = fE(uw_img)
        fN_out, fN_out_c = fN(fE_out)
        fN_out = F.softmax(fN_out, dim=1)
        fI_out = to_img(fI(fE_out, enc_outs, fN_out_c, domain_drive))

        if int(fN_out.max(1)[1].item()) == int(water_type.item()):
            corr += 1

        mse_scores.append(criterion_MSE(fI_out, cl_img).item())

        fI_out = (fI_out * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
        cl_img = (cl_img * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)

        ssim_scores.append(ssim(fI_out, cl_img, multichannel=True))
        psnr_scores.append(psnr(cl_img, fI_out))

    fE.train()
    fI.train()
    if not no_adv_loss:
        fN.train()

    return sum(ssim_scores) / len(dataloader), sum(psnr_scores) / len(dataloader), sum(mse_scores) / len(
        dataloader), corr / len(dataloader)


def backward_I_loss(fI, fE_out, enc_outs, fN_c, cl_img, criterion_MSE, optimizer_fI, lambda_I_loss, retain_graph,
                    domain_drive, Gradient, Color):
    """
        Backpropagate the reconstruction loss
    """

    fI_out = to_img(fI(fE_out, enc_outs, fN_c, domain_drive))
    I_loss = criterion_MSE(fI_out, cl_img) * 60 + 2 * Gradient(fI_out, cl_img) + 2 * Color(fI_out, cl_img)
    optimizer_fI.zero_grad()
    I_loss.backward(retain_graph=retain_graph)
    optimizer_fI.step()

    return fI_out, I_loss


def backward_N_loss(fN, fE_out, actual_target, criterion_CE, optimizer_fN, lambda_N_loss):
    """
        Backpropagate the nuisance loss
    """

    fN_out, _ = fN(fE_out.detach())
    N_loss = criterion_CE(fN_out, actual_target) * lambda_N_loss
    optimizer_fN.zero_grad()
    N_loss.backward()
    optimizer_fN.step()

    return N_loss


def write_to_log(log_file_path, status):
    """
        Write to the log file
    """

    with open(log_file_path, "a") as log_file:
        log_file.write(status + '\n')


@click.command()
@click.argument('name', default='demo')
@click.option('--data_path', default='C:/Libin/Code/Domain-Adversarial/dataset/artificial_sataset/type1_data/gt',
              help='Path of training input data')
@click.option('--label_path', default='C:/Libin/Code/Domain-Adversarial/dataset/artificial_sataset/type1_data/gt',
              help='Path of training label data')
@click.option('--learning_rate', default=1e-3, help='Learning rate')
@click.option('--batch_size', default=2, help='Batch size')
@click.option('--save_interval', default=5, help='Save models after this many epochs')
@click.option('--start_epoch', default=1, help='Start training from this epoch')
@click.option('--end_epoch', default=200, help='Train till this epoch')
@click.option('--num_classes', default=6, help='Number of water types')
@click.option('--num_channels', default=3, help='Number of input image channels')
@click.option('--train_size', default=10, help='Size of the training dataset')
@click.option('--test_size', default=10, help='Size of the testing dataset')
@click.option('--val_size', default=10, help='Size of the validation dataset')
@click.option('--fe_load_path', default=None, help='Load path for pretrained fN')
@click.option('--fi_load_path', default=None, help='Load path for pretrained fE')
@click.option('--fn_load_path', default=None, help='Load path for pretrained fN')
@click.option('--lambda_i_loss', default=100.0, help='Lambda for I loss')
@click.option('--lambda_n_loss', default=1.0, help='Lambda for N loss')
@click.option('--lambda_adv_loss', default=1.0, help='Lambda for adv loss')
@click.option('--fi_threshold', default=0.2, help='Train fI till this threshold')
@click.option('--fn_threshold', default=0.85, help='Train fN till this threshold')
@click.option('--continue_train', is_flag=True, help='Continue training from start_epoch')
@click.option('--neg_entropy', default=True,
              help='Use KL divergence instead of cross entropy with uniform distribution')
@click.option('--no_adv_loss', is_flag=True, help='Use adversarial loss during training or not')
def main(name, data_path, label_path, learning_rate, batch_size, save_interval, start_epoch, end_epoch, num_classes,
         num_channels,
         train_size, test_size, val_size, fe_load_path, fi_load_path, fn_load_path, lambda_i_loss, lambda_n_loss,
         lambda_adv_loss,
         fi_threshold, fn_threshold, continue_train, neg_entropy, no_adv_loss):
    fE_load_path = fe_load_path
    fI_load_path = fi_load_path
    fN_load_path = fn_load_path

    lambda_I_loss = lambda_i_loss
    lambda_N_loss = lambda_n_loss

    fI_threshold = fi_threshold
    fN_threshold = fn_threshold

    # Define datasets and dataloaders
    train_dataset = NYUUWDataset(data_path,
                                 label_path,
                                 size=train_size,
                                 train_start=0,
                                 mode='train')

    val_dataset = NYUUWDataset(data_path,
                               label_path,
                               size=val_size,
                               val_start=420,
                               mode='val')

    test_dataset = NYUUWDataset(data_path,
                                label_path,
                                size=test_size,
                                test_start=460,
                                mode='test')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    fN = Classifier(num_classes).cuda()
    fN_req_grad = True
    fN.train()
    criterion_CE = nn.CrossEntropyLoss().cuda()
    optimizer_fN = torch.optim.Adam(fN.parameters(), lr=learning_rate,
                                    weight_decay=1e-5)
    Gradient = GradientLoss().cuda()
    Color = ColorLoss().cuda()

    # Define models, criterion and optimizers
    fE = UNetEncoder(num_channels).cuda()
    fI = UNetDecoder(num_channels).cuda()

    criterion_MSE = nn.MSELoss().cuda()

    optimizer_fE = torch.optim.Adam(fE.parameters(), lr=learning_rate,
                                    weight_decay=1e-5)
    optimizer_fI = torch.optim.Adam(fI.parameters(), lr=learning_rate,
                                    weight_decay=1e-5)

    fE.train()
    fI.train()

    if continue_train:
        """
            Load pretrained models to continue training
        """

        if fE_load_path:
            fE.load_state_dict(torch.load(fE_load_path))
            print('Loaded fE from {}'.format(fE_load_path))
        if fI_load_path:
            fI.load_state_dict(torch.load(fI_load_path))
            print('Loaded fI from {}'.format(fI_load_path))
        if not no_adv_loss and fN_load_path:
            fN.load_state_dict(torch.load(fN_load_path))
            print('Loaded fN from {}'.format(fN_load_path))

    if not os.path.exists('./checkpoints/{}'.format(name)):
        os.mkdir('./checkpoints/{}'.format(name))

    log_file_path = './checkpoints/{}/log_file.txt'.format(name)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    status = '\nTRAINING SESSION STARTED ON {}\n'.format(now)
    write_to_log(log_file_path, status)

    # Compute the initial cross validation scores
    if continue_train and not no_adv_loss:
        fI_val_ssim, _, _, fN_val_acc = compute_val_metrics(fE, fI, fN, val_dataloader, no_adv_loss)
    else:
        fI_val_ssim = -1
        fN_val_acc = -1
    fI_val_ssim_best = -1
    fN_val_acc_best = -1
    # Train only the encoder-decoder upto a certain threshold
    while fI_val_ssim < fI_threshold and not continue_train:
        epoch = start_epoch

        status = 'Avg fI val SSIM: {}, Avg fN val acc: {}\nCurrent fI threshold: {}, Current fN threshold: {}'.format(
            fI_val_ssim, fN_val_acc, fI_threshold, fN_threshold)
        print(status)
        write_to_log(log_file_path, status)

        for idx, data in tqdm(enumerate(train_dataloader)):
            uw_img, cl_img, water_type, _, _ = data
            uw_img = Variable(uw_img).cuda()
            cl_img = Variable(cl_img, requires_grad=False).cuda()

            fE_out, enc_outs = fE(uw_img)
            optimizer_fE.zero_grad()
            fI_out, I_loss = backward_I_loss(fI, fE_out, enc_outs, -1, cl_img, criterion_MSE, optimizer_fI,
                                             lambda_I_loss,
                                             retain_graph=not no_adv_loss, domain_drive=False, Gradient=Gradient,
                                             Color=Color)

            progress = "\tEpoch: {}\tIter: {}\tI_loss: {}".format(epoch, idx, I_loss.item())

            optimizer_fE.step()

            if idx % 50 == 0:
                save_image(uw_img.cpu().data, './results/uw_img.png')
                save_image(fI_out.cpu().data, './results/fI_out.png')

                print(progress)
                write_to_log(log_file_path, progress)

        if fI_val_ssim_best < fI_val_ssim:
            fI_val_ssim_best = fI_val_ssim
            torch.save(fE.state_dict(), './checkpoints/{}/fE.pth'.format(name))
            torch.save(fI.state_dict(), './checkpoints/{}/fI.pth'.format(name))

        status = 'End of epoch. Models saved.'
        print(status)
        write_to_log(log_file_path, status)

        fI_val_ssim, _, _, fN_val_acc = compute_val_metrics(fE, fI, fN, val_dataloader, no_adv_loss, domain_drive=False)

        start_epoch += 1

    for epoch in range(start_epoch, end_epoch):
        """
            Main training loop
        """

        if not no_adv_loss:
            """
                Print the current cross-validation scores
            """

            status = 'Avg fI val SSIM: {}, Avg fN val acc: {}\nCurrent fI threshold: {}, Current fN threshold: {}'.format(
                fI_val_ssim, fN_val_acc, fI_threshold, fN_threshold)
            print(status)
            write_to_log(log_file_path, status)

        for idx, data in tqdm(enumerate(train_dataloader)):
            uw_img, cl_img, water_type, _, _ = data
            uw_img = Variable(uw_img).cuda()
            cl_img = Variable(cl_img, requires_grad=False).cuda()
            actual_target = Variable(water_type, requires_grad=False).cuda()

            fE_out, enc_outs = fE(uw_img)


            if fN_val_acc < fN_threshold:
                """
                    Train the classifier only
                """

                if not fN_req_grad:
                    fN_req_grad = set_requires_grad(fN, requires_grad=True)

                N_loss = backward_N_loss(fN, fE_out, actual_target, criterion_CE, optimizer_fN, lambda_N_loss)
                progress = "\tEpoch: {}\tIter: {}\tN_loss: {}".format(epoch, idx, N_loss.item())

            else:
                """
                    Train the encoder-decoder only
                """

                fN_out, fN_out_c = fN(fE_out)
                optimizer_fE.zero_grad()

                fI_out, I_loss = backward_I_loss(fI, fE_out, enc_outs, fN_out_c, cl_img, criterion_MSE, optimizer_fI,
                                                 lambda_I_loss,
                                                 retain_graph=not no_adv_loss, domain_drive=False, Gradient=Gradient,
                                                 Color=Color)
                progress = "\tEpoch: {}\tIter: {}\tI_loss: {}".format(epoch, idx, I_loss.item())

                optimizer_fE.step()

                if idx % 50 == 0:
                    save_image(uw_img.cpu().data, './results/uw_img.png')
                    save_image(fI_out.cpu().data, './results/fI_out.png')

            if idx % 50 == 0:
                print(progress)
                write_to_log(log_file_path, progress)

        # Save models
        if fI_val_ssim_best < fI_val_ssim:
            fI_val_ssim_best = fI_val_ssim
            torch.save(fE.state_dict(), './checkpoints/{}/fE.pth'.format(name))
            torch.save(fI.state_dict(), './checkpoints/{}/fI.pth'.format(name))
        if fN_val_acc_best < fN_val_acc:
            fN_val_acc_best = fN_val_acc
            torch.save(fN.state_dict(), './checkpoints/{}/fN.pth'.format(name))

        status = 'End of epoch. Models saved.'
        print(status)
        write_to_log(log_file_path, status)

        fI_val_ssim, _, _, fN_val_acc = compute_val_metrics(fE, fI, fN, val_dataloader, no_adv_loss, domain_drive=True)


if __name__ == "__main__":
    if not os.path.exists('./results'):
        os.mkdir('./results')
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')

    main()