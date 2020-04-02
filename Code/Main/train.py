import os
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import shutil
import numpy as np
from DenseNet_3D import DenseNet_3D
from Model.Discriminator import FCDiscriminator3D
from Config.config import file_paths, train_params, model_params
from Data_preprocessing.data_generator import ISeg2019DataLoader3D, ISeg2019DataLoader3D_Unlabel
from Data_preprocessing.data_utils import load_train_dataset, load_train_dataset_d, generate_img_patches, generate_score_map_patch2Img
from batchgenerators.dataloading import MultiThreadedAugmenter
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms import RndTransform
from batchgenerators.transforms import Compose
from Utils.func import prob_2_entropy

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename, model_best_name):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, model_best_name)
        shutil.copyfile(filename, bestname)


def get_train_transform(patch_size):
    tr_transforms = []
    tr_transforms.append(
        SpatialTransform_2(
            None, [i // 2 for i in patch_size],
            do_elastic_deform=False, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(-15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(-15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_z=(-15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=False, scale=(0.8, 1.2),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=False,
            p_el_per_sample=0.3, p_rot_per_sample=0.3, p_scale_per_sample=0.3
        )
    )
    tr_transforms.append(RndTransform(MirrorTransform(axes=(0, 1, 2)), prob=0.3))
    tr_transforms = Compose(transforms=tr_transforms)
    return tr_transforms


def dice(im1, im2, tid):
    im1 = im1 == tid
    im2 = im2 == tid
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    dsc = 2. * intersection.sum() / (im1.sum() + im2.sum())
    return dsc


def validation(net, dataset):
    ita = 4     # set higher, the result will be better
    dice1_list = []
    dice2_list = []
    dice3_list = []
    for key in dataset.keys():
        img = np.squeeze(dataset[key]['img'], axis=0)   # (2, d, h, w)
        label = np.squeeze(np.squeeze(dataset[key]['seg'], axis=0), axis=0) # (d, h, w)
        crop_data_list, ss_c, ss_h, ss_w, ss_l, padding_size_x, padding_size_y, padding_size_z = \
            generate_img_patches(img, model_params['patch_size'], ita)

        score_map_list = []
        for i in range(len(crop_data_list)):
            tmp_out_patch = net(torch.from_numpy(np.expand_dims(crop_data_list[i], axis=0)).cuda())
            softmax_output = F.softmax(tmp_out_patch.detach(), dim=1)
            output_numpy = softmax_output.argmax(dim=1).cpu().numpy()    # (1, 64, 64, 64)
            arg_output = np.squeeze(output_numpy, axis=0)    # (64, 64, 64)
            score_map_list.append(arg_output)
        pred_map = generate_score_map_patch2Img(score_map_list, ss_h, ss_w, ss_l, padding_size_x, padding_size_y,
                                                padding_size_z, model_params['patch_size'], ita)    # (d, h, w)
        dice1_list.append(dice(pred_map, label, 1))
        dice2_list.append(dice(pred_map, label, 2))
        dice3_list.append(dice(pred_map, label, 3))
    avg_dice1 = np.mean(dice1_list)
    avg_dice2 = np.mean(dice2_list)
    avg_dice3 = np.mean(dice3_list)
    avg_dice = np.mean([avg_dice1, avg_dice2, avg_dice3])
    return avg_dice1, avg_dice2, avg_dice3, avg_dice


def main():
    # init log
    now = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    log_path = train_params['log_path']
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    log = open(os.path.join(log_path, 'log_{}.txt'.format(now)), 'w')
    print_log('save path : {}'.format(log_path), log)

    # prepare dataset
    dataset = load_train_dataset(phase='train', data_list_path=file_paths['train_list'])
    da_dataset = load_train_dataset_d(phase='train', data_list_path=file_paths['test_list'])
    # augmentation
    aug_transforms = get_train_transform(model_params['patch_size'])
    # source domain
    src_data_gen = ISeg2019DataLoader3D(dataset, model_params['batch_size'], model_params['patch_size'], nb_modalities=2,
                                        num_threads_in_multithreaded=4)
    src_aug_gen = MultiThreadedAugmenter(src_data_gen, aug_transforms, num_processes=4, num_cached_per_queue=4)
    src_aug_gen.restart()
    # target domain
    tgt_data_gen = ISeg2019DataLoader3D_Unlabel(da_dataset, model_params['batch_size'], model_params['patch_size'], nb_modalities=2,
                                                num_threads_in_multithreaded=4)
    tgt_aug_gen = MultiThreadedAugmenter(tgt_data_gen, aug_transforms, num_processes=4, num_cached_per_queue=4)
    tgt_aug_gen.restart()

    # define network
    net = DenseNet_3D(ver=model_params['model_ver'])
    net_d = FCDiscriminator3D(num_classes=model_params['nb_classes'], ndf=32)

    # define loss
    seg_loss = torch.nn.CrossEntropyLoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()

    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=train_params['lr_rate'], weight_decay=train_params['weight_decay'],
                                 betas=(train_params['momentum'], 0.999))
    optimizer.zero_grad()
    optimizer_d = torch.optim.Adam(net_d.parameters(), lr=train_params['lr_rate_d'], weight_decay=train_params['weight_decay'],
                                   betas=(0.9, 0.99))
    optimizer_d.zero_grad()

    start_step = 0
    best_dice = 0.
    if train_params['resume_path'] is not None:
        print_log("=======> loading checkpoint '{}'".format(train_params['resume_path']), log=log)
        checkpoint = torch.load(train_params['resume_path'])
        net.load_state_dict(checkpoint['model_state_dict'])
        print_log("=======> loaded checkpoint '{}'".format(train_params['resume_path']), log=log)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, train_params['lr_step_size'], train_params['lr_gamma'])
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, train_params['lr_step_size'], train_params['lr_gamma'])

    # start training
    net.cuda()
    net_d.cuda()
    seg_loss.cuda()
    bce_loss.cuda()
    net.train()
    net_d.train()
    source_label = 1
    target_label = 0
    for step in range(start_step, train_params['nb_iters']):
        loss_seg_value = 0.
        loss_adv_target_value = 0.
        loss_D_src_value = 0.
        loss_D_tgt_value = 0.

        optimizer.zero_grad()
        optimizer_d.zero_grad()

        for sub_iter in range(train_params['nb_accu_iters']):
            # train G
            for param in net_d.parameters():
                param.requires_grad = False

            # train with source
            src_batch = next(src_aug_gen)
            src_input_img = torch.from_numpy(src_batch['data']).cuda()
            src_input_label = torch.from_numpy(np.squeeze(src_batch['seg'], axis=1).astype(np.int64)).cuda()

            src_seg_out = net(src_input_img)
            loss = seg_loss(src_seg_out, src_input_label)
            loss_seg = loss / train_params['nb_accu_iters']
            loss_seg.backward()

            loss_seg_value += loss.data.cpu().numpy()

            # train with target
            tgt_batch = next(tgt_aug_gen)
            tgt_input_img = torch.from_numpy(tgt_batch['data']).cuda()

            tgt_seg_out = net(tgt_input_img)
            tgt_d_out = net_d(prob_2_entropy(F.softmax(tgt_seg_out, dim=1)))
            loss = bce_loss(tgt_d_out, Variable(torch.FloatTensor(tgt_d_out.data.size()).fill_(source_label)).cuda())
            loss_adv_tgt = train_params['lambda_adv_target'] * loss / train_params['nb_accu_iters']
            loss_adv_tgt.backward()

            loss_adv_target_value += loss.data.cpu().numpy()

            # train D
            for param in net_d.parameters():
                param.requires_grad = True

            # train with source
            src_seg_out = src_seg_out.detach()
            src_d_out = net_d(prob_2_entropy(F.softmax(src_seg_out, dim=1)))
            loss = bce_loss(src_d_out, Variable(torch.FloatTensor(src_d_out.data.size()).fill_(source_label)).cuda())
            loss_d_src = loss / train_params['nb_accu_iters']
            loss_d_src.backward()

            loss_D_src_value += loss.data.cpu().numpy()

            # train with target
            tgt_seg_out = tgt_seg_out.detach()
            tgt_d_out = net_d(prob_2_entropy(F.softmax(tgt_seg_out, dim=1)))
            loss = bce_loss(tgt_d_out, Variable(torch.FloatTensor(tgt_d_out.data.size()).fill_(target_label)).cuda())
            loss_d_tgt = loss / train_params['nb_accu_iters']
            loss_d_tgt.backward()

            loss_D_tgt_value += loss.data.cpu().numpy()

        optimizer.step()
        scheduler.step()
        optimizer_d.step()
        scheduler_d.step()

        log_str = 'step {}: lr:{:.8f}, lr_d:{:.8f}, loss_seg:{:.6f}, loss_adv:{:.6f}, loss_D_src:{:.6f}, loss_D_tgt:{:.6f}'\
            .format(step, scheduler.get_lr()[0], scheduler_d.get_lr()[0], loss_seg_value, loss_adv_target_value, loss_D_src_value, loss_D_tgt_value)
        print_log(log_str, log)

        # val and save per N iterations
        if (step + 1) % train_params['snapshot_step_size'] == 0:
            net.eval()
            val_avg_dice1, val_avg_dice2, val_avg_dice3, val_avg_dice = validation(net, dataset)
            val_log_str = 'val step: val_avg_dice:{}, val_avg_dice1:{}, val_avg_dice2:{}, val_avg_dice3:{}' \
                .format(val_avg_dice, val_avg_dice1, val_avg_dice2, val_avg_dice3)
            print_log(val_log_str, log)

            is_best = False
            if val_avg_dice > best_dice:
                best_dice = val_avg_dice
                is_best = True

            save_checkpoint({
                'model_state_dict': net.state_dict(),
            }, is_best, train_params['model_snapshot_path'], 'checkpoint-{}.pth'.format(step+1), 'model_best.pth')
            save_checkpoint({
                'model_d': net_d.state_dict(),
            }, is_best, train_params['model_snapshot_path'], 'checkpoint-d-{}.pth'.format(step+1), 'model_d_best.pth')

            net.train()
    log.close()


if __name__ == '__main__':
    main()

