import os
import torch
import torch.nn.functional as F
import numpy as np
from DenseNet_3D import DenseNet_3D
from Config.config import file_paths, model_params, PROJECT_ROOT
from Data_preprocessing.data_utils import load_test_dataset, \
    generate_img_patches, generate_score_map_patch2Img


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

test_save_path = PROJECT_ROOT + 'Dataset/test_ret'
model_path = PROJECT_ROOT + 'Code/snapshot/checkpoint.pth'


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


def validation_test(net, dataset):
    ita = 4
    pred_map = None
    for key in dataset.keys():
        img = np.squeeze(dataset[key]['img'], axis=0)   # (2, d, h, w)
        crop_data_list, ss_c, ss_h, ss_w, ss_l, padding_size_x, padding_size_y, padding_size_z = \
            generate_img_patches(img, model_params['patch_size'], ita)
        score_map_list = []
        for i in range(len(crop_data_list)):
            tmp_out_patch = net(torch.from_numpy(np.expand_dims(crop_data_list[i], axis=0)).cuda())
            softmax_output = F.softmax(tmp_out_patch, dim=1)
            output_numpy = softmax_output.argmax(dim=1).cpu().numpy()    # (1, 64, 64, 64)
            arg_output = np.squeeze(output_numpy, axis=0)    # (64, 64, 64)
            score_map_list.append(arg_output)
        pred_map = generate_score_map_patch2Img(score_map_list, ss_h, ss_w, ss_l, padding_size_x, padding_size_y,
                                                padding_size_z, model_params['patch_size'], ita)    # (d, h, w)
        pred_map = pred_map.astype(np.uint8)
    return pred_map


net = DenseNet_3D(ver=model_params['model_ver'])
net.cuda()

checkpoint = torch.load(model_path)
net.load_state_dict(checkpoint['model_state_dict'])

net.eval()

for i in range(23, 39):
    print('------->test subject {}: '.format(i + 1))
    subject_name = 'subject-%d-' % (i + 1)
    test_dataset = load_test_dataset(test_ids=[i], data_path=file_paths['src_test_data_path'])
    pred_map = validation_test(net, test_dataset)
    pred_map = np.moveaxis(pred_map, 0, -1)
    mask = test_dataset[i]['mask']
    pred_map[mask == False] = 0
    # save(pred_map, os.path.join(test_save_path, subject_name + 'label.hdr'))
print('=======================================')









