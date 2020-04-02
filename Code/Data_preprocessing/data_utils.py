import h5py
import numpy as np
import math
import os
from medpy.io import load


def load_train_dataset(phase='train', data_list_path=None, leave_out_list=None, data_path=None):
    dataset = {}
    if phase == 'train':
        with open(data_list_path, 'r') as f:
            lines = f.read().splitlines()
            for i, path in enumerate(lines):
                dataset[i] = {}
                with h5py.File(path, 'r') as h5_f:
                    dataset[i]['img'] = h5_f['data'][:]
                    dataset[i]['seg'] = h5_f['label'][:]
    elif phase == 'val':
        for id in leave_out_list:
            dataset[id] = {}
            subject_name = 'subject-%d-' % (id + 1)
            f_T1 = os.path.join(data_path, subject_name + 'T1.hdr')
            inputs_T1, header_T1 = load(f_T1)
            inputs_T1 = inputs_T1.astype(np.float32)

            f_T2 = os.path.join(data_path, subject_name + 'T2.hdr')
            inputs_T2, header_T2 = load(f_T2)
            inputs_T2 = inputs_T2.astype(np.float32)

            f_l = os.path.join(data_path, subject_name + 'label.hdr')
            labels, header_label = load(f_l)
            labels = labels.astype(np.uint8)

            mask = inputs_T1 > 0
            mask = mask.astype(np.bool)

            labels = np.expand_dims(np.expand_dims(labels, axis=0), axis=0)
            labels = np.transpose(labels, axes=[0, 1, 4, 2, 3])

            inputs_T1_norm = (inputs_T1 - inputs_T1[mask].mean()) / inputs_T1[mask].std()
            inputs_T2_norm = (inputs_T2 - inputs_T2[mask].mean()) / inputs_T2[mask].std()

            inputs_T1_norm = inputs_T1_norm[:, :, :, None]
            inputs_T2_norm = inputs_T2_norm[:, :, :, None]

            inputs = np.concatenate((inputs_T1_norm, inputs_T2_norm), axis=3)
            inputs = inputs[None, :, :, :, :]
            inputs = inputs.transpose(0, 4, 3, 1, 2)
            dataset[id]['img'] = inputs
            dataset[id]['seg'] = labels
    return dataset


def load_train_dataset_d(phase='train', data_list_path=None):
    dataset = {}
    if phase == 'train':
        with open(data_list_path, 'r') as f:
            lines = f.read().splitlines()
            for i, path in enumerate(lines):
                dataset[i] = {}
                with h5py.File(path, 'r') as h5_f:
                    dataset[i]['img'] = h5_f['data'][:]
    return dataset


def load_test_dataset(test_ids=None, data_path=None):
    dataset = {}
    for id in test_ids:
        dataset[id] = {}
        subject_name = 'subject-%d-' % (id + 1)
        f_T1 = os.path.join(data_path, subject_name + 'T1.hdr')
        inputs_T1, header_T1 = load(f_T1)
        inputs_T1 = inputs_T1.astype(np.float32)

        f_T2 = os.path.join(data_path, subject_name + 'T2.hdr')
        inputs_T2, header_T2 = load(f_T2)
        inputs_T2 = inputs_T2.astype(np.float32)

        mask = inputs_T1 > 0
        mask = mask.astype(np.bool)

        inputs_T1_norm = (inputs_T1 - inputs_T1[mask].mean()) / inputs_T1[mask].std()
        inputs_T2_norm = (inputs_T2 - inputs_T2[mask].mean()) / inputs_T2[mask].std()

        inputs_T1_norm = inputs_T1_norm[:, :, :, None]
        inputs_T2_norm = inputs_T2_norm[:, :, :, None]

        inputs = np.concatenate((inputs_T1_norm, inputs_T2_norm), axis=3)
        inputs = inputs[None, :, :, :, :]
        inputs = inputs.transpose(0, 4, 3, 1, 2)
        dataset[id]['img'] = inputs
        dataset[id]['mask'] = mask
    return dataset


def convert_to_one_hot(seg):
    vals = np.unique(seg)
    res = np.zeros([len(vals)] + list(seg.shape), seg.dtype)
    for c in range(len(vals)):
        res[c][seg == c] = 1
    return res


def generate_img_patches(img, patch_size, ita):  # (c, d, h, w)
    ss_c, ss_h, ss_w, ss_l = img.shape

    # pad the img if the size is smaller than the crop size
    padding_size_x = max(0, math.ceil((patch_size[0] - ss_h) / 2))
    padding_size_y = max(0, math.ceil((patch_size[1] - ss_w) / 2))
    padding_size_z = max(0, math.ceil((patch_size[2] - ss_l) / 2))
    img = np.pad(img, ((0, 0), (padding_size_x, padding_size_x), (padding_size_y, padding_size_y),
                       (padding_size_z, padding_size_z)), 'constant')

    ss_c, ss_h, ss_w, ss_l = img.shape

    fold_h = math.floor(ss_h / patch_size[0]) + ita
    fold_w = math.floor(ss_w / patch_size[1]) + ita
    fold_l = math.floor(ss_l / patch_size[2]) + ita
    overlap_h = int(math.ceil((ss_h - patch_size[0]) / (fold_h - 1)))
    overlap_w = int(math.ceil((ss_w - patch_size[1]) / (fold_w - 1)))
    overlap_l = int(math.ceil((ss_l - patch_size[2]) / (fold_l - 1)))
    idx_h = [0] if overlap_h == 0 else [i for i in range(0, ss_h - patch_size[0] + 1, overlap_h)]
    idx_h.append(ss_h - patch_size[0])
    idx_h = np.unique(idx_h)
    idx_w = [0] if overlap_w == 0 else [i for i in range(0, ss_w - patch_size[1] + 1, overlap_w)]
    idx_w.append(ss_w - patch_size[1])
    idx_w = np.unique(idx_w)
    idx_l = [0] if overlap_l == 0 else [i for i in range(0, ss_l - patch_size[2] + 1, overlap_l)]
    idx_l.append(ss_l - patch_size[2])
    idx_l = np.unique(idx_l)

    crop_data_list = []
    for itr_h in idx_h:
        for itr_w in idx_w:
            for itr_l in idx_l:
                crop_data = img[:, itr_h: itr_h + patch_size[0], itr_w: itr_w + patch_size[1],
                            itr_l: itr_l + patch_size[2]]
                crop_data_list.append(crop_data)
    return crop_data_list, ss_c, ss_h, ss_w, ss_l, padding_size_x, padding_size_y, padding_size_z


def generate_score_map_patch2Img(patch_list, ss_h, ss_w, ss_l, padding_size_x, padding_size_y, padding_size_z,
                                 patch_size, ita):
    label_0_array = np.zeros((ss_h, ss_w, ss_l))
    label_1_array = np.zeros((ss_h, ss_w, ss_l))
    label_2_array = np.zeros((ss_h, ss_w, ss_l))
    label_3_array = np.zeros((ss_h, ss_w, ss_l))
    label_array = np.zeros((ss_h, ss_w, ss_l, 4))

    fold_h = math.floor(ss_h / patch_size[0]) + ita
    fold_w = math.floor(ss_w / patch_size[1]) + ita
    fold_l = math.floor(ss_l / patch_size[2]) + ita
    overlap_h = int(math.ceil((ss_h - patch_size[0]) / (fold_h - 1)))
    overlap_w = int(math.ceil((ss_w - patch_size[1]) / (fold_w - 1)))
    overlap_l = int(math.ceil((ss_l - patch_size[2]) / (fold_l - 1)))
    idx_h = [0] if overlap_h == 0 else [i for i in range(0, ss_h - patch_size[0] + 1, overlap_h)]
    idx_h.append(ss_h - patch_size[0])
    idx_h = np.unique(idx_h)
    idx_w = [0] if overlap_w == 0 else [i for i in range(0, ss_w - patch_size[1] + 1, overlap_w)]
    idx_w.append(ss_w - patch_size[1])
    idx_w = np.unique(idx_w)
    idx_l = [0] if overlap_l == 0 else [i for i in range(0, ss_l - patch_size[2] + 1, overlap_l)]
    idx_l.append(ss_l - patch_size[2])
    idx_l = np.unique(idx_l)

    p_count = 0
    for itr_h in idx_h:
        for itr_w in idx_w:
            for itr_l in idx_l:
                idx_0 = np.float16(patch_list[p_count] == 0)
                idx_1 = np.float16(patch_list[p_count] == 1)
                idx_2 = np.float16(patch_list[p_count] == 2)
                idx_3 = np.float16(patch_list[p_count] == 3)
                label_0_array[itr_h: itr_h + patch_size[0], itr_w: itr_w + patch_size[1],
                itr_l: itr_l + patch_size[2]] += idx_0
                label_1_array[itr_h: itr_h + patch_size[0], itr_w: itr_w + patch_size[1],
                itr_l: itr_l + patch_size[2]] += idx_1
                label_2_array[itr_h: itr_h + patch_size[0], itr_w: itr_w + patch_size[1],
                itr_l: itr_l + patch_size[2]] += idx_2
                label_3_array[itr_h: itr_h + patch_size[0], itr_w: itr_w + patch_size[1],
                itr_l: itr_l + patch_size[2]] += idx_3

                p_count += 1

    label_array[:, :, :, 0] = label_0_array
    label_array[:, :, :, 1] = label_1_array
    label_array[:, :, :, 2] = label_2_array
    label_array[:, :, :, 3] = label_3_array

    vote_label = np.argmax(label_array, axis=3)
    score_map = vote_label[padding_size_x: vote_label.shape[0] - padding_size_x,
                padding_size_y: vote_label.shape[1] - padding_size_y,
                padding_size_z: vote_label.shape[2] - padding_size_z]

    return score_map


def generate_d_map_patch2Img(patch_list, ss_h, ss_w, ss_l, padding_size_x, padding_size_y, padding_size_z,
                             patch_size, ita):
    label_array = np.zeros((ss_h, ss_w, ss_l))
    cnt_array = np.zeros((ss_h, ss_w, ss_l))

    fold_h = math.floor(ss_h / patch_size[0]) + ita
    fold_w = math.floor(ss_w / patch_size[1]) + ita
    fold_l = math.floor(ss_l / patch_size[2]) + ita
    overlap_h = int(math.ceil((ss_h - patch_size[0]) / (fold_h - 1)))
    overlap_w = int(math.ceil((ss_w - patch_size[1]) / (fold_w - 1)))
    overlap_l = int(math.ceil((ss_l - patch_size[2]) / (fold_l - 1)))
    idx_h = [0] if overlap_h == 0 else [i for i in range(0, ss_h - patch_size[0] + 1, overlap_h)]
    idx_h.append(ss_h - patch_size[0])
    idx_h = np.unique(idx_h)
    idx_w = [0] if overlap_w == 0 else [i for i in range(0, ss_w - patch_size[1] + 1, overlap_w)]
    idx_w.append(ss_w - patch_size[1])
    idx_w = np.unique(idx_w)
    idx_l = [0] if overlap_l == 0 else [i for i in range(0, ss_l - patch_size[2] + 1, overlap_l)]
    idx_l.append(ss_l - patch_size[2])
    idx_l = np.unique(idx_l)

    p_count = 0
    for itr_h in idx_h:
        for itr_w in idx_w:
            for itr_l in idx_l:
                label_array[itr_h: itr_h + patch_size[0], itr_w: itr_w + patch_size[1],
                itr_l: itr_l + patch_size[2]] += patch_list[p_count]
                cnt_array[itr_h: itr_h + patch_size[0], itr_w: itr_w + patch_size[1],
                itr_l: itr_l + patch_size[2]] += 1
                p_count += 1

    vote_label = label_array / cnt_array
    score_map = vote_label[padding_size_x: vote_label.shape[0] - padding_size_x,
                padding_size_y: vote_label.shape[1] - padding_size_y,
                padding_size_z: vote_label.shape[2] - padding_size_z]

    return score_map



