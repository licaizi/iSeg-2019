from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.augmentations.utils import pad_nd_image
import numpy as np


class ISeg2019DataLoader3D(DataLoader):
    def __init__(self, data, batch_size, patch_size, nb_modalities=2, num_threads_in_multithreaded=1,
                 seed_for_shuffle=1234, return_incomplete=False, shuffle=True, infinite=True):
        '''

        :param data:{id_0: {'img':img_data, 'seg':img_seg},
                     id_1: {'img':img_data, 'seg':img_seg}}
        :param batch_size:
        :param patch_size:
        :param nb_modalities:
        :param num_threads_in_multithreaded:
        :param seed_for_shuffle:
        :param return_incomplete:
        :param shuffle:
        :param infinite:
        '''
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete,
                         shuffle, infinite)
        self.patch_size = patch_size
        self.nb_modalities = nb_modalities
        self.indices = list(range(len(data)))

    def generate_train_batch(self):
        idx = self.get_indices()    # get batch_size subjects randomly, return a list
        data = np.zeros((self.batch_size, self.nb_modalities, *self.patch_size), dtype=np.float32)
        seg = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.int64)

        for i, j in enumerate(idx):
            tmp_data = pad_nd_image(self._data[j]['img'], self.patch_size)
            tmp_seg = pad_nd_image(self._data[j]['seg'], self.patch_size)
            cropped_data, cropped_seg = crop(tmp_data, tmp_seg, self.patch_size, crop_type='random')
            data[i] = cropped_data[0]
            seg[i] = cropped_seg[0]
        return {'data': data, 'seg': seg}   # (b, 2, d, h, w), (b, 1, d, h, w)


class ISeg2019DataLoader3D_Unlabel(DataLoader):
    def __init__(self, data, batch_size, patch_size, nb_modalities=2, num_threads_in_multithreaded=1,
                 seed_for_shuffle=1234, return_incomplete=False, shuffle=True, infinite=True):
        '''

        :param data:{id_0: {'img':img_data, 'seg':img_seg},
                     id_1: {'img':img_data, 'seg':img_seg}}
        :param batch_size:
        :param patch_size:
        :param nb_modalities:
        :param num_threads_in_multithreaded:
        :param seed_for_shuffle:
        :param return_incomplete:
        :param shuffle:
        :param infinite:
        '''
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete,
                         shuffle, infinite)
        self.patch_size = patch_size
        self.nb_modalities = nb_modalities
        self.indices = list(range(len(data)))

    def generate_train_batch(self):
        idx = self.get_indices()    # get batch_size subjects randomly, return a list
        data = np.zeros((self.batch_size, self.nb_modalities, *self.patch_size), dtype=np.float32)

        for i, j in enumerate(idx):
            tmp_data = pad_nd_image(self._data[j]['img'], self.patch_size)
            cropped_data, cropped_seg = crop(tmp_data, None, self.patch_size, crop_type='random')
            data[i] = cropped_data[0]
        return {'data': data}   # (b, 2, d, h, w), (b, 1, d, h, w)


