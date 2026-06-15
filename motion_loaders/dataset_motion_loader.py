from data.t2m_dataset import Text2MotionDatasetEval, collate_fn # TODO
from utils.word_vectorizer import WordVectorizer
import numpy as np
from os.path import join as pjoin
from torch.utils.data import DataLoader
from utils.get_opt import get_opt

def get_dataset_motion_loader(
        opt_path, batch_size, fname, device,
        data_root=None, motion_dir=None, text_dir=None, max_motion_length=None):
    opt = get_opt(opt_path, device)
    if data_root is not None:
        opt.data_root = data_root
    if motion_dir is not None:
        opt.motion_dir = motion_dir
    if text_dir is not None:
        opt.text_dir = text_dir
    if max_motion_length is not None:
        opt.max_motion_length = max_motion_length

    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.dataset_name == 't2m' or opt.dataset_name == 'kit':
        print('Loading dataset %s ...' % opt.dataset_name)

        mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
        std = np.load(pjoin(opt.meta_dir, 'std.npy'))

        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        split_file = pjoin(opt.data_root, '%s.txt'%fname)
        dataset = Text2MotionDatasetEval(opt, mean, std, split_file, w_vectorizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=True,
                                collate_fn=collate_fn, shuffle=True)
    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset
