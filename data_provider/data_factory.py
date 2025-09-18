from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, LLSD_Dataset_Custom
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom,
    'LLSD': LLSD_Dataset_Custom
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    # timeenc = 0 if args.embed != 'timeF' else 1   # 时间特征编码方式

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        # Data = Dataset_Pred
    else:
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        args.root_path,
        args.dataset_name,
        args.seq_len,
        args.pred_len,
        flag
    )

    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_flag, num_workers=0)

    return data_set, data_loader

