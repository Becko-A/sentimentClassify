"""
完成数据集的准备
"""
import os.path
import re
import torch

import lib
from lib import ws, max_len

from torch.utils.data import DataLoader, Dataset


def tokenlize(content):
    content = re.sub("<.*?>", "", content)
    filters = ['\.', '\t', '\n', '\x97', '\x96', '#', '$', '%', '&', ':']
    content = re.sub('|'.join(filters), " ", content)
    tokens = [i.strip().lower() for i in content.split()]
    return tokens


class ImdbDataset(Dataset):
    def __init__(self, train=True):
        self.train_data_path = r"D:\bei\data\IMDB\aclImdb\train"
        self.test_data_path = r"D:\bei\data\IMDB\aclImdb\test"
        data_path = self.train_data_path if train else self.test_data_path

        # 把所有文件名放入列表
        temp_data_path = [os.path.join(data_path, 'pos'), os.path.join(data_path, 'neg')]
        self.total_data_path = []  # 所有的评论文件的path
        for path in temp_data_path:
            file_name_list = os.listdir(path)
            file_path_list = [os.path.join(path, i) for i in file_name_list if i.endswith('.txt')]
            self.total_data_path.extend(file_path_list)

    def __getitem__(self, index):
        file_path = self.total_data_path[index]  # D:\bei\data\IMDB\aclImdb\train\pos\0_9.txt
        # 获取label
        label_str = file_path.split("\\")[-2]
        label = 0 if label_str == "neg" else 1
        # 获取内容
        content = open(file_path, encoding='utf-8').read()
        tokens = tokenlize(content)
        return tokens, label

    def __len__(self):
        return len(self.total_data_path)


def collate_fn(batch):
    """
    :param batch: ([tokens,label]，[tokens,label]...)
    :return:
    """
    content, label = list(zip(*batch))
    # print('content1:',content)
    content = [ws.transform(i, max_len=max_len) for i in content]
    # print('content2:', content)
    content = torch.LongTensor(content)
    label = torch.LongTensor(label)
    return content, label


def get_dataloader(train=True, batch_size = lib.batch_size):
    imdb_dataset = ImdbDataset(train)
    data_loader = DataLoader(imdb_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return data_loader


if __name__ == '__main__':
    for idx, (input, target) in enumerate(get_dataloader()):
        print(idx)
        print(input)
        print(target)
        break
