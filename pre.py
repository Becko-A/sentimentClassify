"""
单条数据预测
"""
import os.path
import pickle
from dataset import tokenlize

import torch.nn as nn
from lib import ws, max_len
import lib
import torch.nn.functional as F
from dataset import get_dataloader
import torch
import PySimpleGUI as sg


def gui():
    layout = [
        [sg.Text('输入文本：'), sg.InputText(key='input')],
        [sg.Text('情感判断：'), sg.InputText(key='output')],
        [sg.Button('识别'), sg.Button('关闭')],
    ]
    window = sg.Window('My Window', layout)
    while True:
        event, value = window.Read()

        s1 = value['input']
        output = rec(s1)
        if event in (None, '关闭'):  # 如果用户关闭窗口或点击`关闭`
            break
        if event == '识别':
            window['output'].update(output)
        if event in (None, '关闭'):  # 如果用户关闭窗口或点击`关闭`
            break

    window.Close()


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(len(ws), 100)
        self.lstm = nn.LSTM(input_size=100, hidden_size=lib.hidden_size, num_layers=lib.num_layers,
                            batch_first=True, bidirectional=lib.bidirectional, dropout=lib.drop_out)
        self.fc = nn.Linear(lib.hidden_size * 2, 2)

    def forward(self, input):
        """
        :param input:[batch_size, max_len]
        :return:
        """
        x = self.embedding(input)  # 进行embedding操作，形状：[batch_size, max_len, 100]
        # x:[batch_size, max_len, hidden_size], h_n:[2*2, batch_size, hidden_size]
        x, (h_n, c_n) = self.lstm(x)
        # 获取两个方向最后一层的output concat
        output_fw = h_n[-2, :, :]  # 正向最后一次的输出
        output_bw = h_n[-1, :, :]  # 反向
        output = torch.cat([output_fw, output_bw], dim=-1)  # [batch_size, hidden_size*2]

        out = self.fc(output)
        return F.log_softmax(out, dim=-1)


model = MyModel().to(lib.device)
if os.path.exists('./model/model.pkl'):
    model.load_state_dict(torch.load('./model/model.pkl'))


def rec(content):
    content = tokenlize(content)
    content = content,
    content = (ws.transform(i, max_len=max_len) for i in content)
    content = list(content)
    content = torch.LongTensor(content)
    output = model(content)
    pred = output.max(dim=-1)[-1]
    if pred == torch.tensor(0):
        return '该评论为负面'
    else:
        return '该评论为正面'


if __name__ == '__main__':
    ws = pickle.load(open('./model/ws.pkl', 'rb'))
    data_loader = get_dataloader(train=False, batch_size=1)
    gui()
