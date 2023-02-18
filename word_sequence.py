"""
实现：构建字典，实现方法把句子转化成数字序列和其翻转
"""


class Word2Sequence:
    UNK_TAG = 'UNK'
    PAD_TAG = 'PAD'

    UNK = 0
    PAD = 1

    def __init__(self):
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }

        self.count = {}  # 统计词频

    def fit(self, sentence):
        """
        把单个句子保存到dict中
        :param sentence:[word1,word2...]
        :return:
        """
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min=5, max=None, max_feature=None):
        """
        生成词典
        :param min:最小出现的次数
        :param max: 最大出现的次数
        :param max_feature: 一共保留多少个词语
        :return:
        """
        # 删除count中词频小于min的word
        if min is not None:
            self.count = {word: value for word, value in self.count.items() if value > min}
        # 删除count中词频大于max的word
        if max is not None:
            self.count = {word: value for word, value in self.count.items() if value < max}
        # 限制保留的词语数
        if max_feature is not None:
            temp = sorted(self.count.items(), key=lambda x: x[-1], reverse=True)[:max_feature]
            self.count = dict(temp)

        for word in self.count:
            self.dict[word] = len(self.dict)  # 索引值 1 2 3

        # 得到反转的dict
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len=None):
        """
        把句子转化为序列
        :param sentence:[word1, word2...]
        :param max_len: int,对句子进行填充
        :return:
        """
        # for word in sentence:
        #     self.dict.get(word, self.UNK)
        if max_len is not None:
            if max_len > len(sentence):
                sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))  # 填充
            if max_len < len(sentence):
                sentence = sentence[:max_len]  # 取前max位

        return [self.dict.get(word, self.UNK) for word in sentence]

    def inverse_transform(self, indices):
        """
        把序列转化为句子
        :param indices:[1,2,3...]
        :return:
        """
        return [self.inverse_dict.get(idx) for idx in indices]

    def __len__(self):
        return len(self.dict)


if __name__ == '__main__':
    ws = Word2Sequence()
    ws.fit(['我', '是', '谁'])
    ws.fit(['我', '是', '我'])
    ws.build_vocab(min=0)

    print(ws.dict)
    ret = ws.transform(['我', '爱', '北京'], max_len=10)
    print(ret)
    ret = ws.inverse_transform(ret)
    print(ret)

