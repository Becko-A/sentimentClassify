from word_sequence import Word2Sequence
from dataset import tokenlize
import pickle
import os
from tqdm import tqdm

if __name__ == '__main__':
    ws = Word2Sequence()
    path = r"D:\bei\data\IMDB\aclImdb\train"
    temp_data_path = [os.path.join(path, 'pos'), os.path.join(path, 'neg')]
    for data_path in temp_data_path:
        file_paths = [os.path.join(data_path, file_name) for file_name in os.listdir(data_path) if file_name.endswith('txt')]
        for file_path in tqdm(file_paths):
            sentence = tokenlize(open(file_path,encoding='utf-8').read())
            ws.fit(sentence)

    ws.build_vocab(min=10, max_feature=10000)
    pickle.dump(ws, open('./model/ws.pkl', 'wb'))
    print(len(ws))