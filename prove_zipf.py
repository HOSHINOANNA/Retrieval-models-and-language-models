import re
import numpy as np
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

englishStemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))


def get_frequency(line_list):
    word_count_dict = {}
    for sentense in line_list:
        # passage_index += 1
        text = re.sub("[\s+\.\!\/_,$|%^*(+\"\'><):-]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+",
                      " ", sentense).lower()
        words = word_tokenize(text)
        #filter stop words, stemming
        for w in words:
            # word_tokens.append(englishStemmer.stem(w))
            stemmed_word = englishStemmer.stem(w)
            word_count_dict[stemmed_word] = word_count_dict[
                                                stemmed_word] + 1 if stemmed_word in word_count_dict else 1

    return word_count_dict


def get_txt(path):
    f = open(path, encoding="utf-8")
    string_data = f.read()
    f.close()
    passage_list = string_data.split('\n')
    return passage_list


def draw_zipf(frequency_list):
    plt.title('Zipf-Law', fontsize=18)
    plt.xlabel('log 10 rank', fontsize=18)  # rank
    plt.ylabel('log 10 freq', fontsize=18)  # freq
    length = len(frequency_list)
    x = np.arange(1, length + 1, 1)
    # plt.yticks([pow(10, i) for i in range(0, 4)])
    # plt.xticks([pow(10, i) for i in range(0, 4)])

    log_x = np.log10(x).reshape(-1, 1)
    log_y = np.log10(frequency_list)
    plt.plot(log_x, log_y, 'r', color='g')

    b = 6

    model_y = b - np.log10(x)

    plt.plot(log_x, model_y)

    plt.show()


def run():
    txt_path = 'dataset/passage_collection.txt'
    txt_line = get_txt(txt_path)
    word_count_dict = get_frequency(txt_line)
    frequency_list = sorted(word_count_dict.values(), reverse=True)
    draw_zipf(frequency_list)


if __name__ == '__main__':
    run()
