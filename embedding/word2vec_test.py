import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import os
import sys
import math
from embedding.word2vec import Word2Vec
from data.voc import WordVoc

"""
refer: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
word2vec 테스트 코드
directory 생성: ./model/embedding
학습: python -m embedding.word2vec_test train 1000
시험: python -m embedding.word2vec_test
"""

p_saver = './model/embedding/word2vec'
p_text = './data/constitution.txt'
p_voc = './data/constitution.voc'


def train(n_epoch, voc, word2vec):
    # voc에 skipgram 데이터 생성
    voc.build_skipgram(2)

    n_batch = 128
    n_step = int(voc.len_sgram / n_batch)

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(p_saver)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('load model!!')
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('init model!!')
            sess.run(tf.global_variables_initializer())

        for _ in range(n_epoch):
            total_cost = 0
            for _ in range(n_step):
                # skipgram일 batch 단위로 쪼개어서 가지고 옴
                inputs, labels = voc.next_skipgram(n_batch)
                # 학습
                _, cost = word2vec.train(sess, inputs, labels)
                total_cost += cost

            global_step = sess.run(word2vec.global_step)
            print('Epoch:', '%04d' % global_step, 'cost =', '{:.6f}'.format(total_cost / n_step))

            if global_step % 100 == 0:
                # 100번에 한번씩 저장 함
                saver.save(sess, p_saver + '/dnn.ckpt', global_step=word2vec.global_step)

    print('=== Train Complelte !!! ===')


def test(voc, word2vec):
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(p_saver)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('load model!!')
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('init model!!')
            sess.run(tf.global_variables_initializer())

        matplotlib.rc('font', family='Malgun Gothic')
        for label in voc.lns_voc:
            embedded = word2vec.select(sess, [voc.dic_voc[label]])
            plt.scatter(embedded[0][0], embedded[0][1])
            plt.annotate(label, xy=(embedded[0][0], embedded[0][1]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
        plt.show()

    print('=== Test Complelte !!! ===')


def main(argv):
    # 입력 값 확인 mode 결정 = [train / test]
    f_train = False
    n_epoch = 100
    if 1 < len(argv) and argv[1] == 'train':
        f_train = True
    if 2 < len(argv):
        n_epoch = int(argv[2])
    
    # train 모드에서 입력 테스트 파일 읽어 들임
    lns_word = None
    if f_train and os.path.isfile(p_text):
        with open(p_text, 'r', encoding='UTF8') as f:
            lns_word = []
            for line in f:
                lns_word.append(line.strip())

    # voc 파일 읽어들임 (없으면 skip)
    lns_voc = None
    if os.path.isfile(p_voc):
        print('load voc!!')
        with open(p_voc, 'r', encoding='UTF8') as f:
            lns_voc = []
            for line in f:
                lns_voc.append(line.strip())

    # voc 생성 후 voc 파일이 없으면 파일에 저장
    voc = WordVoc(lns_word, lns_voc)
    if os.path.isfile(p_voc) == False:
        print('make voc!!')
        with open(p_voc, 'w', encoding='UTF8') as f:
            for w in voc.lns_voc:
                f.write(w + '\n')

    word2vec = Word2Vec(voc.len_voc, 2, 128, learning_rate=0.1)

    if f_train:
        # 학습
        train(n_epoch, voc, word2vec)
    else:
        # 시험
        test(voc, word2vec)


if __name__ == "__main__":
    tf.app.run()
