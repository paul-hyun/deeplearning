import numpy as np
import re

class WordVoc(object):
    """
    파일로 부터 전체 문장을 읽어 들여서 Vocabulary를 구성하는 기능을 수행한다.
    단, 이미 구성된 Vocabulary가 있는 경우는 해당 Vocabulary를 사용 한다.
    """

    _PAD_ = "_PAD_"  # 빈칸
    _STA_ = "_STA_"  # 시작
    _EOS_ = "_EOS_"  # 종료
    _UNK_ = "_UNK_"  # Dictionary에 없는 단어

    _PAD_ID_ = 0
    _STA_ID_ = 1
    _EOS_ID_ = 2
    _UNK_ID_ = 3
    _PRE_DEFINED_ = [_PAD_ID_, _STA_ID_, _EOS_ID_, _UNK_ID_]

    def __init__(self, lns_word, lns_voc=None):
        """
        설명: 초기 생성 함수
        Parameters:
            - lns_word: 입력 문자열을 라인단위 배열 행태로 입력
            - lns_voc: Vocabulary 문자열을 라인단위 배열 형태로 입력 (단, None일 경우는 새로 작성 함)
        """
        if lns_voc == None: # 입력 Vocabulary가 없을 경우 입력 문자열을 기준으로 다시 생성 함
            words = []
            for line in lns_word:
                tokens = WordVoc.tokenizer(line)
                words.extend([w for w in tokens if w])
            lns_voc = list(set(words))

        self.lns_voc = lns_voc
        self.tot_voc = self._PRE_DEFINED_.copy()
        self.tot_voc.extend(self.lns_voc)
        self.dic_voc = {n: i for i, n in enumerate(self.tot_voc)}
        self.len_voc = len(self.dic_voc)

        self.idxs_word = []
        if lns_word is not None:
            for line in lns_word:
                self.idxs_word.append(self.build_idx(line))
        self.len_word = len(self.idxs_word)

        self.batch_idx = 0
    
    def build_idx(self, text):
        """
        설명: 문자열을 Vocabulary index 형태로 변환 한다.
        Parameters:
            - text: 입력 문자열
        Return: 문자열이 변환된 Vocabulary index 배열
        """
        tokens = WordVoc.tokenizer(text)
        lind_idx = []
        for token in tokens:
            if token in self.dic_voc:
                lind_idx.append(self.dic_voc[token])
            else:
                lind_idx.append(self._UNK_ID_)
        return lind_idx

    def next_seq2seq(self, n_batch):
        """
        설명: 전체 문자열 중에 n_batch 만큼의 개수를 seq2seq 형태로 가지고 온다.
              단, 마지막을 지난 경우는 처음부터 다시 가지고 온다.
        Parameters:
            - n_batch: batch 크기
        Return: 다음 seq2seq 배치 데이터
        """
        enc_input = []
        dec_input = []
        target = []

        start = self.batch_idx

        if self.batch_idx + n_batch < self.len_word - 1:
            self.batch_idx = self.batch_idx + n_batch
        else:
            self.batch_idx = 0

        batch_set = self.idxs_word[start:start + n_batch]

        max_len_input = 0
        max_len_output = 0
        for i in range(0, len(batch_set), 2):
            max_len_input = max(max_len_input, len(batch_set[i]))
            max_len_output = max(max_len_output, len(batch_set[i + 1]))
        max_len_output += 1

        for i in range(0, len(batch_set), 2):
            enc, dec, tar = self.seq2seq_form(batch_set[i], batch_set[i + 1], max_len_input, max_len_output)

            enc_input.append(enc)
            dec_input.append(dec)
            target.append(tar)

        return enc_input, dec_input, target

    def seq2seq_form(self, input, output, input_max, output_max):
        """
        설명: 문자열을 seq2seq 형태에 맞도록 데이터를 변환한다.
        Parameters:
            - input: 입력 문자열
            - output: 출력 문자열
            - input_max: input 길이
            - output_max: 출력 길이
        Return: seq2seq 배치 데이터
        """
        enc_input = (input + ([self._PAD_ID_] * (input_max - len(input)))) if len(input) < input_max else input
        dec_input = ([self._STA_ID_] + output + ([self._PAD_ID_] * (output_max - len(output) - 1))) if len(output) + 1 < output_max else ([self._STA_ID_] + output)
        target = (output + [self._EOS_ID_] + ([self._PAD_ID_] * (output_max - len(output) - 1))) if len(output) + 1 < output_max else (output + [self._EOS_ID_])

        enc_input.reverse()

        enc_input = np.eye(self.len_voc)[enc_input]
        dec_input = np.eye(self.len_voc)[dec_input]

        return enc_input, dec_input, target
    
    def build_skipgram(self, n_window):
        """
        설명: skipgram 데이터를 만든다.
        Parameters:
            - n_window: Window 사이즈
        """
        all_word = []
        for line in self.idxs_word:
            all_word.extend(line)
        n_all = len(all_word)

        self.idxs_sgram = []
        for i in range(1, n_all):
            word = all_word[i]
            nears = all_word[max([i - n_window, 0]):min([i + n_window, n_all])]
            for near in nears:
                if word != near:
                    self.idxs_sgram.append([word, near])
        self.len_sgram = len(self.idxs_sgram)

    def next_skipgram(self, n_batch):
        """
        설명: 전체 문자열 중에 n_batch 만큼의 개수를 skipgram 형태로 가지고 온다.
              단, 마지막을 지난 경우는 처음부터 다시 가지고 온다.
        Parameters:
            - n_batch: batch 크기
        Return: 다음 seq2seq 배치 데이터
        """
        inputs = []
        labels = []

        start = self.batch_idx

        if self.batch_idx + n_batch < self.len_sgram - 1:
            self.batch_idx = self.batch_idx + n_batch
        else:
            self.batch_idx = 0

        batch_set = self.idxs_sgram[start:start + n_batch]

        for batch in batch_set:
            inputs.append(batch[0])
            labels.append([batch[1]])

        return inputs, labels

    def translate(self, encoded):
        """
        설명: Vocabulary index를 문자열 형태로 변환한다. _EOS_가 있을 경우는 거지까지 문자열을 자른다.
        Parameters:
            - encoded: Vocabulary index
        Return: 일반 문자열
        """
        decoded = [self.tot_voc[i] for i in encoded]

        try:
            end = decoded.index(self._EOS_ID_)
            decoded = ''.join(decoded[:end])
        except ValueError:
            pass

        return decoded

    @staticmethod
    def tokenizer(text: str) -> [str]:
        """
        설명: 문자열을 공백 및 특수문자 단위로 쪼개어 분할 한다.
        Parameters:
            - text: 입력 문자열
        """
        tokens = []
        regx = re.compile("([.,!?\"':;)(])")

        for token in text.split():
            tokens.extend(regx.split(token))

        return [w for w in tokens if w]

