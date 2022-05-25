from pickletools import optimize
import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time

import os

from src.encoder import Encoder
from src.decoder import Decoder
from src.seq2seq import Seq2Seq

from src.utils import display_attention, load_checkpoint, save_checkpoint, \
    translate_sentence, count_parameters, epoch_time, train, evaluate


SEED = 1234
BATCH_SIZE = 12
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
LEARNING_RATE = 0.0005
N_EPOCHS = 0
CLIP = 1


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


spacy_cv = spacy.load('pt_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')


class Transformer_model():

    def __init__(self) -> None:

        self.model = None
        self.optimizer = None
        self.criterion = None

        self.SRC = Field(tokenize=self.tokenize_cv,
                         init_token='<sos>',
                         eos_token='<eos>',
                         lower=True,
                         batch_first=True)

        self.TRG = Field(tokenize=self.tokenize_en,
                         init_token='<sos>',
                         eos_token='<eos>',
                         lower=True,
                         batch_first=True)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.get_dataset_data()
        self.setting_up_train_configurations()

    def get_dataset_data(self) -> None:
        train_data, valid_data, test_data = Multi30k.splits(exts=(".cv", ".en"), fields=(self.SRC, self.TRG),
                                                            test="test", path=".data/criolSet"
                                                            )

        self.SRC.build_vocab(train_data, min_freq=2)
        self.TRG.build_vocab(train_data, min_freq=2)

        self.train_iterator, self.valid_iterator, self.test_iterator = BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=BATCH_SIZE,
            device=self.device
        )

        self.INPUT_DIM = len(self.SRC.vocab)
        self.OUTPUT_DIM = len(self.TRG.vocab)

    def tokenize_cv(self, text):
        """
            Tokenizes Cap-Verdian text from a string into a list of strings
        """
        return [tok.text for tok in spacy_cv.tokenizer(text)]

    def tokenize_en(self, text):
        """
            Tokenizes English text from a string into a list of strings
        """
        return [tok.text for tok in spacy_en.tokenizer(text)]

    def setting_up_train_configurations(self) -> None:
        enc = Encoder(self.INPUT_DIM,
                      HID_DIM,
                      ENC_LAYERS,
                      ENC_HEADS,
                      ENC_PF_DIM,
                      ENC_DROPOUT,
                      self.device)

        dec = Decoder(self.OUTPUT_DIM,
                      HID_DIM,
                      DEC_LAYERS,
                      DEC_HEADS,
                      DEC_PF_DIM,
                      DEC_DROPOUT,
                      self.device)

        source_PAD_IDX = self.SRC.vocab.stoi[self.SRC.pad_token]
        target_PAD_IDX = self.TRG.vocab.stoi[self.TRG.pad_token]

        self.model = Seq2Seq(enc, dec, source_PAD_IDX,
                             target_PAD_IDX, self.device).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE)

        self.criterion = nn.CrossEntropyLoss(ignore_index=target_PAD_IDX)

        try:
            load_checkpoint(torch.load("my_checkpoint.pth.tar"),
                            self.model, self.optimizer)
        except:
            print("No existent checkpoint to load.")

    def count_model_parameters(self) -> None:
        print(
            f'\nThe model has {count_parameters(self.model):,} trainable parameters')

    def initialize_weights(self, m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)

    def train_model(self) -> None:

        best_valid_loss = float('inf')

        for epoch in range(N_EPOCHS):

            start_time = time.time()

            train_loss = train(self.model, self.train_iterator,
                               self.optimizer, self.criterion, CLIP)
            valid_loss = evaluate(
                self.model, self.valid_iterator, self.criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                checkpoint = {
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
                save_checkpoint(checkpoint, "my_checkpoint.pth.tar")

            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(
                f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(
                f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    def evalute_model(self) -> None:
        test_loss = evaluate(self.model, self.test_iterator, self.criterion)

        print(
            f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |'
        )

    def generate_confusion_matrix(self, src) -> None:
        translation, attention = translate_sentence(
            spacy_cv, src, self.SRC, self.TRG, self.model, self.device
        )

        print(f'Source (cv): {src}')
        print(f'Predicted (en): {translation}')

        display_attention(spacy_cv, src, translation, attention)

    def test_model(self, sentences) -> None:
        os.system("clear")
        print("\n                  CV Creole Translator Test ")
        print("-------------------------------------------------------------\n")
        for sentence in sentences:
            translation, _ = translate_sentence(
                spacy_cv, sentence, self.SRC, self.TRG, self.model, self.device
            )
            print(f'  Source (cv): {sentence}')
            print(f'  Predicted (en): {translation}\n')

    def console_model_test(self) -> None:
        os.system("clear")
        print("\n                     CV Creole Translator ")
        print("-------------------------------------------------------------\n")
        while True:
            source = str(input(f'  Source (cv): '))
            translation, _ = translate_sentence(
                spacy_cv, source, self.SRC, self.TRG, self.model, self.device)
            print(f'  Predicted (en): {translation}')


transformer = Transformer_model()
# transformer.test_model(["oi", "oi, manera?", "ondê ke bô ta?"])
transformer.console_model_test()
