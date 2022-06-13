import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from torchtext.data.metrics import bleu_score

from pyter import ter

import spacy

import numpy as np

from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.translate.meteor_score import meteor_score

import random
import math
import time

import os

from src.encoder import Encoder
from src.decoder import Decoder
from src.seq2seq import Seq2Seq
from src.gammar_checker import Grammar_checker
from src.utils import bleu, display_attention, load_checkpoint, save_checkpoint, \
    translate_sentence, count_parameters, epoch_time, train, evaluate, count_parameters,\
    bleu, meteor_score


SEED = 1234
BATCH_SIZE = 10
HID_DIM = 256
ENC_LAYERS = 4
DEC_LAYERS = 4
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
LEARNING_RATE = 3e-4
N_EPOCHS = 100
CLIP = 1


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


spacy_cv = spacy.load('pt_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')


class Transformer_Translator():

    def __init__(self) -> None:

        self.model = None
        self.optimizer = None
        self.criterion = None

        self.grammar = Grammar_checker()
        self.special_tokens = ['<sos>', '<eos>', '<pad>', '<unk>']
        self.writer = SummaryWriter()

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
        self.train_data, self.valid_data, self.test_data = Multi30k.splits(
            exts=(".cv", ".en"), fields=(self.SRC, self.TRG),
            test="test", path=".data/criolSet"
        )

        self.SRC.build_vocab(self.train_data, min_freq=2)
        self.TRG.build_vocab(self.train_data, min_freq=2)

        self.train_iterator, self.valid_iterator, self.test_iterator = BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data),
            batch_size=BATCH_SIZE,
            device=self.device
        )

        self.INPUT_DIM = len(self.SRC.vocab)
        self.OUTPUT_DIM = len(self.TRG.vocab)

    def tokenize_cv(self, text: str):
        """
            Tokenizes Cap-Verdian text from a string into a list of strings
        """
        return [tok.text for tok in spacy_cv.tokenizer(text)]

    def tokenize_en(self, text: str):
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
            load_checkpoint(torch.load("checkpoints/my_checkpoint.pth.tar"),
                            self.model, self.optimizer)
        except:
            print("No existent checkpoint to load.")

    def count_model_parameters(self) -> None:
        print(
            f'\nThe model has {count_parameters(self.model):,} trainable parameters')

    def initialize_weights(self, m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)

    def show_train_metrics(self, epoch: int, epoch_time: str, train_loss: float, 
            train_accuracy: float, valid_loss: float, valid_accuracy:float) -> None:

        print(f' Epoch: {epoch+1:03}/{N_EPOCHS} | Time: {epoch_time}')
        print(
            f' Train Loss: {train_loss:.3f} | Train Acc: {train_accuracy:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(
            f' Val. Loss: {valid_loss:.3f} | Val Acc: {valid_accuracy:.3f} | Val. PPL: {math.exp(valid_loss):7.3f}')
    
    def save_train_metrics(self, epoch: int, train_loss: float, 
            train_accuracy: float, valid_loss: float, valid_accuracy:float) -> None:
        """
            Save the training metrics to be ploted in the tensorboard.
        """
        # All stand alone metrics
        self.writer.add_scalar(
            "Training Loss", train_loss, global_step=epoch)
        self.writer.add_scalar(
            "Training Accuracy", train_accuracy, global_step=epoch)
        self.writer.add_scalar(
            "Validation Loss", valid_loss, global_step=epoch)
        self.writer.add_scalar(
            "Validation Accuracy", valid_accuracy, global_step=epoch)
        
        # Mixing Train Metrics
        self.writer.add_scalars(
            "Training Metrics (Train Loss / Train Accurary)", {
                "Train Loss": train_loss, "Train Accurary": train_accuracy},
            global_step=epoch
        )

        # Mixing Validation Metrics
        self.writer.add_scalars(
            "Training Metrics (Validation Loss / Validation Accurary)", {
                "Validation Loss": valid_loss, "Validation Accuracy": valid_accuracy},
            global_step=epoch
        )
        
        # Mixing Train and Validation Metrics
        self.writer.add_scalars(
            "Training Metrics (Train Loss / Validation Loss)", {
                "Train Loss": train_loss, "Validation Loss": valid_loss},
            global_step=epoch
        )
        self.writer.add_scalars(
            "Training Metrics (Train Accurary / Validation Accuracy)", {
                "Train Accurary": train_accuracy, "Validation Accuracy": valid_accuracy},
            global_step=epoch
        )

    def train_model(self) -> None:

        best_valid_loss = float('inf')

        for epoch in range(N_EPOCHS):

            start_time = time.time()

            train_loss, train_accuracy = train(self.model, self.train_iterator,
                                               self.optimizer, self.criterion, CLIP)
            valid_loss, valid_accuracy = evaluate(
                self.model, self.valid_iterator, self.criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                checkpoint = {
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
                save_checkpoint(checkpoint, "checkpoints/my_checkpoint.pth.tar")
            self.show_train_metrics(
                epoch, f"{epoch_mins}m {epoch_secs}s", train_loss,
                train_accuracy, valid_loss, valid_accuracy
            )
            self.save_train_metrics(
                epoch, train_loss,
                train_accuracy, valid_loss, valid_accuracy
            )

    def evalute_model(self) -> None:
        test_loss = evaluate(self.model, self.test_iterator, self.criterion)

        print(
            f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |'
        )

    def generate_confusion_matrix(self, src: str) -> None:
        translation, attention = translate_sentence(
            spacy_cv, src, self.SRC, self.TRG, self.model, self.device
        )

        print(f'Source (cv): {src}')
        print(f'Predicted (en): {translation}')

        display_attention(spacy_cv, src, translation, attention)

    def test_model(self, test_data) -> None:
        test_data = self.get_test_data()
        os.system("clear")
        print("\n                  CV Creole Translator Test ")
        print("-------------------------------------------------------------\n")
        for data_tuple in test_data:
            src, trg = " ".join(
                data_tuple[0]), self.untokenize_sentence(data_tuple[1])
            translation, _ = translate_sentence(
                spacy_cv, src, self.SRC, self.TRG, self.model, self.device
            )
            print(f'  Source (cv): {src}')
            print(f'  Target (en): {trg}')
            print(
                f'  Predicted (en): {self.untokenize_sentence(translation)}\n')

    def console_model_test(self) -> None:
        os.system("clear")
        print("\n                     CV Creole Translator ")
        print("-------------------------------------------------------------\n")
        while True:
            source = str(input(f'  Source (cv): '))
            translation, _ = translate_sentence(
                spacy_cv, source, self.SRC, self.TRG, self.model, self.device)

            print(
                f'  Predicted (en): {self.untokenize_sentence(translation)}\n')

    def get_translation(self, sentence: str) -> str:
        translation, _ = translate_sentence(
            spacy_cv, sentence, self.SRC, self.TRG, self.model, self.device)

        return self.untokenize_sentence(translation)

    def untokenize_sentence(self, tokens: list) -> str:
        """
            Method to untokenize the pedicted translation.
            Returning it on as an str, with some grammar checks.
        """
        tokens = [token for token in tokens if token not in self.special_tokens]
        translated_sentence = TreebankWordDetokenizer().detokenize(tokens)
        return self.grammar.check_sentence(translated_sentence)

    def get_test_data(self) -> list:
        return [(test.src, test.trg) for test in self.test_data.examples[0:20]]

    def calculate_blue_score(self):
        """
            BLEU (bilingual evaluation understudy) is an algorithm for evaluating 
            the quality of text which has been machine-translated from one natural 
            language to another.
        """
        targets = []
        outputs = []

        for example in self.test_data:
            src = vars(example)["src"]
            trg = vars(example)["trg"]
            predictions = []

            for _ in range(3):
                prediction, _ = translate_sentence(
                    spacy_cv, src, self.SRC, self.TRG, self.model, self.device)
                predictions.append(prediction[:-1])

            print(f'  Source (cv): {" ".join(src)}')
            print(f'  Target (en): {trg}')
            print(f'  Predictions (en):')
            [print(f'      - {prediction}') for prediction in predictions]
            print("\n")

            targets.append(trg)
            outputs.append(predictions)

        score = bleu_score(targets, outputs)
        print(f"Bleu score: {score * 100:.2f}")

    def calculate_meteor_score(self):
        """
            METEOR (Metric for Evaluation of Translation with Explicit ORdering) is 
            a metric for the evaluation of machine translation output. The metric is 
            based on the harmonic mean of unigram precision and recall, with recall 
            weighted higher than precision.
        """
        all_meteor_scores = []

        for example in self.test_data:
            src = vars(example)["src"]
            trg = vars(example)["trg"]
            predictions = []

            for _ in range(4):
                prediction, _ = translate_sentence(
                    spacy_cv, src, self.SRC, self.TRG, self.model, self.device)
                predictions.append(self.untokenize_sentence(prediction))

            all_meteor_scores.append(meteor_score(
                predictions, self.untokenize_sentence(trg)
            ))
            print(f'  Source (cv): {" ".join(src)}')
            print(f'  Target (en): {self.untokenize_sentence(trg)}')
            print(f'  Predictions (en): ')
            [print(f'      - {prediction}') for prediction in predictions]
            print("\n")

        score = sum(all_meteor_scores)/len(all_meteor_scores)
        print(f"Meteor score: {score * 100:.2f}")

    def calculate_ter(self):
        """
            TER. Translation Error Rate (TER) is a character-based automatic metric for 
            measuring the number of edit operations needed to transform the 
            machine-translated output into a human translated reference.
        """
        all_translation_ter = 0

        for example in self.test_data:
            src = vars(example)["src"]
            trg = vars(example)["trg"]

            prediction, _ = translate_sentence(
                spacy_cv, src, self.SRC, self.TRG, self.model, self.device)

            print(f'  Source (cv): {" ".join(src)}')
            print(f'  Target (en): {" ".join(trg)}')
            print(f'  Predictions (en): {" ".join(prediction)}\n')

            all_translation_ter += ter(prediction, trg)
        print(f"Bleu score: {all_translation_ter/len(self.test_data) * 100:.2f}")

    def count_hyperparameters(self) -> None:
        print(
            f'\nThe model has {count_parameters(self.model):,} trainable parameters')
