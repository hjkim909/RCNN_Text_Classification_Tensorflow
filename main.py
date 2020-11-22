import os
import argparse
import logging
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

from build_vocab import build_dictionary
from dataset import CustomTextDataset, collate_fn
from model import RCNN
from train import train
from utils import read_file

logging.basicConfig(format='%(asctime)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)


def main(args):
    model = RCNN(vocab_size=args.vocab_size,
                 embedding_dim=args.embedding_dim,
                 hidden_size=args.hidden_size,
                 hidden_size_linear=args.hidden_size_linear,
                 class_num=args.class_num,
                 dropout=args.dropout)

    train_texts, train_labels = read_file(args.train_file_path)
    test_texts, test_labels = read_file(args.test_file_path)
    word2idx = build_dictionary(train_texts, vocab_size=args.vocab_size)
    logger.info('Dictionary Finished!')

    x_train, y_train = CustomTextDataset(train_texts, train_labels, word2idx)
    x_test, y_test = CustomTextDataset(test_texts, test_labels, word2idx)
    num_train_data = len(x_train)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    hist = train(model, optimizer, x_train, x_test, y_train, y_test, args)
    logger.info('******************** Train Finished ********************')

    tf.saved_model.save(model, "/tmp/module_no_signatures")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_set', action='store_true', default=False)

    # data
    parser.add_argument("--train_file_path", type=str, default="./data/train.csv")
    parser.add_argument("--test_file_path", type=str, default="./data/test.csv")
    parser.add_argument("--model_save_path", type=str, default="./model_saved")
    parser.add_argument("--num_val_data", type=int, default=10000)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)

    # model
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--hidden_size_linear", type=int, default=512)
    parser.add_argument("--class_num", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)

    # training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    set_seed(args)

    main(args)