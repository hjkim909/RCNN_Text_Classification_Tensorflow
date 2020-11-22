import tensorflow as tf


def CustomTextDataset(texts, labels, dictionary, train_ratio = 0.8):
    x = [[dictionary.get(token, 1) for token in token_list] for token_list in texts]
    x = tf.keras.preprocessing.sequence.pad_sequences(x,maxlen = 64 ,padding="post")
    y = labels
    y = tf.one_hot(y, 4)
    return x, y

def collate_fn(data, args, pad_idx=0):
    """Padding"""
    texts, labels = zip(*data)
    texts = [s + [pad_idx] * (args.max_len - len(s)) if len(s) < args.max_len else s[:args.max_len] for s in texts]
    return tf.convert_to_tensor(texts), tf.convert_to_tensor(labels)
    # return torch.LongTensor(texts), torch.LongTensor(labels)