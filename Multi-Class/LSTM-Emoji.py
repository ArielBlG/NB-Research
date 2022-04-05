from LSTM_PP import PreProcess
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import pickle
# Verify cude for lstm
from tensorflow.python.client import device_lib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Bidirectional, Embedding, Dense, Dropout, SpatialDropout1D, LSTM, Activation, \
    Flatten
from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.layers.merge  import concatenate
from tensorflow.keras.regularizers import L1L2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Concatenate
from tensorflow import random as rnd_np
from AttentionClass import attentionlayer
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
import random
import os


def test_gpu():
    import tensorflow as tf
    print(device_lib.list_local_devices())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(tf.test.is_gpu_available())


tokenizer = Tokenizer(num_words=70000, lower=False, char_level=False)
pre_process = PreProcess()
emoji_list = ['#question', '#lightbulb-moment', '#real-world-application', '#learning-goal',
              '#important', '#i-think', '#lets-discuss', '#lost', '#just-curious', '#surprised', '#interesting-topic']


def reset_seed():
    os.environ['PYTHONHASHSEED'] = str(2)
    # tensorflow.random.set_seed(2)
    rnd_np.set_seed(2)
    np.random.seed(2)
    random.seed(2)


def prepare_vocabulary(path, df, save_to_pickle=False):
    print(f" df shape is {df.shape}")
    df['len'] = df['text_pp'].apply(lambda x: len(str(x).split(' ')))
    df = df[df['len'] > 1]
    tokenizer.fit_on_texts(list(df['text_pp']))
    if save_to_pickle:
        pickle.dump(tokenizer, open(os.path.join(path, 'tokenizer.pickle'), 'wb'))

    return tokenizer


def fit_text(text, df=None, path=None, tokenizer_from_pickle=False):
    if tokenizer_from_pickle:
        tokenizer = pickle.load(open(path, 'r'))
        sequences = tokenizer.texts_to_sequences(list(df['text_pp']))
    sequences = tokenizer.texts_to_sequences(text)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    text = pad_sequences(sequences,
                         maxlen=200,  # max len of input
                         padding='post',  # padd at the end of sentece
                         truncating='pre'  # truncate the end if larger than 200
                         )
    return text


def build_lstm_model(n_classes, word_index, w2v, max_seq_len=200, embedding_dim=200, dropout=0.3, embed_l2=0):
    """
    Builds a LSTM model with embedding layer and dropout
    :param n_classes:
    :param word_index:
    :param w2v:
    :param max_seq_len:
    :param embedding_dim:
    :param dropout:
    :param embed_l2:
    :return:
    """
    reset_seed()
    embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))
    for word, i in tqdm(word_index.items()):
        try:
            embedding_vector = w2v.wv[word]
        except:
            embedding_vector = np.zeros(embedding_dim)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            if len(embedding_matrix[i]) != len(embedding_vector):
                raise Exception("could not broadcast input array from shape", str(len(embedding_matrix[i])),
                                "into shape", str(len(embedding_vector)), " Please make sure your"
                                                                          " embedding_dim is equal to embedding_vector file")
                exit(1)

            embedding_matrix[i] = embedding_vector

    print("Building the model...")
    model_input = Input(shape=(max_seq_len,))
    # embed_reg = L1L2(l2=embed_l2) if embed_l2 != 0 else None
    model = Sequential()
    print("Building the embeding layer...")
    embed_layer = Embedding(input_dim=len(word_index) + 1,
                            output_dim=embedding_dim,
                            weights=[embedding_matrix],
                            mask_zero=True,
                            # from keras documentation : whether or not the input value 0 is a special "padding" value that should be masked out,If this is True, then all subsequent layers in the model need to support masking or an exception will be raised.
                            input_length=max_seq_len,
                            name='embedding')

    model.add(embed_layer)
    print("Building the LSTM cells...")
    lstm_1 = Bidirectional(LSTM(32, return_sequences=True))
    model.add(lstm_1)
    lstm_2 = Bidirectional(LSTM(32, return_sequences=True))
    model.add(lstm_2)
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(attentionlayer(return_attention=False))

    if n_classes > 2:
        model.add(Dense(n_classes, activation='softmax'))
    else:
        model.add(Dense(n_classes, activation='sigmoid'))
    print("Finished Building the model!")
    print("compiling model...")
    model.compile(
        loss='sparse_categorical_crossentropy',
        # loss = 'binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    print("finished compiling model!")
    print(model.summary())
    return model


def load_w2v_model(path):
    """
    Loads the word2vec data from the path
    :param path:
    :return:
    """
    return Word2Vec.load(path)


def load_data(path_train, path_test, path_labels_train, path_labels_test):
    """
    Loads the data from the given paths
    :param path_train:
    :param path_test:
    :param path_labels_train:
    :param path_labels_test:
    :return:
    """
    df_train = pd.read_csv(path_train, sep='\t', header=None)
    X_train_df = pd.read_csv(path_train)
    X_train = X_train_df['X_train'].astype(str).values
    X_train = fit_text(X_train)

    X_test_df = pd.read_csv(path_test)
    X_test = X_test_df['X_test_lstm'].astype(str).values
    X_test = fit_text(X_test)

    Y_train_df = pd.read_csv(path_labels_train)
    Y_train = Y_train_df['Y_train'].values

    Y_test_df = pd.read_csv(path_labels_test)
    Y_test = Y_test_df['Y_test'].values

    print("Number of train sentences: {}".format(len(X_train)))
    print("Number of test sentences: {}".format(len(X_test)))
    unique, counts = np.unique(Y_test, return_counts=True)
    labels_dict = dict(zip([i for i in range(11)], emoji_list))
    print(f"Labels dict mapping: {labels_dict}")
    print(f"Number of records for each label: {dict(zip(unique, counts))}")

    return X_train, X_test, Y_train, Y_test


def train(model, tokenizer, w2v, X_train, Y_train):
    """
    Trains the model
    :param model:
    :return:
    """
    model = build_lstm_model(11, tokenizer.word_index, w2v, embedding_dim=200)
    return model


def fit(model, X_train, Y_train):
    x_train, x_validation, y_train, y_validation = train_test_split(X_train,
                                                                    Y_train,
                                                                    stratify=Y_train,
                                                                    random_state=42,
                                                                    test_size=0.10)
    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=1e-4,
                                   patience=2,
                                   verbose=1, mode='auto')
    history = model.fit(x_train, y_train,
                        validation_data=(x_validation, y_validation),
                        epochs=5,
                        batch_size=64,
                        verbose=1,
                        callbacks=[early_stopping])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()


def test(X_test, Y_test, model):
    """
    Tests the model
    :param X_test:
    :param Y_test:
    :param model:
    :return:
    """
    predictions = model.predict(X_test)
    flat_predictions = np.argmax(predictions, axis=1).flatten()
    flat_true_labels = Y_test
    return flat_predictions, flat_true_labels


def evaluation(flat_predictions, flat_true_labels):
    print('Accuracy: %.3f' % accuracy_score(flat_true_labels, flat_predictions))
    # print('MCC: %.3f' % mcc)
    # print("ROC-AUC: %.3f" % roc_auc_score(Y_test,bert_proba[:,1]))
    # average_precision = average_precision_score(Y_test,bert_proba[:,1])
    # print("PR-AUC: %.3f" % average_precision)
    pre_bert = precision_score(flat_true_labels, flat_predictions, average="macro")
    print("Precision: %.3f" % pre_bert)
    recall_bert = recall_score(flat_true_labels, flat_predictions, average="macro")
    print("Recall: %.3f" % recall_bert)
    f1_bert = f1_score(flat_true_labels, flat_predictions, average='weighted')
    print("f1: %.3f" % f1_bert)
    # f1_normal = f1_score(flat_true_labels, flat_predictions)
    # print(f1_normal)
    cf = confusion_matrix(flat_true_labels, flat_predictions)
    print(confusion_matrix(flat_true_labels, flat_predictions))

    df_results = pd.DataFrame(columns=['ACC', 'ROC-AUC', 'PR-AUC', 'PRECISION', 'RECALL', 'F-1'])
    df_results['ACC']['Bert'] = accuracy_score(flat_true_labels, flat_predictions)
    # df_results['ROC-AUC']['Bert']=roc_auc_score(Y_test,bert_proba[:,1])
    # df_results['PR-AUC']['Bert']=average_precision
    df_results['PRECISION']['Bert'] = pre_bert
    # experiment.log_metrics('Precision', pre_bert)
    df_results['RECALL']['Bert'] = recall_bert
    # experiment.log_metrics('Recall', recall_bert)
    df_results['F-1']['Bert'] = f1_bert
    # experiment.log_metrics("F1", f1_bert)
    # experiment.log_confusion_matrix(cf)
    # experiment.log_confusion_matrix(flat_true_labels, flat_predictions)
    # experiment.end()
    print(print(classification_report(flat_true_labels, flat_predictions, digits=3, target_names=emoji_list)))


def main():
    """
    Main function
    :return:
    """
    # test GPU
    test_gpu()
    base_path = "/home/arielblo/Datasets/train_test_ds/"
    w2v_path = "/home/arielblo/Datasets/word2vec_200.model"
    # Prepare vocabulary
    df = pd.read_csv("/home/arielblo/Datasets/all_datasets_february_mc.csv")
    tokenizer = prepare_vocabulary(df=df, path=base_path)

    # Load data
    X_train, X_test, Y_train, Y_test = load_data(path_train=base_path + "lstm_train_sentences.csv",
                                                 path_test=base_path + "lstm_test_sentences.csv",
                                                 path_labels_train=base_path + "train_labels.csv",
                                                 path_labels_test=base_path + "test_labels.csv")

    # Load word2vec
    w2v = load_w2v_model()

    model = build_lstm_model(11, tokenizer.word_index, w2v, embedding_dim=200)

    # Train model
    model = train(model, tokenizer, w2v, X_train, Y_train)

    # Fit model
    fit(model, X_train, Y_train)

    # Test model
    flat_predictions, flat_true_labels = test(X_test, Y_test, model)

    # Evaluation
    evaluation(flat_predictions, flat_true_labels)

