import numpy as np
import pandas as pd
import re
import random
import statistics
import torch
import tensorflow as tf
import os
import plotly.express as px
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, \
    accuracy_score, precision_score, f1_score, confusion_matrix, classification_report

from BertTrainer import BertTrainer
from BERTClassifier import BERTClassifier
from PreProcessClass import PreProcessClass
from BertTest import TestBert

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

emoji_list = ['#question', '#lightbulb-moment', '#real-world-application', '#learning-goal',
              '#important', '#i-think', '#lets-discuss', '#lost', '#just-curious', '#surprised', '#interesting-topic']


def gpu_config():
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # Get the GPU device name.

    # If there's a GPU available...
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")


def load_data(path_train, path_test, path_labels_train, path_labels_test):
    """
    Loads the data from the given paths
    :param path_train:
    :param path_test:
    :param path_labels_train:
    :param path_labels_test:
    :return:
    """
    X_train_df = pd.read_csv(path_train)
    X_train = X_train_df['X_train'].values
    X_test_df = pd.read_csv(path_test)
    X_test = X_test_df['X_test'].values
    Y_train_df = pd.read_csv(path_labels_train)
    Y_train = Y_train_df['Y_train'].values
    Y_test_df = pd.read_csv(path_labels_test)
    Y_test = Y_test_df['Y_test'].values
    return X_train, X_test, Y_train, Y_test


def change_labels_to_clusters(y):
    y_copy = y.copy()
    first_cluster_train = (y_copy == 0) | (y_copy == 8)
    second_cluster_train = (y_copy == 5) | (y_copy == 6)
    third_cluster_train = (y_copy == 4) | (y_copy == 3)  # important & learning-goal
    light_bulb_train = (y_copy == 1)

    # third_cluster_train = (y_copy == 4) | (y_copy == 1) # important & lightbulb moment
    # light_bulb_train = (y_copy == 3)

    real_world_train = (y_copy == 2)
    lost_train = (y_copy == 7)
    surprised_train = (y_copy == 9)
    interesting_train = (y_copy == 10)

    cond_list = [first_cluster_train, second_cluster_train, third_cluster_train, lost_train, light_bulb_train,
                 real_world_train,
                 surprised_train, interesting_train]
    choise_list = list(range(8))
    return np.select(cond_list, choise_list, y_copy)


def evaluate(predictions, bert_proba, true_labels, Y_test):
    # Combine the predictions for each batch into a single list of 0s and 1s.
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    # Combine the correct labels for each batch into a single list.
    flat_true_labels = [item for sublist in true_labels for item in sublist]
    # Calculate the MCC
    # mcc = matthews_corrcoef(flat_true_labels, flat_predictions)

    print("Fine Tuned Bert Results -> Additional Pre-Tain BERT use BertForSequenceClassification")
    print("***********************")
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
    print(confusion_matrix(flat_true_labels, flat_predictions))

    df_results = pd.DataFrame(columns=['ACC', 'ROC-AUC', 'PR-AUC', 'PRECISION', 'RECALL', 'F-1'])
    df_results['ACC']['Bert'] = accuracy_score(flat_true_labels, flat_predictions)
    # df_results['ROC-AUC']['Bert']=roc_auc_score(Y_test,bert_proba[:,1])
    # df_results['PR-AUC']['Bert']=average_precision
    df_results['PRECISION']['Bert'] = pre_bert
    df_results['RECALL']['Bert'] = recall_bert
    df_results['F-1']['Bert'] = f1_bert

    cluster_list = ['first_cluster', 'second_cluster', 'third_cluster', '#lost', '#lightbulb-moment',
                    '#real-world-application', '#surprised', '#interesting-topic']
    print(classification_report(flat_true_labels, flat_predictions, digits=3, target_names=cluster_list))


def main():
    gpu_config()
    base_path = "/home/arielblo/Datasets/train_test_ds/"
    X_train, X_test, Y_train, Y_test = load_data(path_train=base_path + "bert_train_sentences.csv",
                                                 path_test=base_path + "bert_test_sentences.csv",
                                                 path_labels_train=base_path + "train_labels.csv",
                                                 path_labels_test=base_path + "test_labels.csv")
    pre_class = PreProcessClass(emoji_list, batch_size=16, X_train=X_train, X_test=X_test, Y_train=Y_train,
                                Y_test=Y_test)
    pre_class()

    self_model = False  # True if you want to train the model yourself
    if self_model:
        model = BERTClassifier(11, class_weights=pre_class.class_weights)

    trainer = BertTrainer(
        lr=2e-5,  # 2e-6,
        eps=1e-8,
        epochs=3,  # 12,
        # model= model, # uncomment if you want to use your own model
        train_data_loader=pre_class.train_data_loader,
        validation_data_loader=pre_class.validation_data_loader,
        num_labels=np.unique(Y_train).shape[0]
    )
    trainer()

    trainer.train()
    trainer.print_model_graph()

    torch.save(trainer.model, '/home/arielblo/Datasets/models/emoji_mc_model/mc_model')

    test = TestBert(trainer.model,
                    X_test=pre_class.X_test_records,
                    Y_test=pre_class.Y_test_records,
                    batch_size=pre_class.batch_size,
                    flag_model=trainer.flag_model)
    test()
    test.predict()
    predictions = test.predictions
    bert_proba = test.bert_proba
    true_labels = test.true_labels
    Y_test = test.Y_test
    evaluate(predictions, bert_proba, true_labels, Y_test)
