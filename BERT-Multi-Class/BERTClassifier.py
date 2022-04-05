import torch
import torch.nn as nn
from transformers import BertModel


class BERTClassifier(torch.nn.Module):

    def __init__(self, n_classes, class_weights):
        super(BERTClassifier, self).__init__()
        print(class_weights)
        self.n_classes = n_classes
        self.class_weights = torch.tensor(class_weights).float()
        self.bert = BertModel.from_pretrained(
            '/home/arielblo/Models/BetterModel',
            # 'dmis-lab/biobert-v1.1',
            # 'bert-base-uncased',
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False)
        self.pre_classifier = nn.Linear(768, 768)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.dropout = nn.Dropout(p=0.3)
        # self.activation = nn.Sigmoid()
        # self.loss_fn = nn.BCELoss()
        self.relu = nn.ReLU()
        self.activation = nn.Softmax(dim=1)
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights, reduction='mean')

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
        # The embedding vectors are the the embedding vectors of all the tokens in the sequence -> irrelavent for us
        # The pooled output is contains the embedding vector of [CLS] token
        # embedding_vectors, pooled_output = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        # dropout_output = self.dropout(pooled_output)
        # in case follow not working try -> https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f

        # BERT prepends a [CLS] token (short for “classification”) to the start of each sentence
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=None)

        dropout_output = self.dropout(output[1])
        logits = self.classifier(dropout_output)
        final_layer = self.relu(logits)

        loss = None
        if labels is not None:
            loss = self.loss_fn(final_layer.view(-1, self.n_classes), labels.view(-1))

        return loss, final_layer