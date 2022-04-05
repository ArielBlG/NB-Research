from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


class TestBert:

    def __init__(self, model, X_test, Y_test, batch_size, flag_model=False):
        self.flag_model = flag_model
        self.model = model
        self.X_test = X_test
        self.Y_test = Y_test
        self.batch_size = batch_size

    def __call__(self):
        self._tokenize()
        self._mask()
        self._attention()
        self.convertToTorchandDL()

    def _tokenize(self):
        # tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-v1.1',do_lower_case=True)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        print("X_test shape: {}".format(self.X_test.shape))
        self.input_ids = []
        for sentence in self.X_test:
            encoded_sent = tokenizer.encode(
                sentence,
                add_special_tokens=True,
            )
            self.input_ids.append(encoded_sent)

    def _mask(self, MAX_LEN=350):
        self.input_ids = pad_sequences(self.input_ids,
                                       maxlen=MAX_LEN,
                                       dtype="long",
                                       truncating="post",
                                       padding="post")

    def _attention(self):
        self.mask = tf.math.logical_not(tf.math.equal(self.input_ids, 0))
        self.mask = tf.cast(self.mask, dtype=tf.int64).numpy()

    def convertToTorchandDL(self):
        prediction_inputs = torch.tensor(self.input_ids)
        prediction_masks = torch.tensor(self.mask)
        prediction_labels = torch.tensor(self.Y_test)

        print("Prediction inputs length {}".format(prediction_inputs.shape))
        print("Prediction masks length {}".format(prediction_masks.shape))
        print("Prediction labels length {}".format(prediction_labels.shape))

        prediction_data = TensorDataset(prediction_inputs, prediction_masks,
                                        prediction_labels)
        prediction_sampler = SequentialSampler(prediction_data)
        self.prediction_dataloader = DataLoader(prediction_data,
                                                sampler=prediction_sampler,
                                                batch_size=self.batch_size)

    def predict(self):
        self.model.eval()

        self.predictions, self.true_labels = [], []
        self.bert_proba = np.array([], dtype=np.int64).reshape(0, 8)

        for batch in self.prediction_dataloader:
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)

            # unpack the input from data loader
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = self.model(b_input_ids, token_type_ids=None,
                                     attention_mask=b_input_mask)

                if not self.flag_model:
                    sm = torch.nn.Softmax(dim=1)
                    bert_probabilities = sm(outputs[0]).to('cpu').numpy()
                    logits = outputs[0]
                else:
                    logits = outputs[1]
                    bert_probabilities = outputs[1].to('cpu').numpy()

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                self.predictions.append(logits)
                self.true_labels.append(label_ids)
                self.bert_proba = np.concatenate((self.bert_proba, bert_probabilities))

