from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.utils import class_weight
from collections import Counter


class PreProcessClass:

    def __init__(self, emoji_list, batch_size, df=None, test_size=0.15, X_train=None, X_test=None, Y_train=None,
                 Y_test=None):
        if X_train is None or X_test is None or Y_train is None or Y_test is None:
            """Handle the pre splited train and test"""
            print("BUG")
            self.recevied_df_flag = False
            self.df = self.preprocess_df(df)
        else:
            self.recevied_df_flag = True
            self.X_test_records = X_test
            self.Y_test_records = Y_test
            self.X_train_records = X_train
            self.Y_train_records = Y_train

        self.emoji_list = emoji_list
        self.batch_size = batch_size
        self.sentences = None
        self.labels = None
        self.seed = 42
        # experiment.log_parameters("seed", self.seed)
        self.test_size = test_size
        self.class_weights = None

    def __call__(self):
        if not self.recevied_df_flag:
            self.tag_df()
            X_train, X_test, Y_train, Y_test = self.split_train_test()
            self.X_test_records = X_test
            self.Y_test_records = Y_test
            self.X_train_records = X_train
            self.Y_train_records = Y_train
        else:
            self.class_weights = class_weight.compute_class_weight(
                class_weight="balanced",
                classes=np.unique(self.Y_train_records),
                y=self.Y_train_records
            )

        self._tokenize(self.X_train_records)
        self._pad_sent()
        self._mask_sent()
        self.splitTrainAndValidation(Y_train=self.Y_train_records)

    def preprocess_df(self, df):
        """
        Remove unnessecairy strings from dataset
        """
        df.loc[:, 'text_without_emoji'] = df['text_without_emoji'].apply(lambda x: re.sub('#|# ', '', str(x)))
        df.loc[:, 'text_without_emoji'] = df['text_without_emoji'].apply(
            lambda x: re.sub('<!--td {border: 1px solid #ccc;}br {mso-data-placement:same-cell;}-->', '', str(x)))
        print("Total number of records DF : {}".format(df.shape[0]))
        return df

    def tag_df(self):
        """
        transform labels of emojis that are bigger than 1 to 1
        """
        self.sentences = self.df['text_without_emoji'].values
        # mask = self.df[self.emoji_list] > 0
        # self.labels = np.select([mask], [1], 0)
        # self.labels = self.df['#question'].values
        # self.df['label'] = 0
        # self.df['label'] = self.df['label'].mask(self.df['#question'] == 1, 1)
        # self.df['label'] = self.df['label'].mask(self.df['#important'] == 1, 2)
        # print(self.df['label'].unique())
        # print(type(self.df['label']))
        self.labels = self.df['label'].values
        print(self.df['label'].unique())

    def split_train_test(self):
        """
        Function splits the data to train and test
        """
        X_train, X_test, Y_train, Y_test = train_test_split(self.sentences,
                                                            self.labels,
                                                            stratify=self.labels,
                                                            random_state=self.seed,
                                                            test_size=self.test_size)
        print("Number of train sentences: {}".format(len(X_train)))
        print("Number of test sentences: {}".format(len(X_test)))
        unique, counts = np.unique(Y_test, return_counts=True)
        labels_dict = dict(zip([i for i in range(11)], self.emoji_list))
        print(f"Labels dict mapping: {labels_dict}")
        print(f"Number of records for each label: {dict(zip(unique, counts))}")
        # itemCt = Counter(Y_test)
        # maxCt = float(max(Y_test.values()))
        # self.class_weights = {clsID : maxCt/numImg for clsID, numImg in itemCt.items()}
        self.class_weights = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.unique(Y_train),
            y=Y_train
        )

        print(f"Class weights: {dict(zip(np.unique(Y_train), self.class_weights))}")
        return X_train, X_test, Y_train, Y_test

    def _tokenize(self, X_train):
        """
        Tokenize the sentences
        :param X_train:
        :return:
        """
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        # tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-v1.1', do_lower_case=True)
        self.input_ids = []

        for index, sentence in enumerate(X_train):
            # `encode` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #    using encode and not encode_plus since we are feeding one sentence only
            encoded_sent = tokenizer.encode(
                sentence,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                # This function also supports truncation and conversion to pytorch tensors
                # return_tensors = 'pt',     # Return pytorch tensors.
            )
            self.input_ids.append(encoded_sent)
        print("length of input ids: {}".format(len(self.input_ids)))
        print("Original first text: {}".format(X_train[0]))
        print("Token first text: {}".format(self.input_ids[0]))

    def _pad_sent(self, max_padding=350):
        """
          # Pad our input tokens with value 0.
          # "post" indicates that we want to pad and truncate at
          # the end of the sequence, as opposed to the beginning.
        """
        self.input_ids = pad_sequences(self.input_ids,
                                       maxlen=max_padding,
                                       dtype="long",
                                       value=0,
                                       truncating="post",
                                       padding="post")

        # sanity check
        print("Mean sentence size tokenized: {}".format(
            sum([len(sent) for sent in self.input_ids]) / len(self.input_ids)))

    def _mask_sent(self, Y_train=None):
        """
            # Create the attention mask.
            #   - If a token ID is 0, then it's padding, set the mask to 0.
            #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        """
        self.mask = tf.math.logical_not(tf.math.equal(self.input_ids, 0))
        self.mask = tf.cast(self.mask, dtype=tf.int64).numpy()

    def splitTrainAndValidation(self, Y_train=None):

        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(self.input_ids,
                                                                                            Y_train,
                                                                                            stratify=Y_train,
                                                                                            random_state=self.seed,
                                                                                            test_size=self.test_size)

        # do the same for the attention masks
        train_masks, validation_masks, _, _ = train_test_split(self.mask,
                                                               Y_train,
                                                               stratify=Y_train,
                                                               random_state=self.seed,
                                                               test_size=self.test_size)

        print("Number of train sentences: {}".format(len(train_inputs)))
        print("Number of validation sentences: {}".format(len(validation_inputs)))

        self.convertToTorchAndDL(train_inputs,
                                 validation_inputs,
                                 train_labels,
                                 validation_labels,
                                 train_masks,
                                 validation_masks)

    def convertToTorchAndDL(self,
                            train_inputs,
                            validation_inputs,
                            train_labels,
                            validation_labels,
                            train_masks,
                            validation_masks):

        train_inputs = torch.tensor(train_inputs)
        validation_inputs = torch.tensor(validation_inputs)

        train_labels = torch.tensor(train_labels)
        validation_labels = torch.tensor(validation_labels)

        train_masks = torch.tensor(train_masks)
        validation_masks = torch.tensor(validation_masks)

        # create DataLoader for trainin set
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)

        self.train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        # create DataLoader for validation set
        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)

        validation_sampler = RandomSampler(validation_data)
        self.validation_data_loader = DataLoader(validation_data, sampler=validation_sampler,
                                                 batch_size=self.batch_size)





