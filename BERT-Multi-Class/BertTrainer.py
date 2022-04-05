import time
import datetime
import plotly.graph_objects as go
from transformers import (
    BertForSequenceClassification,
    AdamW,
    BertConfig,
    get_linear_schedule_with_warmup,
    pipeline,
    BertTokenizer
)


class BertTrainer:

    def __init__(self,
                 lr,  # learning rate
                 eps,  # epsilon
                 epochs,  # number of epochs
                 train_data_loader,
                 validation_data_loader,
                 verbose=0,
                 model=None,
                 num_labels=11):
        self.verbose = verbose
        self.model = model
        self.optimizer = None
        self.lr = lr
        self.eps = eps
        self.epochs = epochs
        self.train_data_loader = train_data_loader
        self.total_steps = len(self.train_data_loader) * self.epochs
        self.validation_data_loader = validation_data_loader
        self.num_labels = num_labels

    def __call__(self):
        self.initiateBertModel()

    def initiateBertModel(self):
        if self.model is None:
            self.flag_model = False
            self.model = BertForSequenceClassification.from_pretrained(
                '/home/arielblo/Models/BetterModel',
                # '/home/arielblo/Models/',
                # 'bert-base-uncased', # Use the 12-layer BERT model, with an uncased vocab.
                # 'bio-bert/biobert_v1.0_pmc/', # test the bio-bert
                # 'dmis-lab/biobert-v1.1',
                num_labels=self.num_labels,  # The number of output labels--11 for multiclas classification.
                output_attentions=False,  # Whether the model returns attentions weights.
                output_hidden_states=False,  # Whether the model returns all hidden-states.
            )
        else:
            self.flag_model = True

        # Tell pytorch to run this model on the GPU.
        self.model.cuda()
        self._optimizer_scheduler()

        if self.verbose == 1:
            params = list(self.model.named_parameters())
            print('The BERT model has {:} different named parameters.\n'.format(len(params)))
            print('==== Embedding Layer ====\n')
            for p in params[0:5]:
                print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
            print('\n==== First Transformer ====\n')
            for p in params[5:21]:
                print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
                print('\n==== Output Layer ====\n')
            for p in params[-4:]:
                print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    def _optimizer_scheduler(self):
        self.optimizer = AdamW(self.model.parameters(),
                               lr=self.lr,
                               eps=self.eps)

        # Create the learning rate scheduler.
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,  # Default value in run_glue.py
                                                         num_training_steps=self.total_steps)

    def flat_accuracy(self, preds, labels):
        """
        # Function to calculate the accuracy of our predictions vs labels
        """
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def f1_score_func(self, preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = np.argmax(labels, axis=1).flatten()
        precision = precision_score(labels_flat, preds_flat, average="macro")
        recall = recall_score(labels_flat, preds_flat, average="macro")
        f1 = f1_score(labels_flat, preds_flat, average='weighted')
        return f1, precision, recall

    def format_time(self, elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def print_model_graph(self):

        f = pd.DataFrame(trainer.loss_values, columns=['training_loss'])
        f['val_loss'] = trainer.val_loss_values
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=f.index, y=f['training_loss'],
                                 mode='lines',
                                 name='training_loss'))
        fig.add_trace(go.Scatter(x=f.index, y=f['val_loss'],
                                 mode='lines',
                                 name='validation_loss'))
        fig.update_layout(title='Training loss of the Model',
                          xaxis_title='Epoch',
                          yaxis_title='Loss')
        fig.show()

    def train(self, seed_val=42):
        SEED_VAL = seed_val
        random.seed(SEED_VAL)
        np.random.seed(SEED_VAL)
        torch.manual_seed(SEED_VAL)
        torch.cuda.manual_seed_all(SEED_VAL)

        # Store the average loss after each epoch so we can plot them.
        self.loss_values, self.val_loss_values = [], []

        # ========================================
        #               Training
        # ========================================

        for epoch_i in range(self.epochs):

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print("Training..")
            # Measure how long a single training epoch takes.
            t0 = time.time()
            # reset the total loss for a current epoch
            total_loss = 0

            for step, batch in enumerate(self.train_data_loader):
                # progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # calculate elapsed time
                    elapsed = self.format_time(time.time() - t0)
                    # report progress
                    print(
                        'Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.train_data_loader), elapsed))

                # unpack this training batch from data loader
                # when we unpack we copy each tensor to the GPU
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                # device is configured in the GPU configurations above
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                # pyTorch doesn't clear previously calculated gradients so we do it manually...
                self.model.zero_grad()

                # preform forward pass over the model
                outputs = self.model(b_input_ids,
                                     token_type_ids=None,
                                     attention_mask=b_input_mask,
                                     labels=b_labels)
                # The call to `model` always returns a tuple, so we need to pull the loss value out of the tuple.
                loss = outputs[0]

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                # loss is a differentiable variable, and casting it to float won't concede much GPU memoery
                total_loss += float(loss.item())

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # clip the norm of gradients to 1.0 in order to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                self.optimizer.step()

                # update learning rate
                self.scheduler.step()

            # Calculate average loss over the epoch
            avg_train_loss = total_loss / len(self.train_data_loader)

            # store loss value for plotting curve
            self.loss_values.append(avg_train_loss)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(self.format_time(time.time() - t0)))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.
            print("")
            print("Running Validation..")
            t0 = time.time()

            # assign the model to evaluation mode -> the dropout layers behave
            # differently during valuation
            self.model.eval()

            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            for batch in self.validation_data_loader:

                # add batch to GPU
                batch = tuple(t.to(device) for t in batch)

                b_input_ids, b_input_mask, b_labels = batch

                # compute step without storing gradients, saving memory and time
                with torch.no_grad():

                    if not self.flag_model:
                        # Forward pass, calculate logit predictions.
                        # This will return the logits rather than the loss because we have
                        # not provided labels.
                        outputs = self.model(b_input_ids,
                                             token_type_ids=None,
                                             attention_mask=b_input_mask)
                        # get the logits output by the model, values prior softmax
                        logits = outputs[0]
                    else:
                        # Forward pass, calculate logit predictions.
                        # This will return the logits and the loss because we have provided labels.
                        outputs = self.model(b_input_ids,
                                             token_type_ids=None,
                                             attention_mask=b_input_mask,
                                             labels=b_labels)
                        eval_loss = outputs[0].item()
                        logits = outputs[1]

                    # move logits to CPU
                    logits = logits.detach().cpu().numpy()

                    # preds  = outputs[1].detach().cpu().numpy()

                    label_ids = b_labels.to('cpu').numpy()
                    # Calculate accuracy for this batch
                    tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)
                    # tmp_eval_accuracy = self.flat_accuracy(preds, label_ids)

                    # Calculate total accuracy
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_steps += 1

            self.val_loss_values.append(eval_loss)
            print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
            print("  Validation took: {:}".format(self.format_time(time.time() - t0)))
        print("")
        print("training complete")


