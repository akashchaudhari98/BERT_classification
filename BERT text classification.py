import pandas as pd
import numpy as np
import random
from pandas.core.frame import DataFrame
import torch
import time
from transformers import BertTokenizer
from sklearn import preprocessing
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup


class textclassification:
    def __init__(self,
                 data=None,
                 datatype=None,
                 column_name=(),
                 no_of_labels=2,
                 is_trainable=False,
                 model_path=None,
                 data_path=None,
                 max_len=512,
                 tokenizer_config={},
                 batch_size=32,
                 epoch=5) -> None:

        
        self.training_data = data
        self.data_path = data_path
        # should not have anything other than csv, dataframe
        self.training_datatype = datatype
        self.is_trainable = is_trainable
        # cannot be empty
        self.model_path = model_path
        # should only have 2 values
        self.csv_column_name = column_name
        self.no_of_labels = no_of_labels
        self.max_len = max_len
        self.tokenizer_config = tokenizer_config
        self.batch_size = batch_size
        self.epoch = epoch

        if not is_trainable :
            if self.model_path == None or self.model_path == "":
                raise ValueError("No path provided")

            self.model = self.__load_model__(model_path)

    def __get_device__(self):

        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        return device

    def __load_training_data__(self):
        # if datatype is anything other than csv,dataframe
        column_names = self.csv_column_name
        if self.training_datatype == "csv":
            format = self.data_path.split(".")
            if format[1] == "csv":
                training_data = pd.read_csv(self.data_path)
                training_text = training_data[column_names[0]]
                labels = training_data[column_names[1]]

                return training_text, labels

            else:
                raise ValueError("Datatype set as csv, but csv file not passed, please cheak the file format," +
                                 "filename should not have multiple '.' ")

        elif self.training_datatype == "dataframe":
            if isinstance(self.training_data, DataFrame):
                training_data = self.training_data
                training_text = training_data[column_names[0]]
                labels = training_data[column_names[1]]

                return training_text, labels
            else:
                raise ValueError(
                    "datatype set as dataframe but dataframe not passes")

    def __preprocess__(self):
        pass

    def __tokenize__(self, training_text):
        '''tokenizes the traning text according to bert tokenization '''
        encoded_sentences = []
        # if tokenizer config is empty
        if self.tokenizer_config == {}:
            token_config = {'pretrained_model_name_or_path': 'bert-base-uncased',
                            "do_lower_case": "True"}
        else:
            token_config = self.tokenizer_config

        tokenizer = BertTokenizer.from_pretrained(
            **token_config)

        for data in training_text:
            enoded_sent = tokenizer.encode(
                data,
                add_special_tokens=True
            )
            encoded_sentences.append(enoded_sent)

        encoded_sentences = pad_sequences(encoded_sentences, maxlen=self.MAX_LEN, dtype="long",
                                          value=0, truncating="post", padding="post")
        return encoded_sentences

    def __label_preprocessing__(self, label_data,convert = True):

        le = preprocessing.LabelEncoder()
        if convert :
            labels = le.fit_transform(label_data)
            return labels
        else:
            labels = le.inverse_transform(label_data)
            return labels

    def __create_attention_mask__(self, encoded_sent):

        sentences_attention_masks = []
        for encoding in encoded_sent:
            att_mask = [int(token_id > 0) for token_id in encoding]
            sentences_attention_masks.append(att_mask)

        return sentences_attention_masks

    def train(self):

        training_text, labels = self.__load_training_data__()

        training_text = self.__tokenize__(training_text)
        labels = self.__label_preprocessing__(labels)

        sentences_attention_masks = self.__create_attention_mask__()

        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(training_text, labels,
                                                                                            random_state=2018, test_size=0.1)
        train_masks, validation_masks, _, _ = train_test_split(sentences_attention_masks, labels,
                                                               random_state=2018, test_size=0.1)

        train_inputs = torch.tensor(train_inputs)
        validation_inputs = torch.tensor(validation_inputs)

        train_labels = torch.tensor(train_labels)
        validation_labels = torch.tensor(validation_labels)

        train_masks = torch.tensor(train_masks)
        validation_masks = torch.tensor(validation_masks)

        # Create the DataLoader for our training set.
        self.train_data = TensorDataset(
            train_inputs, train_masks, train_labels)
        self.train_sampler = RandomSampler(self.train_data)
        self.train_dataloader = DataLoader(
            self.train_data, sampler=self.train_sampler, batch_size=self.batch_size)

        # Create the DataLoader for our validation set.
        self.validation_data = TensorDataset(
            validation_inputs, validation_masks, validation_labels)
        self.validation_sampler = SequentialSampler(self.validation_data)
        self.validation_dataloader = DataLoader(
            self.validation_data, sampler=self.validation_sampler, batch_size=self.batch_size)

        ############
        #LOAD MODEL#
        ############

        self.model = BertForSequenceClassification.from_pretrained(
                                "bert-base-uncased",
                                num_labels=self.no_of_labels,
                                output_attentions=False,
                                output_hidden_states=False
                                ).to(self.__get_device__)

        self.optimizer = AdamW(self.model.parameters(),
                               lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                               
                               eps=1e-8
                               )

        self.__start_training__()

    def __start_training__(self):

        torch.cuda.empty_cache()
        total_steps = len(self.train_dataloader) * self.epochs

        scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)

        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        loss_values = []

        for epoch_i in range(0, self.epochs):

            # ========================================
            #               Training
            # ========================================

            print("")
            print(
                '======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')

            t0 = time.time()

            total_loss = 0

            self.model.train()

            # For each batch of training data...
            for step, batch in enumerate(self.train_dataloader):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = self.__format_time__(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                        step, len(self.train_dataloader), elapsed))

                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                self.model.zero_grad()

                outputs = self.model(b_input_ids,
                                     token_type_ids=None,
                                     attention_mask=b_input_mask,
                                     labels=b_labels)
                loss = outputs[0]

                total_loss += loss.item()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()

                scheduler.step()

            avg_train_loss = total_loss / len(self.train_dataloader)

            loss_values.append(avg_train_loss)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(
                self.__format_time__(time.time() - t0)))

            print("")
            print("Running Validation...")

            t0 = time.time()

            self.model.eval()

            # Tracking variables
            eval_loss, eval_accuracy = 0, 0

            nb_eval_steps, nb_eval_examples = 0, 0

            for batch in self.validation_dataloader:

                batch = tuple(t.to(self.device) for t in batch)

                b_input_ids, b_input_mask, b_labels = batch

                with torch.no_grad():
                    outputs = self.model(b_input_ids,
                                         token_type_ids=None,
                                         attention_mask=b_input_mask)

                logits = outputs[0]

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)

                eval_accuracy += tmp_eval_accuracy

                nb_eval_steps += 1

                print("  Accuracy: {0:.2f}".format(
                    eval_accuracy/nb_eval_steps))

        print("")
        print("Training complete!")
    
    def save_model(self,location,name):

        print("Saving model {} to {}" .format(name,location))
        model_to_save = self.model
        model_to_save.save_pretrained(location)
        self.tokenizer.save_pretrained(location)
    
    def predict(self,text):
        if not isinstance(text,list):
            text = [text]
        
        encoded_sent = self.__tokenize__(text)
        attention_masks = self.__create_attention_mask__(encoded_sent)

        prediction_inputs = torch.tensor(encoded_sent).to(self.device)
        prediction_masks = torch.tensor(attention_masks).to(self.device)

        outputs = self.model(prediction_inputs, token_type_ids=None,
                             attention_mask=prediction_masks)

        predicted_label = np.argmax(
            outputs[0].detach().cpu().numpy(), axis=1).flatten()
        
        predicted_label = self.__label_preprocessing__(predicted_label,convert=False)

        return predicted_label
    
    def __load_model__(self, model_path):
        print("Loading Model : ", model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model = self.model.to(self.device)

        return self.model
        


if __name__ == "__main__":

    csv_ = "C:/Users/akash/Desktop/archive/IMDB Dataset.csv"

    css = textclassification(
        datatype="csv",
        column_name=("review", "sentiment"),
        no_of_labels=2,
        is_trainable=False,
        model_path=None,
        data_path=csv_,
        max_len=512,
        tokenizer_config={}
    )
    css.train()
