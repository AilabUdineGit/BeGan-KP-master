import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np
from transformers import AdamW, \
    BertPreTrainedModel, BertModel, BertTokenizer, BertForTokenClassification, BertForMultipleChoice, BertForSequenceClassification, \
    XLNetTokenizer, XLNetModel, RobertaModel, RobertaTokenizer, DistilBertModel, DistilBertTokenizer, \
    AlbertTokenizer, AlbertModel
# from params import PARAMS


class NetModel(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.embedding_dim = config.hidden_size
        self.loss_function = nn.BCEWithLogitsLoss()  # gl

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # if PARAMS.USE_BLSTM:
        #     self.bilstm = nn.LSTM(self.embedding_dim, PARAMS.LSTM_SIZE, bidirectional=True)
        #     self.classifier = nn.Linear(PARAMS.LSTM_SIZE * 2, config.num_labels)
        # else:
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        batch_size = len(input_ids)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        # if PARAMS.USE_BLSTM:
        #     bilstm_out, hn = self.bilstm(sequence_output)
        #     logits = self.classifier(bilstm_out)
        # else:
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # (loss), scores, (hidden_states), (attentions)

    def evaluated_loss(self, results, labels):
        return self.loss_function(input=results, target=labels)


class NLPModel:

    def __init__(self, name, tokenizer, model, pretrained_weights):
        self.tokenizer = tokenizer
        self.model = model
        self.name = name
        self.pretrained_weights = pretrained_weights

    def choose(self, model_config=None):
        self.tokenizer = self.tokenizer.from_pretrained(self.pretrained_weights)  # gl: , add_special_tokens=True
        self.model = self.model.from_pretrained(self.pretrained_weights, config=model_config)
        # _ = self.tokenizer.add_tokens(['<digit>'])  # gl: aggiunto il token <digit> che ora viene considerato come una parola.
        # nota: il token <digit> manda in errore la forward() perché andrebbe gestito  a livello di fine-tuning;
        # quindi: lasciare che <digit> venga tokenizzato in maniera generica, sarà comunque considerato una parola specifica.
        self.model.resize_token_embeddings(len(self.tokenizer))
        # _ = self.add_specific_tokens()
        # nota: anche lo special token [EOS] se introdotto andrebbe poi gestito; tralasciare e concatenare semplicemente le KP con un punto
        # capire bene in seguito la questione
        return self

    def add_specific_tokens(self):
        # special_tokens_dict = {'eos_token': '[EOS]', 'unk_token': '[UNK]'}
        # special_tokens_dict = {'eos_token': '[EOS]'}
        special_tokens_dict = {'eos_token': '<eos>'}
        num_added_tok = self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))
        return num_added_tok


NLP_MODELS = {  # models from transformers library
    'BERT': NLPModel('BERT', BertTokenizer, BertForSequenceClassification, "bert-base-uncased"),  # bert-large-uncased
    'XLNet': NLPModel('XLNet', XLNetTokenizer, XLNetModel, "xlnet-large-uncased"),  # bert-base-cased
    'RoBERTa': NLPModel('Roberta', RobertaTokenizer, RobertaModel, "roberta-base"),
    'DistilBERT': NLPModel('DistilBERT', DistilBertTokenizer, DistilBertModel, 'distilbert-base-uncased'),
    'AlBERT': NLPModel('AlBERT', AlbertTokenizer, AlbertModel, 'albert-xlarge')
    # every supported model can be added
    # https://github.com/huggingface/transformers#model-architectures
    # Need to change util.generate_content_file and MAX_LENGHT because of different tokenization, and EMBEDDING_LENGHT
}
