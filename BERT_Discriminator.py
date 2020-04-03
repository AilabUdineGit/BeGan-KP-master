import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np
from transformers import AdamW, \
    BertPreTrainedModel, BertModel, BertTokenizer, BertForTokenClassification, BertForMultipleChoice, BertForSequenceClassification, \
    XLNetTokenizer, XLNetModel, RobertaModel, RobertaTokenizer, DistilBertModel, DistilBertTokenizer, \
    AlbertTokenizer, AlbertModel
# from params import PARAMS


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # first_token_tensor = hidden_states[:, 0]
        # pooled_output = self.dense(first_token_tensor)
        avg = torch.mean(hidden_states, dim=1)  # gl: new
        pooled_output = self.dense(avg)  # gl: new
        pooled_output = self.activation(pooled_output)
        return pooled_output


class NetModel(BertPreTrainedModel):
    # class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, hidden_dim, n_layers):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        # gl: needed for evaluating rewards
        self.sigmoid = nn.Sigmoid()
        self.MegaRNN = nn.GRU(hidden_dim, 2 * hidden_dim, n_layers)
        self.Linear = nn.Linear(2 * hidden_dim, 1)

        # self.sigmoid = nn.Sigmoid()  # gl
        # self.tanh = nn.Tanh()  # gl
        # self.pooler = BertPooler(config)  # gl

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

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # # gl: original
        # pooled_output = outputs[1]
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)

        # gl: new
        hidden_state = outputs[0]
        # hidden_state = self.dropout(hidden_state)
        hidden_state = torch.mean(hidden_state, dim=1)
        hidden_state = self.dropout(hidden_state)
        logits = self.classifier(hidden_state)

        # # gl: same as BertModel
        # hidden_state = outputs[0]
        # hidden_state = self.dropout(hidden_state)
        # hidden_state = self.pooler(hidden_state)  # gl: same as BertModel
        # logits = self.classifier(hidden_state)

        # # gl: with sigmoid
        # logits = self.sigmoid(logits)

        # # # gl: with tanh
        # logits = self.tanh(logits)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


# def evaluated_loss(self, results, labels):
#         return self.loss_function(input=results, target=labels)


class NetModelMC(BertPreTrainedModel):
    def __init__(self, config, hidden_dim, n_layers):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # gl: needed for evaluating rewards
        self.sigmoid = nn.Sigmoid()
        self.MegaRNN = nn.GRU(hidden_dim, 2 * hidden_dim, n_layers)
        self.Linear = nn.Linear(2 * hidden_dim, 1)

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
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # # gl: original
        # pooled_output = outputs[1]
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)

        # gl: new
        hidden_state = outputs[0]
        hidden_state = torch.mean(hidden_state, dim=1)
        hidden_state = self.dropout(hidden_state)
        logits = self.classifier(hidden_state)

        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


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
    'BERT': NLPModel('BERT', BertTokenizer, BertForMultipleChoice, "bert-base-uncased"),  # bert-large-uncased  gl: BertForSequenceClassification; BertForMultipleChoice
    'XLNet': NLPModel('XLNet', XLNetTokenizer, XLNetModel, "xlnet-large-uncased"),  # bert-base-cased
    'RoBERTa': NLPModel('Roberta', RobertaTokenizer, RobertaModel, "roberta-base"),
    'DistilBERT': NLPModel('DistilBERT', DistilBertTokenizer, DistilBertModel, 'distilbert-base-uncased'),
    'AlBERT': NLPModel('AlBERT', AlbertTokenizer, AlbertModel, 'albert-xlarge')
    # every supported model can be added
    # https://github.com/huggingface/transformers#model-architectures
    # Need to change util.generate_content_file and MAX_LENGHT because of different tokenization, and EMBEDDING_LENGHT
}
