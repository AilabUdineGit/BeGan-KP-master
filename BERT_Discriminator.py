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
    # class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

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
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForSequenceClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, logits = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

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
    'BERT': NLPModel('BERT', BertTokenizer, BertForSequenceClassification, "bert-base-uncased"),  # bert-large-uncased  gl: BertForSequenceClassification; BertForMultipleChoice
    'XLNet': NLPModel('XLNet', XLNetTokenizer, XLNetModel, "xlnet-large-uncased"),  # bert-base-cased
    'RoBERTa': NLPModel('Roberta', RobertaTokenizer, RobertaModel, "roberta-base"),
    'DistilBERT': NLPModel('DistilBERT', DistilBertTokenizer, DistilBertModel, 'distilbert-base-uncased'),
    'AlBERT': NLPModel('AlBERT', AlbertTokenizer, AlbertModel, 'albert-xlarge')
    # every supported model can be added
    # https://github.com/huggingface/transformers#model-architectures
    # Need to change util.generate_content_file and MAX_LENGHT because of different tokenization, and EMBEDDING_LENGHT
}
