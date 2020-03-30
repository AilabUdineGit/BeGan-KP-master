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
        self.hidden_dim = config.D_hidden_dim
        self.n_layers = config.D_layers

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # if PARAMS.USE_BLSTM:
        #     self.bilstm = nn.LSTM(self.embedding_dim, PARAMS.LSTM_SIZE, bidirectional=True)
        #     self.classifier = nn.Linear(PARAMS.LSTM_SIZE * 2, config.num_labels)
        # else:
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()
        self.MegaRNN = nn.GRU(hidden_dim, 2 * hidden_dim, n_layers)

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


    def get_hidden_states (self,
            input_ids,
            attention_mask,
            token_type_ids,
            labels):
        position_ids = None
        head_mask = None
        inputs_embeds = None
        h_states_output = self.forward(input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            labels)
        loss, logits = h_states_output[:2]
        a = (input_ids[:, :512] == 102).nonzero() #Sa: position for all the sep
        print("a",a)
        t1 = a[0, 1].item() #sa: pick the first sep that divid src from kp fro samp 1
        t1_ = a[1,1].item()
        t2 = a[2, 1].item()# sa:pick the first sep that divid src from kp fro samp 2
        t2_= a[3,1].item()
        h_abstract_f1 = logits[12][0, :t1, :]
        h_kph_f1 = logits[12][0, t1:t1_, :]
        h_abstract_f2 = logits[12][1, :t2, :]
        h_kph_f2 = logits[12][1, t2:t2_, :]
        print(t1)
        print(t2)
        print(t1_)
        print(t2_)
        print()
        print(h_abstract_f1.shape[1])
        print(h_abstract_f2.shape)
        print("kph1", h_kph_f1.shape)
        print("Kph2",h_kph_f2.shape)
        if t1 >= t2:
            h_abstract = torch.zeros(2,t1,h_abstract_f1.shape[1])
            h_abstract[0, :t1, :] = h_abstract_f1
            h_abstract[1, :t2, :] = h_abstract_f2
            print(h_abstract.shape)
        else:
            h_abstract = torch.zeros(2, t2, h_abstract_f2.shape[1])
            h_abstract[0, :t1, :] = h_abstract_f1
            h_abstract[1, :t2, :] = h_abstract_f2
            print(h_abstract.shape)

        if t1_-t1 >= t2_-t2:
            h_kph = torch.zeros(2, t1_-t1, h_kph_f1.shape[1])
            h_kph[0, :t1_-t1, :] = h_kph_f1
            h_kph[1, :t2_-t2, :] = h_kph_f2
            print(h_kph.shape)
        else:
            h_kph = torch.zeros(2, t2_ - t2, h_kph_f2.shape[1])
            h_kph[0, :t1_ - t1, :] = h_kph_f1
            h_kph[1, :t2_ - t2, :] = h_kph_f2
            print(h_kph.shape)

        return h_abstract,h_kph


    def evaluated_loss(self, results, labels):
        return self.loss_function(input=results, target=labels)

    def Catter(self, kph, rewards, total_len):
        lengths = [len(kp) + 1 for kp in kph]
        max_len = max(lengths)
        x = torch.Tensor([])
        rewards_shape = rewards.repeat(max_len).reshape(-1, rewards.size(0)).t()
        x = torch.Tensor([])
        x = x.to(self.devices)
        for i, keyphrase in enumerate(rewards_shape):
            x = torch.cat((x, keyphrase[:lengths[i]]))
        x = F.pad(input=x, pad=(0, total_len - x.size(0)), mode='constant', value=0)
        return x

    def calculate_rewards(self, abstract_t, kph_t, start_len, len_list, pred_str_list, total_len, gamma=0.99):
        start_len = 1
        print("abstact shape", abstract_t.shape)
        #abstract_t = torch.mean(abstract_t, dim=1)
        #abstract_t = abstract_t.unsqueeze(1)
        print("abstact shape",abstract_t.shape)
        print("khp",kph_t.shape)
        concat_output = torch.cat((abstract_t, kph_t), dim=1)
        concat_output = concat_output.permute(1, 0, 2)
        x, hidden = self.MegaRNN(concat_output)
        output = self.Linear(x)
        output = output.squeeze(2).t()
        avg_outputs = self.sigmoid(output)
        reward_outputs = torch.Tensor([]).to(self.devices)
        # print('len_list = ', len_list)  # 32 elementi, = batch
        for i, len_i in enumerate(len_list):
            avg_outputs[i, start_len:start_len + len_i] = avg_outputs[i, start_len:start_len + len_i]
            batch_rewards = self.Catter(pred_str_list[i], avg_outputs[i, start_len:start_len + len_i], total_len)
            reward_outputs = torch.cat((reward_outputs, batch_rewards))  # gl: torch.Size([192]); dtype: torch.float32
        # print('reward_outputs.size(): ' + str(reward_outputs.size()) + '; dtype: ' + str(reward_outputs.dtype))  # gl
        return reward_outputs




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
