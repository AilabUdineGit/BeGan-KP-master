import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertTokenizer, BertForSequenceClassification, \
    XLNetTokenizer, XLNetModel, RobertaModel, RobertaTokenizer, DistilBertModel, DistilBertTokenizer, \
    AlbertTokenizer, AlbertModel, AutoModel, AutoTokenizer, AutoModelForSequenceClassification
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

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.loss_function = nn.BCEWithLogitsLoss()  # gl

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # self.sigmoid = nn.Sigmoid()  # gl: normalizza l'output tra 0 e 1, forse non l'ideale per la reward
        # # self.devices = device
        # self.MegaRNN = nn.GRU(hidden_dim, 2 * hidden_dim, n_layers)
        # self.Linear = nn.Linear(2 * hidden_dim, 1)

        # # gl: needed for evaluating rewards
        # self.sigmoid = nn.Sigmoid()
        # self.MegaRNN = nn.GRU(hidden_dim, 2 * hidden_dim, n_layers)
        # self.Linear = nn.Linear(2 * hidden_dim, 1)

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
        sequence_output = outputs[0]
        # torch.save(sequence_output, 'prova/sequence_output.pt')  # gl saving tensors
        # sequence_output = self.dropout(sequence_output)
        sequence_output = torch.mean(sequence_output, dim=1)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        # # gl: same as BertModel
        # hidden_state = outputs[0]
        # hidden_state = self.dropout(hidden_state)
        # hidden_state = self.pooler(hidden_state)  # gl: same as BertModel
        # logits = self.classifier(hidden_state)

        # gl: with sigmoid
        # logits = self.sigmoid(logits)  # gl: normalizza l'output tra 0 e 1, forse non l'ideale per la reward

        # # # gl: with tanh
        # logits = self.tanh(logits)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        # torch.save(outputs, 'prova/R_outputs.pt')  # gl saving tensors

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

    # def get_hidden_states(self, input_ids, attention_mask, token_type_ids, labels):
    #     position_ids = None
    #     head_mask = None
    #     inputs_embeds = None
    #     h_states_output = self.forward(input_ids,
    #                                    attention_mask,
    #                                    token_type_ids,
    #                                    position_ids,
    #                                    head_mask,
    #                                    inputs_embeds,
    #                                    labels)
    #     loss, logits = h_states_output[:2]
    #     #   print("input_ids[0]",list(logits[12].size())[0])
    #     batch = list(logits[12].size())[0]
    #     # a = (input_ids[:, :512] == 102).nonzero()  # Sa: position for all the sep
    #     a = (input_ids[:, :input_ids.shape[1]] == 102).nonzero()  # Sa: position for all the sep
    #     #    print("a",a)
    #     #    print("a",a.shape)
    #     t = []
    #     t_even = []
    #     t_odd = []
    #     for i in range(batch * 2):
    #         t.append(a[i, 1].item())
    #     for x in range(len(t)):
    #         if x % 2 == 0:
    #             t_even.append(t[x])
    #         else:
    #             t_odd.append(t[x])
    #     t_diff = []
    #     zip_object = zip(t_odd, t_even)
    #     for t_odd_i, t_even_i in zip_object:
    #         t_diff.append(t_odd_i-t_even_i)
    #     t_max = max(t_even)
    #     t_min = max(t_diff)
    #     h_abstract = torch.zeros(batch, t_max, list(logits[12].size())[2])
    #     h_kph = torch.zeros(batch, t_min, list(logits[12].size())[2])
    #     for i in range(batch):
    #         h_abstract[i, :t_even[i], :] = logits[12][i, :t_even[i], :]
    #         h_kph[i, :t_odd[i]-t_even[i], :] = logits[12][i, :t_odd[i]-t_even[i], :]
    # # a = (input_ids[:, :512] == 102).nonzero() #Sa: position for all the sep
    # #     t1 = a[0, 1].item() #sa: pick the first sep that divid src from kp fro samp 1
    # #     t1_ = a[1,1].item()
    # #     t2 = a[2, 1].item()# sa:pick the first sep that divid src from kp fro samp 2
    # #     t2_= a[3,1].item()
    # #     h_abstract_f1 = logits[12][0, :t1, :]
    # #     h_kph_f1 = logits[12][0, t1:t1_, :]
    # #     h_abstract_f2 = logits[12][1, :t2, :]
    # #     h_kph_f2 = logits[12][1, t2:t2_, :]
    # #     if t1 >= t2:
    # #         h_abstract = torch.zeros(2,t1,h_abstract_f1.shape[1])
    # #         h_abstract[0, :t1, :] = h_abstract_f1
    # #         h_abstract[1, :t2, :] = h_abstract_f2
    # #         print(h_abstract.shape)
    # #     else:
    # #         h_abstract = torch.zeros(2, t2, h_abstract_f2.shape[1])
    # #         h_abstract[0, :t1, :] = h_abstract_f1
    # #         h_abstract[1, :t2, :] = h_abstract_f2
    # #     if t1_-t1 >= t2_-t2:
    # #         h_kph = torch.zeros(2, t1_-t1, h_kph_f1.shape[1])
    # #         h_kph[0, :t1_-t1, :] = h_kph_f1
    # #         h_kph[1, :t2_-t2, :] = h_kph_f2
    # #     else:
    # #         h_kph = torch.zeros(2, t2_ - t2, h_kph_f2.shape[1])
    # #         h_kph[0, :t1_ - t1, :] = h_kph_f1
    # #         h_kph[1, :t2_ - t2, :] = h_kph_f2
    #     return h_abstract, h_kph

    # def evaluated_loss(self, results, labels):
    #    return self.loss_function(input=results, target=labels)

    # def Catter(self, kph, rewards, total_len, device):
    #     lengths = [len(kp) + 1 for kp in kph]
    #     max_len = max(lengths)
    #     x = torch.Tensor([])
    #     rewards_shape = rewards.repeat(max_len).reshape(-1, rewards.size(0)).t()
    #     x = torch.Tensor([])
    #     # x = x.to(self.devices)
    #     x = x.to(device)
    #     for i, keyphrase in enumerate(rewards_shape):
    #         # print("keypharse", keyphrase, i)
    #         # print("reward", rewards_shape.shape)
    #         # print("lengths",lengths)
    #         if i <= len(lengths)-1:
    #             x = torch.cat((x, keyphrase[:lengths[i]]))
    #             # print("x after",x)
    #     x = F.pad(input=x, pad=(0, total_len - x.size(0)), mode='constant', value=0)
    #     return x
    #
    # def calculate_rewards(self, abstract_t, kph_t, start_len, len_list, pred_str_list, total_len, devices, gamma=0.99):
    #     start_len = 1
    #     # print("abstact shape", abstract_t.shape)
    #     abstract_t = torch.mean(abstract_t, dim=1)
    #     abstract_t = abstract_t.unsqueeze(1)
    #     # print("abstact shape",abstract_t.shape)
    #     # print("khp",kph_t.shape)
    #     concat_output = torch.cat((abstract_t, kph_t), dim=1)
    #     # print("conat_output",concat_output.shape)
    #     concat_output = concat_output.permute(1, 0, 2)
    #     # print("conat_output",concat_output.shape)
    #     device = torch.device("cuda")  # a CUDA device object
    #     concat_output = concat_output.to(device)
    #     x, hidden = self.MegaRNN(concat_output)
    #     output = self.Linear(x)
    #     output = output.squeeze(2).t()
    #     avg_outputs = self.sigmoid(output)
    #     # reward_outputs = torch.Tensor([]).to(self.devices)  # gl
    #     reward_outputs = torch.Tensor([]).to(devices)
    #     # print('len_list = ', len_list)  # 32 elementi, = batch
    #     for i, len_i in enumerate(len_list):
    #         avg_outputs[i, start_len:start_len + len_i] = avg_outputs[i, start_len:start_len + len_i]
    #         batch_rewards = self.Catter(pred_str_list[i], avg_outputs[i, start_len:start_len + len_i], total_len)
    #         reward_outputs = torch.cat((reward_outputs, batch_rewards))  # gl: torch.Size([192]); dtype: torch.float32
    #     # print('reward_outputs.size(): ' + str(reward_outputs.size()) + '; dtype: ' + str(reward_outputs.dtype))  # gl
    #     return reward_outputs


class NetModelMC(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # # gl: needed for evaluating rewards
        # self.sigmoid = nn.Sigmoid()
        # self.MegaRNN = nn.GRU(hidden_dim, 2 * hidden_dim, n_layers)
        # self.Linear = nn.Linear(2 * hidden_dim, 1)

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
    'BERT': NLPModel('BERT', BertTokenizer, BertForSequenceClassification, "bert-base-uncased"),  # BertForMultipleChoice, BertForSequenceClassification
    # bert-large-uncased  gl: BertForSequenceClassification; BertForMultipleChoice
    'XLNet': NLPModel('XLNet', XLNetTokenizer, XLNetModel, "xlnet-large-uncased"),  # bert-base-cased
    'RoBERTa': NLPModel('Roberta', RobertaTokenizer, RobertaModel, "roberta-base"),
    'DistilBERT': NLPModel('DistilBERT', DistilBertTokenizer, DistilBertModel, 'distilbert-base-uncased'),
    'AlBERT': NLPModel('AlBERT', AlbertTokenizer, AlbertModel, 'albert-xlarge'),
    'SpanBERT': NLPModel('SpanBERT', AutoTokenizer, AutoModelForSequenceClassification, 'SpanBERT/spanbert-base-cased')  # AutoModel, AutoModelForSequenceClassification
    # every supported model can be added
    # https://github.com/huggingface/transformers#model-architectures
    # Need to change util.generate_content_file and MAX_LENGHT because of different tokenization, and EMBEDDING_LENGHT
}
