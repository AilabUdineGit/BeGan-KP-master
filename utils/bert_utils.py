import numpy as np
from nltk.tokenize import MWETokenizer


class Kp20kTokenizer:

    def __init__(self, mweTokenizer, emb_tokenizer):
        self.mweTokenizer = mweTokenizer
        self.emb_tokenizer = emb_tokenizer

    def tokenize(self, sentence):
        tokens = self.emb_tokenizer.tokenize(sentence)
        # '<', 'digit', '>'
        # '<', 'unk', '>'
        tokens = self.mweTokenizer.tokenize(tokens)
        return tokens


class Reader:
    # mweTokenizer = MWETokenizer([('<', 'digit', '>'), ('<', 'unk', '>'), ('[', 'CLS', ']'), ('[', 'SEP', ']')], separator="")
    mweTokenizer = MWETokenizer([('<', 'digit', '>'),
                                 ('<', 'unk', '>'),
                                 ('<', 'pad', '>'),
                                 ('<', 'bos', '>'),
                                 ('<', 'eos', '>'),
                                 ('<', 'sep', '>'),
                                 ('<', 'peos', '>'),
                                 ('[', 'CLS', ']'),
                                 ('[', 'SEP', ']')],
                                separator="")
    # skip_words = ["[CLS]", "[SEP]", "<unk>", "<digit>"]
    skip_words = ["[CLS]", "[SEP]", "<unk>", "<digit>", "<pad>", "<bos>", "<eos>", "<sep>", "<peos>"]
    # tokenizer = Tokenizer('eg')

    # def __init__(self, model, title_repetitions=1, num_labels=3):
    def __init__(self, bert_tokenizer):
        self.tokenizer = bert_tokenizer

    def title_fix(self, sentences):
        # la prima frase contiene sia titolo che la prima frase del documento
        phrases = sentences[0].replace("\n\t", " ").split("\n")  # per la struttura dei documenti posso separarli così
        title = phrases[0]
        titles = []
        for i in range(self.title_repetitions):
            titles.append(title)
        return titles + phrases[1:] + sentences[1:]

    def tokenize_with_map_to_origin(self, text_words):
        """
        :param text_words: list of words (tokens) of the text
        :return:
        """
        tokens_map = []
        sentences_tokens = []
        all_tokens = []
        index = 0
        all_words = []
        tokens = []
        for i, word in enumerate(text_words):
            all_words.append(word)
            if word in self.skip_words:
                _tokens = [word]
            else:
                # _tokens = emb_tokenizer.tokenize(word)
                _tokens = self.tokenizer.tokenize(word)
            for token in _tokens:
                tokens.append(token)
                all_tokens.append(token)
                tokens_map.append(index)
            index += 1
        sentences_tokens.append(tokens)
        return all_tokens, tokens_map, all_words

    def fix_text_for_model(self, text):
        if self.model == "BERT" or self.model == "DistilBERT":
            return "[CLS] " + text + " [SEP]"
        return text

    """
    '' METHOD: read_terms
    '' PARAMETERS: 
    '''' dataset = dataset name
    '''' typ = type of data (train, test or validate)
    """

    def read_terms(self, dataset, typ, emb_tokenizer):
        x, tokens_maps, x_ppro, y, files_list, x_text, y_text, expected_kps = [], [], [], [], [], [], [], []
        f_text, list_keyphr = [], []
        path = "datasets/%s/%s" % (dataset, typ)
        kp20tokenizer = Kp20kTokenizer(self.mweTokenizer, emb_tokenizer)

        for f in os.listdir(path):
            "------------------ HULTH DATASET----------------------------------------------------------"
            if dataset == "Hulth2003":
                if not f.endswith(".uncontr"):
                    continue
                f_uncontr = open(os.path.join(path, f), "rU")
                f_text = open(os.path.join(path, f.replace(".uncontr", ".abstr")), "rU")
                text = "".join(map(str, f_text))
                text = self.fix_text_for_model(text)
                kp_uncontr = "".join(map(str, f_uncontr))

                list_keyphr = kp_uncontr.replace("\n\t", " ").split("; ")
                tokenized_keyphr = [emb_tokenizer.tokenize(kp) for kp in list_keyphr]

                doc = text.replace("\n\t", " ").split("\n")  # separo titolo e contenuto
                doc[0] = Reader.tokenizer.sentence_tokenize(doc[0])
                doc[1] = Reader.tokenizer.sentence_tokenize(doc[1])
                list_sentences = Reader.tokenizer.sentence_tokenize(text)

                list_sentences = self.title_fix(list_sentences)

                text_vec, sentences_vec, tokens_map, word_doc = self.tokenize_with_map_to_origin(list_sentences,
                                                                                                 emb_tokenizer)

                files_list.append(f)
                x.append(text_vec)
                x_text.append(word_doc)
                tokens_maps.append(tokens_map)
                x_ppro.append(sentences_vec)
                exp_val, exp_kps = self.calc_expected_values(text_vec, tokenized_keyphr, list_keyphr)
                y.append(exp_val)
                expected_kps.append(exp_kps)

            "------------------ Kp20k DATASET----------------------------------------------------------"

            if dataset == "Kp20k":
                file_name = ''.join(['kp20k_', typ.lower(), '.json'])
                with open(os.path.join(path, file_name)) as f:
                    count_doc = 1
                    for line in f:
                        if count_doc <= 20000:  # needed because 500k is too large
                            d = json.loads(line)
                            text = ''.join([d["title"], d["abstract"]])
                            text = self.fix_text_for_model(text)

                            kp_list = d["keyword"]
                            y_text.append(kp_list)
                            kp_list = kp_list.split(";")

                            list_keyphr = [kp20tokenizer.tokenize(kp) for kp in kp_list]
                            list_sentences = Reader.tokenizer.sentence_tokenize(text)

                            text_vec, sentences_vec, tokens_map, word_doc = self.tokenize_with_map_to_origin(
                                list_sentences, kp20tokenizer)

                            files_list.append(str(count_doc))
                            x.append(text_vec)
                            x_text.append(word_doc)
                            x_ppro.append(sentences_vec)
                            tokens_maps.append(tokens_map)
                            exp_val, exp_kps = self.calc_expected_values(text_vec, list_keyphr, kp_list)
                            y.append(exp_val)
                            expected_kps.append(exp_kps)
                            count_doc = count_doc + 1

            "------------------ Krapivin 2009 DATASET------------------------------------------------------"
            if dataset == "Krapivin2009":
                if not f.endswith(".key"):
                    continue
                f_key = open(os.path.join(path, f), "rU")
                f_text = open(os.path.join(path, f.replace(".key", ".txt")), "rU")
                text = "".join(map(str, f_text))
                text = self.fix_text_for_model(text)
                key_phrases = "".join(map(str, f_key))
                key_phrases = key_phrases.strip('\n')

                list_keyphr = [Reader.tokenizer.word_tokenize(kp) for kp in key_phrases.split("\n")]
                list_sentences = Reader.tokenizer.sentence_tokenize(text)

                text_vec = []
                for string in list_sentences:
                    for token in Reader.tokenizer.word_tokenize(string):
                        text_vec.append(token)
                files_list.append(f)
                x.append(text_vec)
                exp_val, exp_kps = self.calc_expected_values(text_vec, list_keyphr, key_phrases)
                y.append(exp_val)
                expected_kps.append(exp_kps)

            "---------------------------------------------------------------------------------------------"

        return x, tokens_maps, x_ppro, y, files_list, x_text, expected_kps

    """
    '' METHOD: calc_expected_values
    '' PARAMETERS: 
    '''' text_vec = vector of the words in text document
    '''' list_keyphr = list of document keyphrases
    """

    def calc_expected_values(self, text_vec, list_keyphr, kps):
        y_inner = np.zeros(np.shape(text_vec))

        f = lambda a, b: [x for x in range(len(a)) if a[x:x + len(b)] == b]
        found_kps = []
        for index, kp in enumerate(list_keyphr):
            arr_indices = f(text_vec, kp)  # returns the indices at which the pattern starts
            for i in arr_indices:
                if kps[index] not in found_kps:
                    found_kps.append(kps[index])
                y_inner[i] = 1
                if len(kp) > 1:
                    y_inner[(i + 1):(i + 1) + len(kp) - 1] = (2 if self.num_labels == 3 else 1)
        return y_inner, found_kps
