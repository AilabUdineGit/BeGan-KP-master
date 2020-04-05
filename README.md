# Keyphrase-GAN with Bert Discriminator
We are implementing a GAN architecture to generate high-quality Keyphrases (KPs) from scientific abstracts. Our work rely much on the paper <a href="https://arxiv.org/abs/1909.12229">Keyphrase Generation for Scientific Articles using GANs</a> wich, in turn, uses the Generator from <a href = "https://github.com/kenchan0226/keyphrase-generation-rl"> keyphrase-generation-rl </a> and <a href = "https://github.com/memray/seq2seq-keyphrase-pytorch"> seq2seq-keyphrase-pytorch </a> .
Our contribution is the introduction of a Bert Discriminator.

## Dataset and Data Preprocessing
kp20k dataset is used to train both Generator (G) and Discriminator (D). Dataset is composed by +500k sample documents, each consisting of a scientific abstracts and the related KPs.
To preprocess the data run 
```terminal
python3 preprocess.py -data_dir data/kp20k_separated -remove_eos -include_peos -sample_size 2000
```
The attribute `_separated` means that present KPs are separated from the absent or abstract ones. Separation is performed useing the special token `<peos>`. Default value for `-sample_size = 1000`.

## Bert model for Discriminator
Different Bert models from <a href = https://github.com/huggingface/transformers> Huggingface Transformers </a> have been tested.
Current used model is <a href = https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification> BertForSequenceClassification </a>. It can be used to perform both classification passing it an `option.num_labels >= 2`, or regression with `option.num_labels = 1`. Current model in use is a regressor so `option.num_labels = 1`.

Some changes have been made in the structure, notably the output to classify the sample is not retrieved from the [CLS] token but from an average of _all_ the input tokens (cfr. <a href = https://huggingface.co/transformers/model_doc/bert.html#bertmodel> BertModel </a>: `pooler_output` in the `Returns` section) 

![Alt text](Images/bert-sentence-pair.png?raw=true "Bert Discriminator")

### Bert input
In our architecture the Discriminator perform the task to split the real KPs (generated by humans) from the fake ones (generated by the last version of Generator), for each sample document.
In order to be used as Bert input, samples has to be Bert-tokenized. Here for more details about <a href = https://github.com/google-research/bert#tokenization > Bert tokenization </a>.
The final pattern is:
```terminal
[CLS] <abstract> [SEP] <KP1> <KP2> ... <KPn> [SEP]
```
where `<abstract>` are the tokens of the abstract and `<KPi>` are those of the _i-th_ KP. For each abstract, two input sequences as above are built: one with the real KPs, and one with the fake ones. Then the model is trained with both real and fake sequences, each with the appropriate label: 0 for fake, 1 for real.

### Training of Discriminator
Be sure the option `-use_bert_discriminator` in config.py is set to True, then run
```terminal
python GAN_Training.py  -data data/kp20k_separated/ -vocab data/kp20k_separated/ -exp_path exp/%s.%s -exp kp20k -epochs 5 -copy_attention -train_ml -one2many -one2many_mode 1 -batch_size 4 -model [MLE_model_path] -train_discriminator 
```
Consider a batch_size of only 4 as Bert models are very heavy.

### Bert output
A tuple of tensors is returned as output:
- output[0]: classification or regression loss, float tensor of shape `(1,)`, only returned if _labels_ are provided in input;
- output[1]: classification or regression scores, float tensor of shape `(batch_size, config.num_labels)`;
- output[2]: hidden states, tuple of float tensors (one for each layer) of shape `(batch_size, sequence_length, hidden_size)`.

Current values are: `config.num_labels = 1` (regression), `batch_size = 4`, `sequence_length = 320`, `hidden_size = 768`.


***********************************************************************

From here on, the readme of the original project

# Keyphrase-GAN
This repository contains the code for the paper <a href="https://arxiv.org/abs/1909.12229">Keyphrase Generation for Scientific Articles using GANs</a>.We have built a novel adversarial method to improve upon the generation of keyphrases using supervised approaches.Our Implementation is built on the starter code from <a href = "https://github.com/kenchan0226/keyphrase-generation-rl"> keyphrase-generation-rl </a> and <a href = "https://github.com/memray/seq2seq-keyphrase-pytorch"> seq2seq-keyphrase-pytorch </a> . Pls comment any issues in the issues section.

![Alt text](Images/Discriminator.jpg?raw=true "Schematic of Proposed Discriminator")
## Dependencies 



## Adversarial Training
First start by creating a virtual environment and install all required dependencies.
```terminal
pip install virtualenv
virtualenv mypython
pip install -r requirements.txt
source mypython/bin/activate
```

### Data 
The GAN model is trained on close to 500000 examples of the kp20k dataset and evaluated on the Inspec (Huth) , Krapivin , NUS , Semeval Datasets . After Downloading this repo , create a `Data` folder within it . Download all the required datasets from [this](https://drive.google.com/open?id=1DbXV1mZXm_o9bgfwPV9PV0ZPcNo1cnLp) and store it in the `Data` folder . The Folders with `_sorted` suffix contain present keyphrases which are sorted in the order of there occurence , and the ones with `_seperated` suffix contains present and absent keyphrases seperated by a `<peos>` token . In order to preprocess the kp20k dataset , run 
```terminal
python3 preprocess.py -data_dir data/kp20k_sorted -remove_eos -include_peos
```

If you cant preprocess and want to temporarily run the repository , to can download the datasets with 10000 examples [here](https://drive.google.com/drive/folders/1YIJOAAR8rK8oiAfPK-5aJwgwlmw0uie_?usp=sharing) .

### Training the MLE model 
The first step in GAN training involves training the MLE model as a baseline using maximum likelihood loss . The paper has used CatSeq model as a baseline . In order to train Catseq model without copy attention run
```terminal
python3 train.py -data data/kp20k_sorted/ -vocab data/kp20k_sorted/ -exp_path exp/%s.%s -exp kp20k -epochs 25 -train_ml -one2many -one2many_mode 1 -batch_size 32
```
or with copy attention run
```terminal
python3 train.py -data data/kp20k_sorted/ -vocab data/kp20k_sorted/ -exp_path exp/%s.%s -exp kp20k -epochs 25 -train_ml -one2many -one2many_mode 1 -batch_size 32 -copy_attention
```

Note Down the Checkpoints Location while training .

### Training the Discriminator 

Now that the baseline MLE model is trained we need to train the Discriminator using the MLE model as Generator. The Discriminator is a hierarchal blstm which uses attention mechanism to calculate embeddings for all the keyphrases.

```terminal
python GAN_Training.py  -data data/kp20k_sorted/ -vocab data/kp20k_sorted/ -exp_path exp/%s.%s -exp kp20k -epochs 5 -copy_attention -train_ml -one2many -one2many_mode 1 -batch_size 32 -model [MLE_model_path] -train_discriminator 
```

All additional flags have been detailed at the end of the repository.

### Reinforcement Learning 
As Discriminator Gradients cannot directly backpropagate towards the Generator because of the Discrete Nature of text the Generator is trained by means of policy gradient reinforcement learning techniques . In order to train using RL run

```terminal
 python GAN_Training.py -data data/kp20k_sorted/ -vocab data/kp20k_sorted/ -exp_path exp/%s.%s -exp kp20k -epochs 20 -copy_attention -train_ml -one2many -one2many_mode 1 -batch_size 32 -model [model_path]  -train_rl   -Discriminator_model_path [Discriminator_path]
```

### Training Options
```
-D_hidden_dim : set hidden dimensions of Discriminator LSTM
-D_layers : set no.of layers in each LSTM in the Discriminator
-D_embedding_dim : No.of embedding dimensions to be used in the Discriminator 
-pretrained_Discriminator : supply a pretrained Discriminator in 2nd or later iterations of GAN Training.
-Discriminator_model_path : path to pretrained Discriminators
-learning_rate : Sets learning rate for Discriminator when used with -train_discriminator 
-learning_rate_rl : Sets learning rate for Generator during RL Training
```
cite our paper as 
```
@misc{swaminathan2019keyphrase,
    title={Keyphrase Generation for Scientific Articles using GANs},
    author={Avinash Swaminathan and Raj Kuwar Gupta and Haimin Zhang and Debanjan Mahata and Rakesh Gosangi and Rajiv Ratn Shah},
    year={2019},
    eprint={1909.12229},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
