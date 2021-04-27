import torch
from flask import Flask, redirect, url_for, request, jsonify
# train.py
import io
import torch
import numpy as np
import pandas as pd
# from pymagnitude import Magnitude

# we use tensorflow but not for training model
import tensorflow as tf
from sklearn import metrics
import pickle
import json

global data, tokenizer, pred

# model.py
import torch
import torch.nn as nn


# load_vocab = pd.read_json('vocab.json')
# vocab_obj = Vocab(load_vocab)

app = Flask(__name__)




class LSTM(nn.Module):
  def __init__(self, embedding_matrix=0):
    """
    :param embedding_matrix: numpy array with vectors for all words
    """
    super(LSTM, self).__init__()
    # number of words = number of rows in embedding matrix
    #embedding_matrix.shape[0]
    num_words = 30192 
    # dimension of embedding is num of columns in the matrix
    #embedding_matrix.shape[1]
    embed_dim = 300 
    # we define an input embedding layer
    print("embedding",num_words, embed_dim)
    self.embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=embed_dim)
    # embedding matrix is used as weights of 
    # the embedding layer
    # self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
    # we dont want to train the pretrained embeddings
    # self.embedding.weight.requires_grad = False
    # a simple bidirectional LSTM with
    # hidden size of 128
    self.lstm = nn.LSTM(embed_dim,128,bidirectional=True,batch_first=True)
    # output layer which is a linear layer
    # we have only one output
    # input (512) = 128 + 128 for mean and same for max pooling
    self.out = nn.Linear(512, 1)

  def forward(self, x):
    # pass data through embedding layer
    # the input is just the tokens
    x = self.embedding(x)
    # move embedding output to lstm
    x, _ = self.lstm(x)
    # apply mean and max pooling on lstm output
    avg_pool = torch.mean(x, 1)
    max_pool, _ = torch.max(x, 1)
    
    # concatenate mean and max pooling
    # this is why size is 512
    # 128 for each direction = 256
    # avg_pool = 256 and max_pool = 256
    out = torch.cat((avg_pool, max_pool), 1)
    # pass through the output layer and return the output
    out = self.out(out)
    # return linear output
    return out


DEVICE = "cpu"



def load_vectors(fname):
    # Download GloVe vectors (uncomment the below)
    GLOVE_FILENAME = fname
    glove_index = {}
    n_lines = sum(1 for line in open(GLOVE_FILENAME, encoding="utf8"))
    with open(GLOVE_FILENAME, encoding="utf8") as fp:
        for line in tqdm(fp, total=n_lines):
            split = line.split()
            word = split[0]
            vector = np.array(split[1:]).astype(float).tolist()
            glove_index[word] = vector
    print("total length index",len(glove_index))
    # with open("vocab.json", "w") as outfile: 
    #     json.dump(glove_index, outfile)
    # print("file saved")
    return glove_index






def create_embedding_matrix(word_index, embedding_dict):
  """
  This function creates the embedding matrix.
  :param word_index: a dictionary with word:index_value
  :param embedding_dict: a dictionary with word:embedding_vector
  :return: a numpy array with embedding vectors for all known words
  """
  # initialize matrix with zeros
  embedding_matrix = np.zeros((len(word_index) + 1, 300))
  # loop over all the words
  for word, i in word_index.items():
    # if word is found in pre-trained embeddings, 
    # update the matrix. if the word is not found,
    # the vector is zeros!
    if word in embedding_dict:
      embedding_matrix[i] = embedding_dict[word]
  # return embedding matrix
  return embedding_matrix


# def sentence_prediction(sentence):
#     # we use tf.keras for tokenization
#     # you can use your own tokenizer and then you can 
#     # get rid of tensorflow

    
#     MAX_LEN = 256
#     text_seq = tokenizer.texts_to_sequences([sentence])
#     text_seq_pad = tf.keras.preprocessing.sequence.pad_sequences(
#     text_seq, maxlen=MAX_LEN
#     )

#     description = torch.tensor(text_seq_pad, dtype=torch.long)

#     # embedding_dict = load_vectors("pretrained/glove.6B.300d.txt")
#     embedding_dict = data
#     embedding_matrix = create_embedding_matrix(
#     tokenizer.word_index, embedding_dict
#     )

#     device = torch.device('cpu')
#     if torch.cuda.is_available():
#       map_location=lambda storage, loc: storage.cuda()
#     else:
#       map_location='cpu'


#     model = LSTM(embedding_matrix)
#     # send model to device
#     model.load_state_dict(torch.load('./models/HNI_model_0.bin', map_location=map_location))
#     model.eval()

#     reviews = description.to(device, dtype=torch.long)
#     predictions = model(reviews)
#     # move predictions and targets to list
#     # we need to move predictions and targets to cpu too
#     print("pre",predictions)
#     predictions = predictions.detach().numpy().tolist()
#     print("predictions", predictions)


  


 
class Prediction:
  def __init__(self,tokenizer, glove_embedding=0):
    self.tokenizer = tokenizer
    # self.glove_embedding = glove_embedding

  def predict(self,sentence):
    MAX_LEN = 256
    text_seq = self.tokenizer.texts_to_sequences([sentence])
    text_seq_pad = tf.keras.preprocessing.sequence.pad_sequences(
    text_seq, maxlen=MAX_LEN
    )

    description = torch.tensor(text_seq_pad, dtype=torch.long)

    # embedding_dict = load_vectors("pretrained/glove.6B.300d.txt")
    # embedding_matrix = create_embedding_matrix(
    # tokenizer.word_index, self.glove_embedding
    # )

    device = torch.device('cpu')
    if torch.cuda.is_available():
      map_location=lambda storage, loc: storage.cuda()
    else:
      map_location='cpu'

    model = LSTM()
    # send model to device
    model.load_state_dict(torch.load('./models/HNI_model_0.bin', map_location=map_location))
    model.eval()
    with torch.no_grad():
      reviews = description.to(device, dtype=torch.long)
      predictions = model(reviews)
      # move predictions and targets to list
      # we need to move predictions and targets to cpu too
      print("pre",predictions)
      main_prediction = predictions.detach().numpy().tolist()[0][0]
      predictions = torch.round(torch.sigmoid(predictions))
      sigmoid_prediction = torch.sigmoid(predictions).detach().numpy().tolist()[0][0]
      predictions = predictions.detach().numpy().tolist()
      return int(predictions[0][0]),sigmoid_prediction,main_prediction




import json 

# with open('vocab.json', 'r') as f:
#   for line in f:
#     data = json.loads(line)



@app.route("/predict",methods=['POST'])
def generate():
    request_data=request.get_json()
    print("request",request_data)
    input = request_data['description']
    prediction,sigmoid_prediction,main_prediction = pred.predict(input)
    prediction_label = "Incident" if prediction==0 else "Request"
    return jsonify({
      'TICKET_TYPE':prediction_label,
      "TICKET_LABEL":prediction,
      "MODEL_PREDICTION":main_prediction,
      "SIGMOID":sigmoid_prediction
    })



if __name__ == '__main__':
  with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

  pred = Prediction(tokenizer)

  app.run()







    



