from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import SnowballStemmer
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM
from keras.models import load_model
from gensim import corpora

np.random.seed(0)

train_data = pd.read_csv('./data/train.tsv', sep='\t', header=0)
test_data = pd.read_csv('./data/test.tsv', sep='\t', header=0)

train_phrase = train_data['Phrase'].values
test_phrase = test_data['Phrase'].values
sentiment = train_data['Sentiment'].values
num_labels = len(np.unique(sentiment))

stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
stemmer = SnowballStemmer('english')

if __name__ == "__main__":

    processed_train = []
    for doc in train_phrase:
       tokens = word_tokenize(doc)
       filtered = [word for word in tokens if word not in stop_words]
       stemmed = [stemmer.stem(word) for word in filtered]
       processed_train.append(stemmed)
    
    processed_test = []
    for doc in test_phrase:
       tokens = word_tokenize(doc)
       filtered = [word for word in tokens if word not in stop_words]
       stemmed = [stemmer.stem(word) for word in filtered]
       processed_test.append(stemmed)

    processed_docs = np.concatenate((processed_train, processed_test), axis=0)
    dictionary = corpora.Dictionary(processed_docs)
    dictionary_size = len(dictionary.keys())
    
#    train_words, test_words, words_length, word_ids  = [], [], [], []
#    test_words, word_ids = [], []
#    
#    for doc in processed_train:
#        word_ids = [dictionary.token2id[word] for word in doc]
#        train_words.append(word_ids)
#        words_length.append(len(word_ids))
#
#    for doc in processed_test:
#        word_ids = [dictionary.token2id[word] for word in doc]
#        test_words.append(word_ids)
#        words_length.append(len(word_ids))
# 
#    seq_len = np.round((np.mean(words_length) + 2*np.std(words_length))).astype(int)
#
#    train_words = sequence.pad_sequences(np.array(train_words), maxlen=seq_len)
#    test_words = sequence.pad_sequences(np.array(test_words), maxlen=seq_len)
#    y_train = np_utils.to_categorical(sentiment, num_labels)

#    model = Sequential()
#    model.add(Embedding(dictionary_size, 128))
#    model.add(Dropout(0.2))
#    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
#    model.add(Dense(num_labels))
#    model.add(Activation('softmax'))
#    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#    model.fit(train_words, y_train, epochs=3, batch_size=256, verbose=1)
#    
#    model.save("model.h5")
    
    # input data

    model = load_model("model.h5")
    
    sample = pd.read_csv('sample.csv',encoding = "ISO-8859-1")
    sample = sample['Phrase'].values

    processed_sample = []
    for doc in sample:
       tokens = word_tokenize(doc)
       filtered = [word for word in tokens if word not in stop_words]
       stemmed = [stemmer.stem(word) for word in filtered]
       processed_sample.append(stemmed)

    dictionary1 = dictionary.add_documents(processed_sample)
    
    word_id_sample = []
    word_ids = []
    
    for doc in processed_sample:
        word_ids = [dictionary.token2id[word] for word in doc]
        word_id_sample.append(word_ids)

    word_id_sample = sequence.pad_sequences(np.array(word_id_sample), maxlen=seq_len)
    test_pred = model.predict_classes(word_id_sample)