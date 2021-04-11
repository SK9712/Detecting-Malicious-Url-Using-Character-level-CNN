import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


#Loading dataset
load_data = pd.read_csv('Data/URL.txt', header=None,sep='\t')

##Loading saved model
#saved_model=load_model('model_final_version_2.h5')

#Initializing data_values
data=load_data[0].values
data=[s.lower() for s in data]

#Initializing corresponding label for data_values
outcome=load_data[1].values

#Hot Embedding labels
outcome=to_categorical(outcome)

#Splitting the dataset into training and testing data
data_train, data_test, y_train, y_test = train_test_split(
        data, outcome, test_size=0.20, random_state=1000)

#Initializing tokenizer for character-level splitting
tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')

##Creating a vocabulary set based on training data
#tk.fit_on_texts(data_train)

#Creating a vocabulary set of 69 characters manually
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
char_dict = {}
for i, char in enumerate(alphabet):
    char_dict[char] = i + 1
tk.word_index = char_dict.copy()
tk.word_index[tk.oov_token] = max(char_dict.values()) + 1

#Converting characters of each training and 
#testing data observations to their corresponding values in vocabulary set
train_sequences = tk.texts_to_sequences(data_train)
test_sequences = tk.texts_to_sequences(data_test)

#Add padding to make each observations of training 
#and testing data to a fixed length input of 1014
train_data = pad_sequences(train_sequences, maxlen=1014, padding='post')
test_data = pad_sequences(test_sequences, maxlen=1014, padding='post')

#Converting each observation of training and testing data to arrays
train_data = np.array(train_data, dtype='float32')
test_data = np.array(test_data, dtype='float32')

