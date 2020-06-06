from Data_Preprocessing import *
from urllib.parse import unquote_plus
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding



#NEURAL NETWORK CREATION BEGINS

#Hot embedding of vocabulary set and other parameter initialization
vocab_size=len(tk.word_index)
input_size = 1014
embedding_size = 69
embedding_weights = []  
embedding_weights.append(np.zeros(vocab_size))  
for char, i in tk.word_index.items(): 
    onehot = np.zeros(vocab_size)
    onehot[i - 1] = 1
    embedding_weights.append(onehot)
embedding_weights = np.array(embedding_weights)
fully_connected_layers = [1024, 1024]

#Character Embedding begins

#Initializing input layer
inputs = Input(shape=(input_size,), name='input', dtype='int64')

#Initializing embedding layer and input layer is passed onto this layer
embedding_layer = Embedding(vocab_size + 1,
                            embedding_size,
                            input_length=input_size,
                            weights=[embedding_weights])
x = embedding_layer(inputs)

#Character Embedding ends

#Feature Extraction begins

#Initializing conv_filter_size, conv_window_size, pooling_window_size
conv_layers = [[256, 7, 3],
               [256, 7, 3],
               [256, 3, -1],
               [256, 3, -1],
               [256, 3, -1],
               [256, 3, 3]]

#Initializing 6 convolution layers and 3 max-pooling layers and
#previous layers are passed onto subsequent layers
for filter_num, filter_size, pooling_size in conv_layers:
    x = Conv1D(filter_num, filter_size)(x)
    x = Activation('relu')(x)
    if pooling_size != -1:
        x = MaxPooling1D(pool_size=pooling_size)(x)
           
#Feature Extraction ends

#Classification begins
        
#Flattening the feature maps produced from max-pooling layer 
#and is passed onto dense layer
x = Flatten()(x)  

#initializing 2 hidden layers and a drop-out layer is 
#added to each hidden layer to avoid over-fitting
for dense_size in fully_connected_layers:
    x = Dense(dense_size, activation='relu')(x)  
    x = Dropout(0.5)(x)
    
#Initializing output layer
predictions = Dense(4, activation='softmax')(x)

#Classification ends

#Model instantiation
model = Model(inputs=inputs, outputs=predictions)

#Configuring the model for training
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Training the model
history=model.fit(train_data, y_train,
          validation_data=(test_data, y_test),
          batch_size=128,
          epochs=10,
          verbose=2)

#NEURAL NETWORK CREATION ENDS

##Graphical Representation of trained model 
#import matplotlib.pyplot as plt
#plt.style.use('ggplot')
#def plot_history(history):
#    acc = history.history['accuracy']
#    val_acc = history.history['val_accuracy']
#    loss = history.history['loss']
#    val_loss = history.history['val_loss']
#    x = range(1, len(acc) + 1)
#
#    plt.figure(figsize=(16, 10))
#    plt.subplot(1, 2, 1)
#    plt.plot(x, acc, 'b', label='Training acc')
#    plt.plot(x, val_acc, 'r', label='Validation acc')
#    plt.title('Training and validation accuracy')
#    plt.legend()
#    plt.subplot(1, 2, 2)
#    plt.plot(x, loss, 'b', label='Training loss')
#    plt.plot(x, val_loss, 'r', label='Validation loss')
#    plt.title('Training and validation loss')
#    plt.legend()

#plot_history(history)

##Saving the trained model
#model.save('model_final_version_2.h5')

##Testing the trained model
#test='http://www.krazl.com/blog/?p=77" or 1=1'
#test=unquote_plus(test)
#test=[s.lower() for s in test]
#test="".join(test)
#test=np.array([test])
#test=tk.texts_to_sequences(test)
#test=pad_sequences(test, maxlen=1014, padding='post')
#pred=model.predict(test)
#pred=saved_model.predict(test)  
#pred=(pred>0.5)
#
#if pred[0][0]==True:
#    print("The URL is Safe")
#elif pred[0][1]==True:
#    print("The URL is susceptible to Cross-site Scripting attack")
#elif pred[0][2]==True:
#    print("The URL is susceptible to Directory traversal attack")
#elif pred[0][3]==True:
#    print("The URL is susceptible to SQL injection attack")
    
