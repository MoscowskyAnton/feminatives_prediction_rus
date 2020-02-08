'''
Project: RNN model, that predict feminatives word by given male profession word (russian)
Author: Moscowsky Anton moscowskyad@gmail.com
Inspiration: https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
Date created: 12.01.2020
'''
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Bidirectional, Concatenate, Dropout, Embedding
import keras.backend as kb
import numpy as np
import pandas as pd
from keras.utils import plot_model
import matplotlib.pyplot as plt
from trainingplot import TrainingPlot
from pickle import load

def get_chars(words, add_startend = False):    
    for i,word in enumerate(words):
        if add_startend:
            word = '\t' + word + '\n'
            words[i] = word        
    return words
    
def pretrained_embedding_layer(char_to_vec_map, char_to_index):    
    vocab_len = len(char_to_index) + 1
    
    emb_dim = char_to_vec_map["я"].shape[0]
    
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    for char, idx in char_to_index.items():
        emb_matrix[idx, :] = char_to_vec_map[char] 
        
    embedding_layer = Embedding(vocab_len, emb_dim, trainable = False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])    
    return embedding_layer

def words_to_indices(X, word_to_index, max_len):
    m = X.shape[0]
    X_indices = np.zeros((X.shape[0],max_len))    
    for i in range(m):
        sentence_words =X[i].lower().split()
        j = 0
        for w in sentence_words:            
            X_indices[i, j] = word_to_index[w]            
            j = j + 1
    return X_indices
    
def main():
    num_samples = -1
    batch_size = 32
    epochs = 10
    latent_dim = 265 # units in LSTM
    validation_split = 0.01
    data_train = pd.read_csv('../data_professions/professions_train.csv')
    data_test = pd.read_csv('../data_professions/professions_train.csv')

    input_words = data_train['male'].values[:num_samples].tolist()
    target_words = data_train['female'].values[:num_samples].tolist()
    test_words = data_test['male'].values.tolist()

    # load embeddings
    char_to_vec_map = load(open('../data_embeddings/char_to_vec_map.pkl', 'rb'))
    char_to_index = load(open('../data_embeddings/char_to_index.pkl', 'rb'))
    vocab_size = len(char_to_vec_map)
    print('Vocabulary Size: {}'.format(vocab_size))
    #print(char_to_index)
    #print(char_to_vec_map)
    #exit()

    # Vectorize the data
    #characters = set()    
    #characters.add('\0')
    input_words = get_chars(input_words)
    # adding a 'start char'    
    decoder_words = get_chars(target_words)
    target_words = get_chars(target_words, True)
    test_words = get_chars(test_words)
    #characters = sorted(list(characters))
    
    
    encoder_seq_length = max([len(txt) for txt in input_words+test_words])
    decoder_seq_length = max([len(txt) for txt in target_words])
    
    
    encoder_input_data = np.zeros((len(input_words),encoder_seq_length,vocab_size),dtype='float32')
    decoder_input_data = np.zeros((len(input_words),decoder_seq_length,vocab_size),dtype='float32')
    decoder_target_data = np.zeros((len(input_words),decoder_seq_length,vocab_size),dtype='float32')    
    encoder_input_data_test = np.zeros((len(test_words),encoder_seq_length,vocab_size),dtype='float32')
    
    for i, word in enumerate(input_words):
        seq = [char_to_index[char] for char in word]
        seq += [char_to_index['\n'] for i in range(len(seq),encoder_seq_length)]
        encoder_input_data[i,:,:] = to_categorical(np.array(seq), num_classes=vocab_size)
        
    for i, word in enumerate(decoder_words):
        seq = [char_to_index[char] for char in word]
        seq += [char_to_index['\n'] for i in range(len(seq),decoder_seq_length)]
        decoder_input_data[i,:,:] = to_categorical(np.array(seq), num_classes=vocab_size)
    
    for i, word in enumerate(target_words):
        seq = [char_to_index[char] for char in word]
        seq += [char_to_index['\n'] for i in range(len(seq),decoder_seq_length)]
        decoder_target_data[i,:,:] = to_categorical(np.array(seq), num_classes=vocab_size)
      
    for i, word in enumerate(test_words):
        seq = [char_to_index[char] for char in word]
        seq += [char_to_index['\n'] for i in range(len(seq),encoder_seq_length)]
        encoder_input_data_test[i,:,:] = to_categorical(np.array(seq), num_classes=vocab_size)
    
    
    '''
    vocab_size = len(characters)
    
    print("Number of unique characters: ",vocab_size)
    print("Max word lenght for inputs: ",encoder_seq_length)
    print("Max word lenght for ouputs: ",decoder_seq_length)
    
    token_index = dict([(char,i) for i,char in enumerate(characters)])
    #print(token_index) 
    
    encoder_input_data = np.zeros((len(input_words),encoder_seq_length,vocab_size),dtype='float32')
    decoder_input_data = np.zeros((len(input_words),decoder_seq_length,vocab_size),dtype='float32')
    decoder_target_data = np.zeros((len(input_words),decoder_seq_length,vocab_size),dtype='float32')
    
    encoder_input_data_test = np.zeros((len(test_words),encoder_seq_length,vocab_size),dtype='float32')
    
    for i, (input_word,target_word) in enumerate(zip(input_words,target_words)):
        for t, char in enumerate(input_word):
            encoder_input_data[i, t, token_index[char]] = 1.        
        for t in range(len(input_word),encoder_seq_length):
            encoder_input_data[i, t, token_index['\0']] = 1.        
        for t, char in enumerate(target_word):
            decoder_input_data[i, t, token_index[char]] = 1.
            if t > 0:
                #decoder target data will be ahead by one timestamp (idk why) and will not include the 'start char'
                decoder_target_data[i, t-1, token_index[char]] = 1.    
        
        for t in range(len(target_word),decoder_seq_length):
            decoder_input_data[i, t, token_index['\0']] = 1.
            decoder_target_data[i, t-1, token_index['\0']] = 1.         
    
    for i, test_word in enumerate(test_words):
        for t, char in enumerate(test_word):
            encoder_input_data_test[i, t, token_index[char]] = 1.
        
        for t in range(len(test_word),encoder_seq_length):
            encoder_input_data_test[i, t, token_index['\0']] = 1.        
        
    '''
    # define the input ans process
    encoder_inputs = Input(shape=(None, vocab_size))   
    
    embedding_layer = pretrained_embedding_layer(char_to_vec_map, char_to_index)
    
    encoder1 = Bidirectional(LSTM(latent_dim,return_sequences=True))
    encoder = Bidirectional(LSTM(latent_dim, return_state=True))
    encoder_outputs, state_h_f, state_c_f, state_h_b, state_c_b = encoder(Dropout(0.25)(encoder1(embedding_layer(encoder_inputs))))
    # discard output, leave state only
    state_h = Concatenate()([state_h_f, state_h_b])
    state_c = Concatenate()([state_c_f, state_c_b])
    encoder_states = [state_h, state_c]
    
    # decoder
    decoder_inputs = Input(shape=(None,vocab_size))
    
    decoder_lstm1 = LSTM(latent_dim * 2, return_sequences=True, return_state=True)
    decoder_lstm = LSTM(latent_dim * 2, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_lstm1(embedding_layer(decoder_inputs), initial_state=encoder_states))
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # define model!
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    plot_model(model,to_file='../model_images/model_graph.png')
    
    acc = str('categorical_accuracy')
    plot_losses = TrainingPlot(acc)
    
    # Train!
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[acc])
    history = model.fit([encoder_input_data, decoder_input_data],
              decoder_target_data, 
              batch_size=batch_size,
              epochs=epochs,
              shuffle = True,
              validation_split=validation_split,
              callbacks = [plot_losses])
    
    print(history.history.keys())
    '''
    plt.figure(0)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train','Val'], loc='best')
    plt.pause(1)
    
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train','Val'], loc='best')
    plt.pause(1)
    '''
    #model.save('DeepFemenistic.h5')
    
    # define sampling model
    encoder_model = Model(encoder_inputs, encoder_states)
    plot_model(encoder_model,to_file='../model_images/encoder_model_graph.png')
    
    decoder_state_input_h = Input(shape=(latent_dim*2,))
    decoder_state_input_c = Input(shape=(latent_dim*2,))
    decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
    #decoder_outputs, state_h_f, state_c_f, state_h_b, state_c_b = decoder_lstm(decoder_inputs, initial_state = decoder_state_inputs)
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_lstm1(decoder_inputs, initial_state = decoder_state_inputs))
    #state_h = Concatenate()([state_h_f, state_h_b])
    #state_c = Concatenate()([state_c_f, state_c_b])
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)
    
    plot_model(decoder_model,to_file='../model_images/decoder_model_graph.png')
    
    reverse_char_index = dict((i, char) for char, i in token_index.items())
    
    def decode_word(input_word):
        states_value = encoder_model.predict(input_word)
        
        target_seq = np.zeros((1,1,vocab_size))
        target_seq[0,0,token_index['\t']] = 1.
        
        stop_condition = False
        decoded_word = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
            
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_char_index[sampled_token_index]
            decoded_word += sampled_char
            
            if(sampled_char == '\n' or len(decoded_word) > decoder_seq_length):
                stop_condition = True
                
            target_seq = np.zeros((1,1, vocab_size))
            target_seq[0,0,sampled_token_index] = 1.
            
            states_value = [h, c]
        
        return decoded_word
    
    print("++ Part of train set ++")
    for ind in range(int(len(input_words)*(1-2*validation_split)),int(len(input_words)*(1-validation_split))):
        input_seq = encoder_input_data[ind:ind+1]
        decoded_word =decode_word(input_seq)                    
        print("{} -> {}".format(input_words[ind],decoded_word))    
    
    print("++ Validation set ++")
    for ind in range(int(len(input_words)*(1-validation_split)),len(input_words)):
        input_seq = encoder_input_data[ind:ind+1]
        decoded_word =decode_word(input_seq)                    
        print("{} -> {}".format(input_words[ind],decoded_word))
    
    print("++++++++ Test set +++++++++")
    for ind in range(len(test_words)):
        input_seq = encoder_input_data_test[ind:ind+1]
        decoded_word =decode_word(input_seq)                    
        print("{} -> {}".format(test_words[ind],decoded_word))
    
if __name__ == '__main__':
    main()
