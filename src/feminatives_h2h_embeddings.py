'''
Project: RNN model, that predict femanetives word by given male profession word (russian)
Author: Moscowsky Anton moscowskyad@gmail.com
Inspiration: https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
Date created: 12.01.2020
'''

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Bidirectional, Concatenate, Dropout
import keras.backend as kb
import numpy as np
import pandas as pd
from keras.utils import plot_model
import matplotlib.pyplot as plt
from trainingplot import TrainingPlot
from pickle import load
from keras.losses import categorical_crossentropy

def get_chars(words, add_startend = False):    
    for i,word in enumerate(words):
        if add_startend:
            word = '\t' + word + '\n'
            words[i] = word
        
    return words

def apply_embeddings_input(words, char_to_vec_map, seq_len, shift = False):
    embeddings_len = len(list(char_to_vec_map.values())[0])
    data = np.zeros((len(words),seq_len,embeddings_len),dtype='float32')
    
    for i, word in enumerate(words):
        if shift:
            for j, letter in enumerate(list('\t'+word+'\n')):
                data[i,j,:] = char_to_vec_map[letter]
            for k in range(j, seq_len):
                data[i,j,:] = char_to_vec_map['\n']
        else:
            for j, letter in enumerate(list(word+'\n')):
                data[i,j,:] = char_to_vec_map[letter]
            for k in range(j, seq_len):
                data[i,j,:] = char_to_vec_map['\n']            
    
    return data
 
 
 
def my_categorical_crossentropy(y, y_prime):
    fy_ = np.exp(y_prime)/np.sum(np.exp(y_prime))
    return np.sum( y * np.log(fy_) )
              
    
def main():
    num_samples = -1
    batch_size = 32
    epochs = 100
    latent_dim = 265 # units in LSTM
    validation_split = 0.1
    data_train = pd.read_csv('../data_professions/professions_train.csv')
    data_test = pd.read_csv('../data_professions/professions_test.csv')

    input_words = data_train['male'].values[:num_samples].tolist()
    target_words = data_train['female'].values[:num_samples].tolist()
    test_words = data_test['male'].values.tolist()

    # load embeddings
    char_to_vec_map = load(open('../data_embeddings/char_to_vec_map.pkl', 'rb'))
    char_to_index = load(open('../data_embeddings/char_to_index.pkl', 'rb'))
    vocab_size = len(char_to_vec_map)
    print('Vocabulary Size: {}'.format(vocab_size))
    print(char_to_index)

    #exit()
    
    # Vectorize the data    
    
    
    #input_words = get_chars(input_words)
    # adding a 'start char'    
    #target_words = get_chars(target_words, True)
    #test_words = get_chars(test_words)
    #characters = sorted(list(characters))
    
    encoder_seq_length = max([len(txt) for txt in input_words+test_words]) + 1
    decoder_seq_length = max([len(txt) for txt in target_words]) + 2 # !!!
    num_tokens = len(list(char_to_vec_map.values())[0])
    
    #print("Number of unique characters: ",num_tokens)
    #print("Max word lenght for inputs: ",encoder_seq_length)
    #print("Max word lenght for ouputs: ",decoder_seq_length)
    
    #token_index = dict([(char,i) for i,char in enumerate(characters)])
    #print(token_index) 
    
    #encoder_input_data = np.zeros((len(input_words),encoder_seq_length,num_tokens),dtype='float32')
    #decoder_input_data = np.zeros((len(input_words),decoder_seq_length,num_tokens),dtype='float32')
    #decoder_target_data = np.zeros((len(input_words),decoder_seq_length,num_tokens),dtype='float32')
    
    #encoder_input_data_test = np.zeros((len(test_words),encoder_seq_length,num_tokens),dtype='float32')
    
    encoder_input_data = apply_embeddings_input(input_words, char_to_vec_map, encoder_seq_length)
    #print(encoder_input_data)
    #exit()
    encoder_input_data_test = apply_embeddings_input(test_words, char_to_vec_map, encoder_seq_length)
    decoder_input_data = apply_embeddings_input(target_words, char_to_vec_map, decoder_seq_length, True)
    decoder_target_data = apply_embeddings_input(target_words, char_to_vec_map, decoder_seq_length ) 
    '''
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
    encoder_inputs = Input(shape=(None, num_tokens))    
    encoder1 = Bidirectional(LSTM(latent_dim,return_sequences=True))
    encoder = Bidirectional(LSTM(latent_dim, return_state=True))
    encoder_outputs, state_h_f, state_c_f, state_h_b, state_c_b = encoder(Dropout(0.25)(encoder1(encoder_inputs)))
    # discard output, leave state only
    state_h = Concatenate()([state_h_f, state_h_b])
    state_c = Concatenate()([state_c_f, state_c_b])
    encoder_states = [state_h, state_c]
    
    # decoder
    decoder_inputs = Input(shape=(None,num_tokens))
    
    decoder_lstm1 = LSTM(latent_dim * 2, return_sequences=True, return_state=True)
    decoder_lstm = LSTM(latent_dim * 2, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_lstm1(decoder_inputs, initial_state=encoder_states))
    decoder_dense = Dense(num_tokens, activation='sigmoid') # softmax
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # define model!
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    plot_model(model,to_file='model_graph.png')
    
    acc = str('acc') # categorical_accuracy
    plot_losses = TrainingPlot(acc)
    
    # Train!
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=[acc])
    history = model.fit([encoder_input_data, decoder_input_data],
              decoder_target_data, 
              batch_size=batch_size,
              epochs=epochs,
              shuffle = True,
              validation_split=validation_split)
              #callbacks = [plot_losses])
    
    # categorical_crossentropy
    
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
    plot_model(encoder_model,to_file='encoder_model_graph.png')
    
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
    
    plot_model(decoder_model,to_file='decoder_model_graph.png')
    
    reverse_char_index = dict((i, char) for char, i in char_to_index.items())
    print(reverse_char_index)
    
    def decode_word(input_word):
        states_value = encoder_model.predict(input_word)
        
        target_seq = np.zeros((1,1,num_tokens))
        #target_seq[0,0,token_index['\t']] = 1.
        target_seq[0,0,:] = char_to_vec_map['\t']
        
        stop_condition = False
        decoded_word = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
            
            #sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char_emb = output_tokens[0, -1, :]
            
            min_d = float('inf')
            sampled_char = '\n'
            for index, emb_word in char_to_vec_map.items():
                #d = np.sum( np.power(sampled_char_emb - emb_word, 2) )
                #d = categorical_crossentropy(sampled_char_emb, emb_word, from_logits=False, label_smoothing=0)
                d = my_categorical_crossentropy(sampled_char_emb, emb_word)
                #print(sampled_char_emb, emb_word, d)
                if d < min_d:
                    min_d = d
                    sampled_char = index
            
            decoded_word += sampled_char
            
            if(sampled_char == '\n' or len(decoded_word) > decoder_seq_length):
                stop_condition = True
                
            target_seq = np.zeros((1,1, num_tokens))
            #target_seq[0,0,sampled_token_index] = 1.
            target_seq[0,0,:] = char_to_vec_map[sampled_char]
            
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
