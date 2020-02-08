# Inspired by https://machinelearningmastery.com/develop-character-based-neural-language-model-keras/
import re

#azbuka = ['ц', 'к', 'н', 'г', 'ш', 'щ', 'з', 'х', 'ф', 'в', 'п', 'р', 'л', 'д', 'ж', 'ч', 'с', 'м', 'т', 'б', 'у', 'е', 'ы', 'а', 'о', 'э', 'я', 'и', 'ю', 'ё']


# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
 
def main():
    # load text
    raw_text = load_doc('../data_embeddings/War_and_peace_I.txt')
    #print(raw_text)
    
    # clean
    tokens = re.findall(r"[а-я]+", raw_text.lower())
    # and start word symbol
    for i in range(len(tokens)):
        tokens[i] = '\t' + tokens[i] + '\r'
        #print(tokens[i])
    
    # generate sequences
    sequences = []
    len_seq = 4
    for w in tokens:    
        #print(w)
        for i in range(len_seq, len(w)):
            seq = w[i-len_seq:i+1]
            #print(seq)
            sequences.append(seq)


    print('Total Sequences: {}'.format(len(sequences)))

    # save sequences to file
    out_filename = '../data_embeddings/WaP_I_char_sequences.txt'
    save_doc(sequences, out_filename)

if __name__ == '__main__':
    main()
