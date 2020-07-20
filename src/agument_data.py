'''
Project: RNN model, that predict femanetives word by given male profession word (russian)
Author: Moscowsky Anton moscowskyad@gmail.com
File: Data augumention (idea from Michail Surtsukov)
Date created: 12.01.2020
'''
import numpy as np
import pandas as pd

#FILE_IN = '../data_professions/professions_train.csv'
#FILE_OUT = '../data_professions/professions_train_agumented2.csv'
FILE_IN = '../data_professions/professions_train_agumented3.csv'
FILE_OUT = '../data_professions/professions_train_agumented3.csv'

list_con = ['ц', 'к', 'н', 'г', 'ш', 'щ', 'з', 'х', 'ф', 'в', 'п', 'р', 'л', 'д', 'ж', 'ч', 'с', 'м', 'т', 'б']
list_vow = ['у', 'е', 'ы', 'а', 'о', 'э', 'я', 'и', 'ю', 'ё']

def find_max_aoi(wordM, wordF):
    for i, (m, f) in enumerate(zip(list(wordM), list(wordF))):
        if m != f:
            return i
    return i

def change_con(wordM, wordF):
    #aoi = int(len(wordM)/2)
    aoi = find_max_aoi(wordM, wordF)
    if aoi == 0:
        return False, 0, None
    t_i = np.random.randint(0,aoi)
    t_l = wordM[t_i]
    if t_l != wordF[t_i]:
        return False, 1, None
    if t_l in list_con:
        t_c = np.random.choice(list_con)
        if t_c == t_l:
            return False, 2, None
        wordM_ = list(wordM)
        wordF_ = list(wordF)
        wordM_[t_i] = t_c
        wordF_[t_i] = t_c
        return True, "".join(wordM_), "".join(wordF_)
    return False, 3, None

def change_vow(wordM, wordF):
    #aoi = int(len(wordM)/2)
    aoi = find_max_aoi(wordM, wordF)
    if aoi == 0:
        return False, 0, None
    t_i = np.random.randint(0,aoi)
    t_l = wordM[t_i]
    if t_l != wordF[t_i]:
        return False, 1, None
    if t_l in list_vow:
        t_c = np.random.choice(list_vow)
        if t_c == t_l:
            return False, 2, None
        wordM_ = list(wordM)
        wordF_ = list(wordF)
        wordM_[t_i] = t_c
        wordF_[t_i] = t_c
        return True, "".join(wordM_), "".join(wordF_)
    return False, 3, None
    
def cut_first_sym(wordM, wordF):
    wordM_ = list(wordM)
    wordF_ = list(wordF)
    wordM_ = "".join(wordM_[1:])
    wordF_ = "".join(wordF_[1:])
    return wordM_, wordF_

def main():
    target_N = 10000
    min_len = 5
    clear_data = pd.read_csv(FILE_IN)
    
    clear_x = clear_data['male'].values.tolist()
    clear_y = clear_data['female'].values.tolist()
    #clear = zip(clear_x, clear_y)
    
    agumented_x = []
    agumented_y = []
    cnt = 0
    super_break = False
    while len(agumented_x) < target_N:
        np.random.seed( cnt )
        for wordM, wordF in zip(clear_x, clear_y):
            res, wordM_t, wordF_t = change_con(wordM, wordF)
            if res:
                if wordM_t not in agumented_x and wordF_t not in agumented_y:
                #if True:
                    agumented_x.append(wordM_t)
                    agumented_y.append(wordF_t)
                    print("{}: {}/{}".format(cnt,len(agumented_x),target_N))
                    if( len(agumented_x) == target_N):
                        super_break = True
                        break
            #else:O
                #print("con {}".format(wordM_t))
                    
        for wordM, wordF in zip(clear_x, clear_y):
            res, wordM_t, wordF_t = change_vow(wordM, wordF)
            if res:
                if wordM_t not in agumented_x and wordF_t not in agumented_y:
                #if True:
                    agumented_x.append(wordM_t)
                    agumented_y.append(wordF_t)
                    print("{}: {}/{}".format(cnt, len(agumented_x),target_N))
                    if( len(agumented_x) == target_N):
                        super_break = True
                        break
            #else:
                #print("vow {}".format(wordM_t))
        for wordM, wordF in zip(clear_x, clear_y):
            if len(wordM) < min_len:
                continue
            wordM_t, wordF_t = cut_first_sym(wordM, wordF)
            if wordM_t not in agumented_x and wordF_t not in agumented_y:
                agumented_x.append(wordM_t)
                agumented_y.append(wordF_t)
                print("{}: {}/{}".format(cnt, len(agumented_x),target_N))
                if( len(agumented_x) == target_N):
                    super_break = True
                    break
                
        if(super_break):
            break
        #print(cnt)
        cnt+=1
        
    print("Saving data to {} ...".format(FILE_OUT))
    df = pd.DataFrame(list(zip(agumented_x,agumented_y)),columns=['male','female'])
    df.to_csv(FILE_OUT)
 
if __name__ == '__main__':
    main()
