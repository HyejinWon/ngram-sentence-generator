import nltk
import pickle
import os.path
import numpy as np
from collections import Counter
from sentence_generator import openPickle, makePickle


def bigram(sentence):
    '''
    This function make char bigram sequence.
    (Did not used nltk bigram maker)

    Arguments:
    sentence - input one sentence
    
    Return:
    result - bigram sequence
    '''
    sentence = sentence #문장의 시작기호를 넣어줍니다.
    result = [(sentence[i], sentence[i+1]) for i in range(len(sentence)-1)]
    result.insert(0, ('<s>', sentence[0]))
    return result

def sentence_score(sentence, cpd, cpd_uni):
    ''' Compute sentence score
    When compute sentence score using n-gram feature, sentence probability can be zero.
    To handle this problem, I added a very small score.

    Argument:
    sentence - bigram sequence
    cpd - pre-trained bigram conditional frequency dictionary
    cpd_uni - pre-trained unigram frequency dictionary

    Return:
    p - sentence probability score
    '''
    p = 1
    for i in sentence:

        r = cpd[i[0]][i[1]] / cpd_uni[i[0]]
        if r == 0.0:
            r = 1e-7
        p *= r
        #print('word : ',i,'score : ',r)
    return p

def unigram_make():
    f = open('processed_wiki_ko.txt','r')
    fline = f.readlines()
    uni_counter = Counter()
    for i in fline:
        uni_counter += Counter(i)
    uni_counter['<s>'] = len(fline)

    return uni_counter

if __name__ == "__main__":
    #assignment 1
    
    user_input = input('put input sentence : ')
    bi_input = bigram(user_input)
    
    if os.path.isfile('1gram.pkl') == False:
        cfd_uni = unigram_make()
        makePickle(cfd_uni, '1gram.pkl')
        cfd_bi = openPickle('2gram.pkl')
    elif os.path.isfile('2gram.pkl') and os.path.isfile('1gram.pkl'):
        cfd_bi = openPickle('2gram.pkl')
        cfd_uni = openPickle('1gram.pkl')
    else:
        assert 0, 'You should make 2gram.pkl file from sentence_generator.py'

    score = sentence_score(bi_input, cfd_bi, cfd_uni)
    print('sentence score is : ',score)
    
    #assignment 2
    sentences = ['나는 밥을 좋아했다','노는 밥을 좋아했다','내는 밥을 좋아했다','누난 밥을 좋아했다','넌은 밥을 좋아했다','논은 밥을 좋아했다']
    print('sentence list : ', sentences)
    bi_sentences = [bigram(i) for i in sentences]
    score = [sentence_score(i, cfd_bi, cfd_uni) for i in bi_sentences] #sentences score

    sort_index = np.argsort(score)[::-1] # 스코어 점수 내림차순으로 정렬

    # sort score and sentence based sort_index
    sort_score = [score[index] for index in sort_index] 
    sort_sentence = [sentences[index] for index in sort_index]
    for sc, sent in zip(sort_score, sort_sentence):
        print(sc,' : ',sent)