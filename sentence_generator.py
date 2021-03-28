import nltk
import random
import pickle
import os.path

def makePickle(dictionary, pickle_name):
    with open(pickle_name,'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('save dictionary to pickle...')

def openPickle(pickle_name):
    with open(pickle_name,'rb') as handle:
        dictionary_data = pickle.load(handle)
    return dictionary_data

def tokenize(doc):
    '''
    document lines to char sequence
    '''
    return [t for l in doc for t in l]

def generate(cpd, cfd_bi, _ngram):
    '''
    generate sentence function

    args
    cpd : user n-gram char Counter
    cfd_bi : bi-gram char Counter
    _ngram : user input / type is str
    '''
    c = "<s>"
    sentence = []
    while True:
        if c == '<s>': #첫 음절 생성
            eumjol = random.choice(cfd_bi['<s>'].most_common(3)) #문장 시작 기호 <s> 뒤에 가장 많이 출현한 3개를 나열하고, 이를 랜덤으로 추출합니다.
            c = (eumjol[0][0]) #eumjol = (음절: 빈도) 로 구성되어 있습니다.

            for i in range(3, int(_ngram)+1): #만약 3-gram이상이라면 앞에 <s>기호를 추가로 넣어줍니다.
                sentence.append('<s>')
                temp = list(c)
                temp.insert(0, '<s>')
                c = tuple(temp)
        else: #첫음절 이후 음절들 생성
            eumjol = random.choice(cpd[c].most_common(3)) #현재 c 이후에 가장 많이 등장하는 3개를 나열하고, 이를 랜덤으로 추출합니다.
            #n-gram마다 c에 들어가는 형상이 다르기 때문에, 이를 맞춰줍니다.
            if _ngram =='3':
                c = (sentence[-1], eumjol[0][0])
            elif _ngram == '4':
                c = (sentence[-2], sentence[-1], eumjol[0][0])
            elif _ngram == '5':
                c = (sentence[-3], sentence[-2], sentence[-1], eumjol[0][0])

        sentence.append(eumjol[0][0]) #최종적으로 선택된 음절을 sentence에 추가합니다.
        
        #문장 생성을 멈추는 부분입니다.
        if eumjol[0][0] == '</s>':
            break
        if len(sentence) >= 10 and eumjol[0][0] == '다':
            break

    return "".join(sentence)


if __name__ == "__main__":
    f = open('sample_test.txt','r')
    fline = f.readlines()

    _ngram = input('which n-gram(3-5) do you want? : ')

    assert int(_ngram) in (3,4,5), 'n-gram range should be 3-5'

    # 기존에 만들어 둔 n-gram 파일을 사용합니다.
    if os.path.isfile('2ngram.pickle'):
        cfd_bi = openPickle('2ngram.pickle')
        cfd = openPickle(_ngram+'ngram.pickle')

    else: #새로 n-gram파일을 생성합니다.
        sentences = []
        sentences_uni = []

        for i in fline: #bi-gram과 n-gram을 생성합니다
            token = tokenize(i)
            ngram = nltk.ngrams(token, int(_ngram), pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="</s>")
            sentences += [t for t in ngram]

            bigram = nltk.ngrams(token, 2,  pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="</s>")
            sentences_uni += [t for t in bigram]

        cfd_bi = nltk.ConditionalFreqDist(sentences_uni)
        cfd = nltk.ConditionalFreqDist()
        
        #bigram을 제외하고는 dictionary 저장 구조를 아래와 같이 별도로 지정해줘야 합니다.
        for i in sentences:
            if _ngram == '3':
                condition = (i[0], i[1])
                cfd[condition][i[2]] += 1
            elif _ngram =='4':
                condition = (i[0], i[1], i[2])
                cfd[condition][i[3]] += 1
            elif _ngram == '5':
                condition = (i[0], i[1], i[2],i[3])
                cfd[condition][i[4]] += 1
        
        #생성한 ngram을 저장합니다.
        makePickle(cfd_bi, 'bigram.pkl')
        makePickle(cfd, _ngram+'gram.pkl')

    #문장을 생성합니다.
    generate_sentence = generate(cfd, cfd_bi, _ngram)
    print(generate_sentence)