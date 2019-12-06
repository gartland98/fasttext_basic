import torch
from random import shuffle
import random
from collections import Counter
import argparse
import os


def subsampling(word_seq):
    ###############################  Output  #########################################
    # subsampled : Subsampled sequence                                               #
    ##################################################################################
    coeff = 0.00001
    subsample_list = []
    for i in range(len(word_seq.keys())):
        word_fraction = torch.Tensor([word_seq[i] / sum(word_seq.values())])
        subsample= (torch.sqrt(word_fraction/ coeff) + 1) * (coeff / word_fraction)
        subsample_list.append(subsample)
    return subsample_list


def skipgram_fasttext(inputWords, inputMatrix, outputMatrix):
################################  Input  ##########################################
# centerWord : Index of a centerword (type:int)                                   #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################
    hidden_layer=0
    for i in inputWords:
        hidden_layer+=inputMatrix[i]
    hidden_layer = [hidden_layer]
    hidden_layer = torch.stack(hidden_layer)
    output_layer = torch.mm(hidden_layer, outputMatrix.t())
    e=torch.exp(output_layer)
    softmax=e/e.sum()
###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################

    loss=0
    for i in softmax:
        loss-=torch.log(i[-1])

    dsoftmax=softmax.clone()
    for prob in dsoftmax:
        prob[-1]-=1



    grad_out = torch.mm(dsoftmax.t(), hidden_layer)
    grad_in = torch.mm(dsoftmax, outputMatrix)

    return loss, grad_in, grad_out


def word2vec_trainer(input_seq, target_seq, ind2word,char2ind, num_of_ngram, numwords, stats, NS=20, dimension=100, learning_rate=0.025, epoch=3):
# train_seq : list(tuple(int, list(int))

# Xavier initialization of weight matrices
    W_in = torch.randn(num_of_ngram, dimension) / (dimension**0.5)
    W_out = torch.randn(numwords, dimension) / (dimension**0.5)
    i=0
    losses=[]
    print("# of training samples")
    print(len(input_seq))
    print()
    for _ in range(epoch):
        for word_inputs, word_output, ngram_inputs in zip(input_seq, target_seq, char2ind):#generate n-gramlist
            i += 1
            trigram = [''.join(i) for i in zip(ind2word[word_inputs][:-2],ind2word[word_inputs][1:-1],ind2word[word_inputs][2:])]
            quadgram = [''.join(i) for i in zip(ind2word[word_inputs][:-3],ind2word[word_inputs][1:-2], ind2word[word_inputs][2:-1],ind2word[word_inputs][3:])]
            fivegram = [''.join(i) for i in zip(ind2word[word_inputs][:-4], ind2word[word_inputs][1:-3], ind2word[word_inputs][2:-2], ind2word[word_inputs][3:-1], ind2word[word_inputs][4:])]
            sixgram = [''.join(i) for i in zip(ind2word[word_inputs][:-5], ind2word[word_inputs][1:-4], ind2word[word_inputs][2:-3], ind2word[word_inputs][3:-2], ind2word[word_inputs][4:-1], ind2word[word_inputs][5:])]
            trigram_index=[char2ind[i] for i in trigram]
            quadgram_index=[char2ind[i] for i in quadgram]
            fivegram_index=[char2ind[i] for i in fivegram]
            sixgram_index=[char2ind[i] for i in sixgram]
            input_index=[char2ind[ind2word[word_inputs]]]
            ngram_inputs=trigram_index+quadgram_index+fivegram_index+sixgram_index+input_index
            negtable = [i for i in stats if i != word_output][:NS]
            negtable.append(word_output)
            activated = torch.LongTensor(negtable)
            L, G_in, G_out = skipgram_fasttext(ngram_inputs, W_in, W_out[activated])
            W_in[ngram_inputs] -= learning_rate * G_in
            W_out[activated] -= learning_rate * G_out
            losses=[]
            losses.append(L.item())
            if i % 1000 == 0:
                avg_loss = sum(losses) / len(losses)
                print("Loss %d : %f" % (i,avg_loss))

        return W_in, W_out


def sim(testword, char2ind, ind2char, matrix):
    length = (matrix * matrix).sum(1) ** 0.5
    wi = char2ind[testword]
    inputVector = matrix[wi].reshape(1, -1) / length[wi]
    sim = torch.mm(inputVector, matrix.t())[0] / length
    values, indices = sim.squeeze().topk(5)

    print()
    print("===============================================")
    print("The most similar words to \"" + testword + "\"")
    for ind, val in zip(indices, values):
        print(ind2char[ind.item()] + ":%.3f" % (val,))
    print("===============================================")
    print()



def main():
    parser = argparse.ArgumentParser(description='fasttext')
    parser.add_argument('part', metavar='partition', type=str,
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    args = parser.parse_args()
    part = args.part

    # Load and preprocess corpus
    print("loading...")
    if part == "part":
        text = open('text8', mode='r').readlines()[0][:1000000]  # Load a part of corpus for debugging
    elif part == "full":
        text = open('text8', mode='r').readlines()[0]  # Load full corpus for submission
    else:
        print("Unknown argument : " + part)
        exit()

    print("preprocessing...")
    corpus = text.split()
    corpus_revised=[]
    for i in corpus:
        corpus_revised.append('<'+i+'>')
    stats = Counter(corpus_revised)

    words= []
    #Discard rare words
    for word in corpus_revised:
        if stats[word]>4:
            words.append(word)

    freqtable = []
    vocab = set(words)
    vocabs=[]
    for i in vocab:
        vocabs.append(i)
    trigram = [''.join(i) for j in vocab for i in zip(j[:-2], j[1:-1], j[2:])]
    quadgram=[''.join(i) for j in vocab for i in zip(j[:-3],j[1:-2],j[2:-1],j[3:])]
    fivegram=[''.join(i) for j in vocab for i in zip(j[:-4],j[1:-3],j[2:-2],j[3:-1],j[4:])]
    sixgram=[''.join(i) for j in vocab for i in zip(j[:-5],j[1:-4],j[2:-3],j[3:-2],j[4:-1],j[5:])]

    #Give an index number to a word
    w2i = {}
    w2i[" "]=0
    i = 1
    for word in vocab:
        w2i[word] = i
        i+=1
    i2w = {}
    for k,v in w2i.items():
        i2w[v]=k

    c2i = {}
    c2i[" "] = 0
    i = 1
    for char in trigram+quadgram+fivegram+sixgram+vocabs:
        if char not in c2i:
            c2i[char] = i
            i += 1
    i2c = {}
    for k, v in c2i.items():
        i2c[v] = k

    freqdict = {}
    freqdict[0] = 10
    for word in vocab:
        freqdict[w2i[word]] = stats[word]

    #Frequency table for negative sampling
    for k,v in stats.items():
        f = int(v**0.75)
        for _ in range(f):
            if k in w2i.keys():
                freqtable.append(w2i[k])

    #print("build training set1...")

    input_set = []
    target_set = []
    window_size = 5

    ########################################################## subsampling code ##########################################################
    # I just change this part of code as comment because it takes a long time. But according to the paper subsampling improve accuracy of analogy task
    # Therefore if you want high accuracy I recommend you to use this part of code as well
    # subsampling function is in the upper part of this function
    #subsample_class=[]
    #for j in range(len(freqdict)):
        #subsample_classify = [1 for _ in range(int(subsampling(freqdict)[j] * 1000))] + [0 for _ in range(1000 - int(subsampling(freqdict)[j] * 1000))]
        #subsample_class.append(subsample_classify)
        #if len(subsample_class) % 50==0:
            #print(len(subsample_class))
    #############################################################################################################################################


    print('build training set2...')


    for j in range(len(words)):
        ################# subsampling ########################
        #choice = random.choice(subsample_class[w2i[words[j]]])
        #if choice==1:
        ########################################################
        if j < window_size:
            input_set += [w2i[words[j]] for _ in range(window_size * 2)]
            target_set += [0 for _ in range(window_size - j)] + [w2i[words[k]] for k in range(j)] + [w2i[words[j + k + 1]] for k in range(window_size)]
        elif j >= len(words) - window_size:
            input_set += [w2i[words[j]] for _ in range(window_size * 2)]
            target_set += [w2i[words[j - k - 1]] for k in range(window_size)] + [w2i[words[len(words) - k - 1]] for k in range(len(words) - j - 1)] + [0 for _ in range(j + window_size - len(words) + 1)]
        else:
            input_set += [w2i[words[j]] for _ in range(window_size * 2)]
            target_set += [w2i[words[j - k - 1]] for k in range(window_size)] + [w2i[words[j + k + 1]] for k in range(window_size)]

    print("Vocabulary size")
    print(len(w2i))
    print()

    #Training section
    emb,_ = word2vec_trainer(input_set, target_set,i2w,c2i,len(c2i),len(w2i), freqtable, NS=20, dimension=64, epoch=1, learning_rate=0.01)
    testwords = ['<department>','<knowing>','<urbanize>','<imperfection>']
    for tw in testwords:
        sim(tw, c2i, i2c, emb)

main()
