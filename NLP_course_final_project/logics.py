from nltk.corpus import brown


def trainingSet():

    print("/n/n************Training Set************/n/n")
    corpus = brown.tagged_sents(tagset='universal')[:-100]
    tag_dict = {}
    word_dict = {}

    for sent in corpus:
        for elem in sent:
            w = elem[0]
            tag = elem[1]
            if w not in word_dict:
                word_dict[w] = 0
            if tag not in tag_dict:
                tag_dict[tag] = 0
            word_dict[w] += 1
            tag_dict[tag] += 1

    test_data = brown.tagged_sents(tagset='universal')[-10:]
    return test_data, tag_dict, word_dict, corpus