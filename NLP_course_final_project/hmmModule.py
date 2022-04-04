import numpy as np
import nltk
import sklearn.metrics as sm
import logics


class hmmClinet:


    def __init__(self):

        self.viterbi = []
        self.brown_tags = None
        self.CPD_tag_words = None
        self.CFD_tags = None
        testD, tagD, wd, corpus = logics.trainingSet()
        self.test_data = testD
        self.tag_dict = tagD
        self.corpus = corpus
        self.word_dict = wd


    def learnTransition(self):

        print("/n/n************Learn Transition************/n/n")
        brown_tags_words = []
        for sent in self.corpus:
            brown_tags_words.append(("START", "START"))
            brown_tags_words.extend([(tag, word) for (word, tag) in sent])
            brown_tags_words.append(("END", "END"))

        print(brown_tags_words[0:30])

        # conditional frequency distribution of the word given the tags
        CFD_tag_words = nltk.ConditionalFreqDist(brown_tags_words)
        # conditional probability distribution of the word given the tags
        # P(wi | ti)
        self.CPD_tag_words = nltk.ConditionalProbDist(CFD_tag_words, nltk.MLEProbDist)

        self.brown_tags = [tag for (tag, word) in brown_tags_words]

        # make conditional frequency distribution: count(t{i-1} ti)
        CFD_tags = nltk.ConditionalFreqDist(nltk.bigrams(self.brown_tags))
        # make conditional probability distribution, using maximum likelihood estimate:
        # P(ti | t{i-1})
        self.CPD_tags = nltk.ConditionalProbDist(CFD_tags, nltk.MLEProbDist)

        tag_trans_matrix = []
        j = -1
        for tag1 in self.tag_dict:
            tag_trans_matrix.append([])
            j = j + 1
            for tag2 in self.tag_dict:
                tag_trans_matrix[j].append(self.CPD_tags[tag1].prob(tag2))
        print("State Transition matrix:")
        print(tag_trans_matrix)


    def Viterbi(self, sentence):

        print("/n/n************Viterbi************/n/n")
        sentlen = len(sentence)
        distinct_tags = set(self.brown_tags)

        backpointer = []

        viterbi_init = {}
        backpointer_init = {}

        for tag in distinct_tags:
            if tag == "START": continue
            viterbi_init[tag] = self.CPD_tags["START"].prob(tag) * self.CPD_tag_words[tag].prob(sentence[0])
            backpointer_init[tag] = "START"
        self.viterbi.append(viterbi_init)
        backpointer.append(backpointer_init)

        for wordindex in range(1, len(sentence)):
            cur_viterbi = {}
            cur_backpointer = {}
            prev_viterbi = self.viterbi[-1]

            for tag in distinct_tags:
                if tag == "START": continue
                if sentence[wordindex] not in self.word_dict.keys():
                    best_prevtag = max(prev_viterbi.keys(), key=lambda prevtag: \
                        prev_viterbi[prevtag] * self.CPD_tags[prevtag].prob(tag) * 0.0001)
                    cur_viterbi[tag] = prev_viterbi[best_prevtag] * \
                                       self.CPD_tags[best_prevtag].prob(tag) * 0.0001
                else:
                    best_prevtag = max(prev_viterbi.keys(), key=lambda prevtag: \
                        prev_viterbi[prevtag] * self.CPD_tags[prevtag].prob(tag) *
                        self.CPD_tag_words[tag].prob(sentence[wordindex]))
                    cur_viterbi[tag] = prev_viterbi[best_prevtag] * \
                                       self.CPD_tags[best_prevtag].prob(tag) * \
                                       self.CPD_tag_words[tag].prob(sentence[wordindex])
                cur_backpointer[tag] = best_prevtag

            self.viterbi.append(cur_viterbi)
            backpointer.append(cur_backpointer)

        prev_viterbi = self.viterbi[-1]
        best_prevtag = max(prev_viterbi.keys(), key=lambda prevtag: prev_viterbi[prevtag] * \
                                                                    self.CPD_tags[prevtag].prob("END"))

        prob_best_seq = prev_viterbi[best_prevtag] * self.CPD_tags[best_prevtag].prob("END")
        best_tag_seq = ["END", best_prevtag]

        backpointer.reverse()
        current_best_tag = best_prevtag
        for bp in backpointer:
            best_tag_seq.append(bp[current_best_tag])
            current_best_tag = bp[current_best_tag]

        best_tag_seq.reverse()

        return best_tag_seq[1:-1]

    # sentence = ['I', "can't", 'drive', 'a', 'car', '.']
#    sentence = ['you', "can't", 'very', 'well', 'sidle', 'up', 'to', 'people', 'on', 'the', 'street', 'and', 'ask',
#                'if', 'they', 'want', 'to', 'buy', 'a', 'hot', 'Bodhisattva', '.']

#    tag_seq = self.Viterbi(sentence)
#    print("/nThe given sentence:", sentence)
#    print("/nThe POS tags:", tag_seq)



    def resultsHmm(self):
        print("/n/n************Results************/n/n")
        test_list = []
        y_true = []
        y_pred = []
        i = -1
        for sent in self.test_data:
            test_list.append([])
            i = i + 1
            for elem in sent:
                test_list[i].append(elem[0])
                y_true.append(elem[1])

        print(test_list[0])

        print("/nActual tags:")
        print("************")
        print(y_true)
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print(test_list)
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        for sent in test_list:
            y_pred = y_pred + self.Viterbi(sent)

        print("/nPredicted tags:")
        print("************")
        print(y_pred)
        acc = sm.accuracy_score(y_true, y_pred)
        print("/nTotal testing accuracy: ", acc)
        precision, recall, f1, sup = sm.precision_recall_fscore_support(y_true, y_pred, average='weighted', labels=np.unique(y_pred))
        print("/nTotal precision score: ", precision)
        print("/nTotal recall score: ", recall)
        print("/nTotal f1 score: ", f1)
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('Almog sent')
        sentence = ['Hello', 'I', 'am', 'Almog', 'and', 'I', 'have']
        print(sentence)
        print('Prediction')
        print(self.Viterbi(sentence))
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')
        print('**************************************************************')

        return precision, recall, f1, acc

    def start(self):
        print("/n/n************Start************/n/n")
        logics.trainingSet()
        self.learnTransition()

        # sentence = ['I', "can't", 'drive', 'a', 'car', '.']
        sentence = ['you', "can't", 'very', 'well', 'sidle', 'up', 'to', 'people', 'on', 'the', 'street', 'and', 'ask',
                    'if', 'they', 'want', 'to', 'buy', 'a', 'hot', 'Bodhisattva', '.']

        tag_seq = self.Viterbi(sentence)
        print("/nThe given sentence:", sentence)
        print("/nThe POS tags:", tag_seq)
        print("\n\n\n********************Done HMM********************\n\n\n")

        return self.resultsHmm()







