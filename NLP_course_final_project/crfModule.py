import sklearn_crfsuite
from sklearn_crfsuite import metrics
import logics
import pickle
import warnings
warnings.filterwarnings('ignore')


class crfClinet():

    def __init__(self, fit=False):

        testD, tagD, wd, corpus = logics.trainingSet()
        self.fit = fit
        self.test_data = testD
        self.tag_dict = tagD
        self.train_sents = corpus
        self.word_dict = wd
        self.crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )

    def start(self):

        X_train = [self.sent2features(s) for s in self.train_sents]
        y_train = [self.sent2labels(s) for s in self.train_sents]

        X_test = [self.sent2features(s) for s in self.test_data]
        y_test = [self.sent2labels(s) for s in self.test_data]

        if self.fit:
            self.crf.fit(X_train, y_train)
            crf = self.crf

            with open('model.pkl', 'wb') as f:
                pickle.dump(crf, f)

        else:
            with open('model.pkl', 'rb') as f:
                self.crf = pickle.load(f)
        y_pred = self.crf.predict(X_test)
        labels = list(self.crf.classes_)

        f1 = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)

        precision = metrics.flat_precision_score(y_test, y_pred, average='weighted', labels=labels)

        recall = metrics.flat_recall_score(y_test, y_pred, average='weighted', labels=labels)

        acc = metrics.flat_accuracy_score(y_test, y_pred)

        return precision, recall, f1, acc




    def word2features(self, sent, i):
        word = sent[i][0]
        features = {
                        'bias': 1.0,
                        'word': word,
                        'is_first': i == 0,
                        'is_last': i == len(sent) - 1,
                        'is_capitalized': word[0].upper() == word[0],
                        'is_all_caps': word.upper() == word,
                        'is_all_lower': word.lower() == word,
                        'prefix-1': word[0],
                        'prefix-2': word[:2],
                        'prefix-3': word[:3],
                        'suffix-1': word[-1],
                        'suffix-2': word[-2:],
                        'suffix-3': word[-3:],
                        'prev_word': '' if i == 0 else sent[i - 1][0],
                        'next_word': '' if i == len(sent) - 1 else sent[i + 1][0],
                        'has_hyphen': '-' in word,
                        'is_numeric': word.isdigit(),
                        'capitals_inside': word[1:].lower() != word[1:]
                    }

        return features


    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]


    def sent2labels(self, sent):
        return [label for i, label in sent]

# print(sent2features(corpus[0]))