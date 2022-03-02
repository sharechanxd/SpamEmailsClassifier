from utils import *


class naiveBayes():
    def __init__(self, voclist, labels,vc):
        self.v = voclist
        self.labels = labels
        self.prior_spam = 0
        self.p_spam = 0
        self.p_ham = 0
        self.vc = vc

    def fit(self, laplace=1):
        self.prior_spam = sum(self.labels) / len(self.v)
        num_words = len(self.v[0])
        words_in_spam = np.ones(num_words)
        words_in_ham = np.ones(num_words)

        # This is similar to laplace para to adjust zero conditions
        spam_words_num = laplace
        ham_words_num = laplace

        for i in range(len(self.v)):
            if self.labels[i] == 1:
                words_in_spam += self.v[i]
                spam_words_num += sum(self.v[i])
            else:
                words_in_ham += self.v[i]
                ham_words_num += sum(self.v[i])
        self.p_spam = np.log(words_in_spam / spam_words_num)
        self.p_ham = np.log(words_in_ham / ham_words_num)

    def predict(self, testdata):
        """

        :param testdata: needs to be words vector
        :return:
        """
        vocab_marked = [0] * len(self.vc)
        for word in testdata:
            if word in self.vc:
                vocab_marked[self.vc.index(word)] += 1
        testdata = np.array(vocab_marked)
        p1 = sum(testdata * self.p_spam) + np.log(self.prior_spam)
        p0 = sum(testdata * self.p_ham) + np.log(1 - self.prior_spam)
        return p1 > p0
