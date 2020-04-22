import math
import sys
import os
import random
import argparse

import jumbleProblem

'''
Tuan Do, Kenneth Lai
'''


# A code snippet to get class from string
def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def edit_distance(s1, s2):
    m = len(s1) + 1
    n = len(s2) + 1

    tbl = {}
    for i in range(m):  tbl[i, 0] = i;
    for j in range(n):  tbl[0, j] = j;

    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            tbl[i, j] = min(tbl[i, j - 1] + 1, tbl[i - 1, j] + 1, tbl[i - 1, j - 1] + cost)

    return tbl[i, j]


class Tester(object):
    '''
    Returns the perplexity of the data in the specified sentence
    collection according to the specified language model.  The perplexity
    is defined to be 2 to the power of the cross entropy, which in turn is
    defined as the negative of the average (over the dataset) of the log
    (base 2) of the probability, according to the model, of each datum.
    Lower perplexity indicates a better fit.

    -----
    languageModel: LanguageModel
    sentences: [[string]]
    --
    return: double
    '''

    @staticmethod
    def computePerplexity(languageModel, sentences):
        logProbability = 0.0;
        numSymbols = 0.0;

        for sentence in sentences:
            logProbability += languageModel.getSentenceLogProbability(sentence)
            numSymbols += len(sentence)

        avgLogProbability = logProbability / numSymbols

        perplexity = math.pow(0.5, avgLogProbability)

        return perplexity

    '''
    Computes the word error rate obtained using the specified language
    model to help predict correct answers to the specified list of Jumble
    sentences problems.  Each problem includes a correct
    answer and a set of candidate answers.  Here we compute the score from
    the language model, select the candidate answer with the highest
    probability, and report the edit
    distance (roughly, the number of words it got wrong -- see above)
    between the selected answer and the correct answer.  
    (If multiple candidate answers tie for the best score, we report their average edit
    distance from the correct answer.)
    This also computers the "% correct" score which is the number of sentences
    you choose exactly correct.

    -----
    languageModel: LanguageModel
    jumbleProblems: [JumboProblem]
    showGuesses: bool: True if you want to print the highest scoring sentences

    -- 
    return: (wer, correct)
    '''

    @staticmethod
    def computeWordErrorRate(languageModel, jumbleProblems, showGuesses):
        totalDistance = 0.0;
        totalWords = 0.0;
        totalWER = 0.0;
        absoluteCorrect = 0.0;

        for jProblem in jumbleProblems:
            correctSentence = jProblem.getCorrectSentence()
            bestGuess = None
            bestScore = -sys.float_info.max
            numWithBestScores = 0.0
            distanceForBestScores = 0.0
            for guess in jProblem.getNBestSentences():
                score = languageModel.getSentenceLogProbability(guess)
                distance = edit_distance(correctSentence, guess)
                if score == bestScore:
                    numWithBestScores += 1.0
                    distanceForBestScores += distance
                elif score > bestScore or bestGuess == None:
                    bestScore = score
                    bestGuess = guess
                    distanceForBestScores = distance
                    numWithBestScores = 1.0

            if showGuesses == 'True':
                print('Correct = %s' % ' '.join(correctSentence))
                print('Guess   = ' + ' '.join(bestGuess))

            if (distanceForBestScores == 0):
                absoluteCorrect += 1

            totalDistance += distanceForBestScores / numWithBestScores
            totalWords += len(correctSentence)
            totalWER += distanceForBestScores / (numWithBestScores * len(correctSentence))

        return totalWER / len(jumbleProblems), absoluteCorrect / len(jumbleProblems)

    '''
    jumbleProblems: [JumbleProblems]
    return [[String]]
    '''

    @staticmethod
    def getCorrectSentences(jumbleProblems):
        correctSentences = []
        for jProblem in jumbleProblems:
            correctSentences.append(jProblem.getCorrectSentence())
        return correctSentences;


# =======================================================================
if __name__ == "__main__":
    r = random.Random()

    parser = argparse.ArgumentParser(description='A script to test your language model')
    parser.add_argument('--data', action='store', dest='data',
                        help='Data path', default='data')
    parser.add_argument('--train', action='store', dest='trainFile',
                        help='File to train on', default='train-data.txt')
    parser.add_argument('--dev', action='store', dest='devFile',
                        help='File to test on development', default='dev-data.txt')
    parser.add_argument('--test', action='store', dest='testFile',
                        help='File to test on', default='test-data-no-oov.txt')
    parser.add_argument('--model', action='store', dest='model',
                        help='What model to use', default='unigram.Unigram')

    parser.add_argument('--showguesses', action='store', dest='showguesses',
                        help='show what you guess', default='True')
    parser.add_argument('--jumble', action='store', dest='jumble',
                        help='run Jumble (jumbled sentence) evaluation?', default='True')
    parser.add_argument('--generate', action='store', dest='generate',
                        help='generate some sentences?', default='True')
    parser.add_argument('--check', action='store', dest='check',
                        help='check probabilities sum to 1', default='True')

    print('')
    results = parser.parse_args()

    # set up file locations ...............................................
    dataPath = results.data
    trainFile = os.path.join(dataPath, results.trainFile)
    devFile = os.path.join(dataPath, results.devFile)
    testFile = os.path.join(dataPath, results.testFile)
    jumblePath = os.path.join(dataPath, 'jumble-dev')

    # load sentence data ..................................................
    # print("Training data will be read from " + trainFile)

    with open(trainFile, 'r') as fh:
        trainSentences = [line.split() for line in fh.readlines()]

    # print("Validation data will be read from " + devFile)
    with open(devFile, 'r') as fh:
        devSentences = [line.split() for line in fh.readlines()]

    # print("Testing data will be read from " + testFile + "\n")
    with open(testFile, 'r') as fh:
        testSentences = [line.split() for line in fh.readlines()]

    # load jumbled sentence problems ................................
    if results.jumble:
        jumbleProblems = jumbleProblem.JumbleProblem.readJumbleProblems(jumblePath)

    for k in range(1, 100):
        model = get_class(results.model)()
        model.k = k
        model.train(trainSentences)
        print('when K is {}'.format(k))
        print('Training set perplexity: = %.5f' % Tester.computePerplexity(model, trainSentences))
        print('Testing set perplexity: = %.5f' % Tester.computePerplexity(model, testSentences))
        # evaluate on Jumble data ................................................
        if results.jumble == 'True':
            print('Jumbled sentences: True answer perplexity: = %.5f' % Tester.computePerplexity(model,
                                                                                                 Tester.getCorrectSentences(
                                                                                                     jumbleProblems)))
            # Get the WER and % correct scores.
            scores = Tester.computeWordErrorRate(model, jumbleProblems, results.showguesses);
            print("Jumbled sentences: Word Error Rate = %.5f%%" % (float(scores[0]) * 100))
            print("Jumbled sentences: Percent Correct = %.5f%%" % (float(scores[1]) * 100))
            print('')

