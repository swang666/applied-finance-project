'''This file aims to find the best number of topics
for lda model
'''
import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel
from ucla_topic_analysis.model import lda

def find_optimal_num_topics(a, b):
    ''' this function takes the range of topics number
    for each number it runs an lda model, calculate its
    coherence score, return the model with highest score

    Args: int a, b

    Returns: CoherenceScores: list of floats
    '''
    CoherenceScores = []
    for i in range(a, b+1):
        Mylda = lda.Lda(i, 1)
        Mylda.build_lda_model()
        path = 'ucla_topic_analysis/validation/num_topics_' + str(i) + '.gensim'
        Mylda.model.save(path)
        #calculate coherence score
        coherence_model_lda = CoherenceModel(model=Mylda.model, texts=Mylda.text_data,
                                             dictionary=Mylda.dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        CoherenceScores.append(coherence_lda)
    return CoherenceScores

def plot_graph(a, b):
    '''this function plots the coherence score values
    into a graph

    Args: int a, b

    Returns: a plot
    '''
    Y = find_optimal_num_topics(a, b)
    X = range(a, b+1)
    plt.plot(X, Y)
    plt.xlabel('Num Topics')
    plt.ylabel('Coherence Score')
    plt.legend(('coherence_values'), loc='best')
    plt.show()
