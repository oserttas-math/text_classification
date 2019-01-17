"""
Some of the functions developed for this project.

"""
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

def split_vectorize_classify(X, y, parameters, tfidf=False, ngram_range=(1, 1)):
    
    if tfidf:
        pipeline = Pipeline([
            ('vect', TfidfVectorizer(ngram_range=ngram_range)),
            ('clf', MultinomialNB()),
        ])
        print('TFIDF VECTORIZER')
    else:
        pipeline = Pipeline([
            ('vect', CountVectorizer(ngram_range=ngram_range)),
            ('clf', MultinomialNB()),
        ])
        print('COUNT VECTORIZER')
        
    # Split:
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=32)
    
    # Vectorize and fit:
    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1)    
    grid_search.fit(X_train, y_train)
    
    best_parameters = grid_search.best_estimator_.get_params()
    best_score = grid_search.best_score_

    print('Best Score: {:0.2f}%'.format(best_score*100))
    for param_name in sorted(parameters.keys()):
        print('\n{}\n\tsearched={}\n\tbest={}'.format(param_name, parameters[param_name], best_parameters[param_name]))
        
    # Save MultinomialNB classifier:
    clf = grid_search.best_estimator_.named_steps['clf']

    # Save the vectorizer:
    vectorizer = grid_search.best_estimator_.named_steps['vect']

    clf.fit(vectorizer.fit_transform(X_train), y_train)
    
    print()
    # grid_search.score(xtrain, ytrain) would also work here:
    print("Accuracy on Training Data: {:0.2f}%".format(clf.score(vectorizer.fit_transform(X_train), y_train)*100))

    # grid_search.score(xtest, ytest) would also work here:
    print("    Accuracy on Test Data: {:0.2f}%".format(clf.score(vectorizer.transform(X_test), y_test)*100))
    
    # Classify:
    confusion = confusion_matrix(y_test, grid_search.predict(X_test))
    
    return clf, vectorizer, confusion


def log_likelihood(clf, X, y):
    log_proba = clf.predict_log_proba(X)
    rotten = y==0
    fresh = y==1
    return log_proba[rotten, 0].sum() + log_proba[fresh, 1].sum()

def cross_val_score(clf, X, y, vectorizer, scoring_func):
    n_folds = 5
    sum_of_scores = 0
    
    # Split into 5 train/test sets, train and score each set-
    for train, test in KFold(n_folds, random_state=32).split(X):  
        clf.fit(vectorizer.fit_transform(X[train]), y[train])
        sum_of_scores += scoring_func(clf, vectorizer.transform(X[test]), y[test])
     
    # Return the average score:
    return sum_of_scores/n_folds

def split_vect_clfy_maxlikelihood(X, y, parameters, tfidf=False, ngram_range=(1, 1)):    
    best_max_df = None
    best_alpha = None
    max_score = -np.inf
    
    if tfidf:
        print('TFIDF VECTORIZER')
        # Search for best parameters by maximizing log-likelihood:
        for min_df in parameters['vect__min_df']:
            for max_df in parameters['vect__max_df']:
                for alpha in parameters['clf__alpha']:
                    for fit_prior in parameters['clf__fit_prior']:
                        vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range)
                        clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
                        score = cross_val_score(clf, X, y, vectorizer, log_likelihood)

                        if score > max_score:
                            max_score = score
                            best_min_df, best_max_df, best_alpha, best_fit_prior = min_df, max_df, alpha, fit_prior
                            
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=32)
        vectorizer = TfidfVectorizer(min_df=best_min_df, max_df=best_max_df, ngram_range=ngram_range)
        
    else:
        print('COUNT VECTORIZER')
        # Search for best parameters by maximizing log-likelihood:
        for min_df in parameters['vect__min_df']:
            for max_df in parameters['vect__max_df']:
                for alpha in parameters['clf__alpha']:
                    for fit_prior in parameters['clf__fit_prior']:
                        vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range)
                        clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
                        score = cross_val_score(clf, X, y, vectorizer, log_likelihood)

                        if score > max_score:
                            max_score = score
                            best_min_df, best_max_df, best_alpha, best_fit_prior = min_df, max_df, alpha, fit_prior
                            
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=32)
        vectorizer = CountVectorizer(min_df=best_min_df, max_df=best_max_df, ngram_range=ngram_range)
    
    print("Best min_df: {}\nBest max_df: {}\nBest alpha: {}\nBest fit_prior: {}".format(best_min_df, best_max_df, best_alpha, best_fit_prior))
    print()
    
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    clf = MultinomialNB(alpha=best_alpha, fit_prior=best_fit_prior).fit(X_train, y_train)
    
    print("Accuracy on Training Data: {:0.2f}%".format(clf.score(X_train, y_train)*100))
    print("    Accuracy on Test Data: {:0.2f}%".format(clf.score(X_test, y_test)*100))
    
    # Classify:
    confusion = confusion_matrix(y_test, clf.predict(X_test))
    
    return clf, vectorizer, confusion


    