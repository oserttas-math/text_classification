# Text Classification with Naive Bayes
From [Wikipedia](https://en.wikipedia.org/wiki/Natural_language_processing#Rule-based_vs._statistical_NLP):
>In the early days, many language-processing systems were designed by hand-coding a set of rules, e.g. by writing grammars or devising heuristic rules for stemming. However, this is rarely robust to natural language variation.
<br><br>
>The machine-learning paradigm calls instead for using statistical inference to automatically learn such rules through the analysis of large corpora of typical real-world examples (**a corpus** (plural, "corpora") **is a set of documents, possibly with human or computer annotations**).
<br><br>
>Many different classes of machine-learning algorithms have been applied to natural-language-processing tasks. These algorithms take as input a large set of "features" that are generated from the input data. Some of the earliest-used algorithms, such as decision trees, produced systems of hard if-then rules similar to the systems of hand-written rules that were then common. Increasingly, however, research has focused on statistical models, which make soft, probabilistic decisions based on attaching real-valued weights to each input feature. Such models have the advantage that they can express the relative certainty of many different possible answers rather than only one, producing more reliable results when such a model is included as a component of a larger system.

<br>

Also from [Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier):
>Naive Bayes has been studied extensively since the 1950s. It was introduced under a different name into the text retrieval community in the early 1960s, and remains a popular (baseline) method for text categorization, the problem of judging documents as belonging to one category or the other (such as spam or legitimate, sports or politics, etc.) with word frequencies as the features. With appropriate pre-processing, it is competitive in this domain with more advanced methods including support vector machines.

---
The dataset explored includes reviews from all 'Top Critics' for each movie in the Rotten Tomatoes DVD & Streaming movies database.

---
### SUMMARY:
Working with a corpus of 144,595 small documents (1-2 sentence reviews), Multinomial Naive Bayes classification was implemented on models for which:
* vectorization of the corpus involved basic counting, or
* the vectorized count was normalized;
* scikit-learn's default scoring metric (accuracy) was used for hyperparameter tuning, or
* the log-likelihood was maximized during hyperparameter tuning;
* features contained exactly one word, or
* features could include 1-2 words.

Normalization never helped and resulted in over-fit training data. Both scoring metrics returned similar results, but maximizing the log-likelihood involved the least amount of over-fitting. Considering features as 1-2 words improved performance, namely in the test exercise meant to confuse the model:

>"This movie is not remarkable in any way."
* **Prediction**: 57.03% likely ROTTEN
<br><br>    

>"The movie is remarkable in every way!"
* **Prediction**: 92.51% likely FRESH
<br><br>    

>"Airplane! never gets old. My kid laughed throughout the entire thirty-something year old classic."
* **Prediction**: 69.56% likely FRESH
<br><br>    

>"The only hours I've ever wished to have back were the two I spent watching Inception."
* **Prediction**: 96.68% likely ROTTEN
<br><br>


---
### REPORTS:
* [Text Classification with Naive Bayes Notebook](http://nbviewer.jupyter.org/github/humburgc/text_classification/blob/master/text_classification_naive_bayes.ipynb)
