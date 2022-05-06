# %%
from sklearn.datasets import fetch_20newsgroups
from pprint import pprint

newsgroups_train = fetch_20newsgroups(subset="train")

pprint(list(newsgroups_train.target_names))
# %%
print(newsgroups_train.filenames.shape)
print(newsgroups_train.target.shape)
print(newsgroups_train.target[:10])
# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# categories = ['alt.atheism', 'talk.religion.misc',
#               'comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset="train")
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
vectors.shape

# %%
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

newsgroups_test = fetch_20newsgroups(subset="test")
vectors_test = vectorizer.transform(newsgroups_test.data)
clf = MultinomialNB(alpha=0.01)
clf.fit(vectors, newsgroups_train.target)
pred = clf.predict(vectors_test)
metrics.f1_score(newsgroups_test.target, pred, average="macro")
# %%
newsgroups_test = fetch_20newsgroups(
    subset="test", remove=("headers", "footers", "quotes")
)
vectors_test = vectorizer.transform(newsgroups_test.data)
pred = clf.predict(vectors_test)
metrics.f1_score(pred, newsgroups_test.target, average="macro")
# %%
