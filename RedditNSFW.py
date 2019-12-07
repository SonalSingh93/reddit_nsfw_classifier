import requests
import random
import os

os.environ["OMP_NUM_THREADS"] = "15"
os.environ["OPENBLAS_NUM_THREADS"] = "15"
os.environ["JOBLIB_START_METHOD"] = "forkserver"

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from gensim import models
import numpy as np
import sys
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def get_posts(subreddit, time_frame, after=None, count=None):
    url = "http://reddit.com/r/" + subreddit + "/top.json?t=" + time_frame + "&limit=100"

    if after:
        url += "&after=" + after

    if count:
        url += "&count=" + str(count)

    params = {
        "User-Agent": "Windows/Python/1.0"
    }

#     print(url)
    response = requests.get(url, headers=params)
    response.raise_for_status()
    response_obj = response.json()

#     print(response_obj)

    return response_obj


def process_posts(response_obj):
    data = response_obj['data']
    posts = data['children']

    processed_posts = []

    for post in posts:
        post_data = post['data']

        score = post_data['score']
        comments = post_data['num_comments']

        if score < 10 or comments < 10:
            continue

        title = post_data['title']
        text = post_data['selftext']
        nsfw = post_data['over_18']

        processed_post = {
            "text": title + " " + text,
            "nsfw": nsfw
        }

        processed_posts.append(processed_post)

    return processed_posts


def make_bow(processed_posts):
    bow = set()

    for post in processed_posts:
        text = post["text"]
        text = text.lower()
        split_text = text.split(" ")

        bow.update(split_text)

    return bow


def get_bow_vectors(processed_posts, bow):
    feature_vectors = []
    target_labels = []

    for post in processed_posts:
        post_text = post["text"]
        post_label = post["nsfw"]
        post_label_encoded = 1 if post_label else 0

        feature_vector = []

        for word in bow:
            word_count = post_text.count(word)
            feature_vector.append(word_count)

        feature_vectors.append(feature_vector)
        target_labels.append(post_label_encoded)

    return feature_vectors, target_labels


# according to http://redditlist.com/all
# note: excluding r/announcements and r/blog
top_40_overall_subreddits = ["funny", "AskReddit", "gaming", "pics", "science",
                             "worldnews", "aww", "movies", "todayilearned", "videos",
                             "Music", "IAmA", "news", "gifs", "EarthPorn", "ShowerThoughts",
                             "askscience", "Jokes", "explainlikeimfive", "books", "food", "LifeProTips", "DIY",
                             "mildlyinteresting", "Art", "sports", "space", "gadgets", "nottheonion", "television",
                             "television", "photoshopbattles", "Documentaries", "GetMotivated", "listentothis",
                             "UpliftingNews", "tifu", "InternetIsBeautiful", "history", "Futurology", "philosophy", "OldSchoolCool"]

top_20_sfw_subreddits = ["funny", "AskReddit", "gaming", "pics", "science",
                         "worldnews", "aww", "movies", "todayilearned", "videos",
                         "Music", "IAmA", "news", "gifs", "EarthPorn", "ShowerThoughts",
                         "askscience", "Jokes", "explainlikeimfive", "books"]

# WARNING: nsfw subreddits have nsfw names, as you'd exepct
top_20_nsfw_subreddits = ["gonewild", "nsfw", "realgirls", "nsfw_gif", "holdthemoan",
                          "imgoingtohellforthis", "bustypetite", "cumsluts", "legalteens",
                          "petitegonewild", "nsfw_gifs", "adorableporn", "girlsfinishingthejob",
                          "asiansgonewild", "rule34", "amateur", "biggerthanyouthought", "collegesluts",
                          "porninfifteenseconds", "tittydrop"]

top_40_combined_subreddits = top_20_nsfw_subreddits
top_40_combined_subreddits.extend(top_20_sfw_subreddits)

time_frame = "year"

posts_to_get_per_subreddit = 1000
all_processed_posts = []

for subreddit in top_40_combined_subreddits:
    print("Starting to fetch for subreddit: " + subreddit)

    retrieved_posts = 0
    after = None
    while retrieved_posts < posts_to_get_per_subreddit:
        posts = get_posts(subreddit, time_frame, after, retrieved_posts)

        processed_posts = process_posts(posts)

        if len(processed_posts) == 0:
            print("warn: Got no posts that met minimum score threshold. Quitting early with only " + str(retrieved_posts) + " total posts from " + subreddit)
            break

        all_processed_posts.extend(processed_posts)
        retrieved_posts += len(processed_posts)

        after = posts['data']['after']

        if after is None:
            print("warn: Did not find 'after' value in response. Quitting early with only " + str(retrieved_posts) + " total posts from " + subreddit)
            break


# duplicates
print(len([post for post in all_processed_posts if all_processed_posts.count(post) > 1]))
# all_processed_posts = [post for post in all_processed_posts if all_processed_posts.count(post) == 1]

# check how many nsfw
nsfw_posts = [post for post in all_processed_posts if post['nsfw']]
print(len(nsfw_posts))
print(len(all_processed_posts))


random.shuffle(all_processed_posts)

train_end = int(0.8 * len(all_processed_posts))
train_set = all_processed_posts[:train_end]
test_set = all_processed_posts[train_end:]

bow = make_bow(all_processed_posts)

train_feature_vectors, train_target_labels = get_bow_vectors(train_set, bow)
test_feature_vectors, test_target_labels = get_bow_vectors(test_set, bow)

# ########################################################## Naive Bayes
nb = MultinomialNB()
nb.fit(train_feature_vectors, train_target_labels)
test_predictions = nb.predict(test_feature_vectors)

accuracy = accuracy_score(test_target_labels, test_predictions)
precision = precision_score(test_target_labels, test_predictions)
recall = recall_score(test_target_labels, test_predictions)
f1 = f1_score(test_target_labels, test_predictions)

print("Accuracy: " + str(accuracy))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F1: " + str(f1))

# Top 20 NSFW + Top 20 SFW, 1000 posts each, full count BoW with NB:
#Accuracy: 0.8222609161532279
#Precision: 0.8981581798483207
#Recall: 0.7041336353340883
#F1: 0.7893985081733059

# ########################################################## Logistic regression classifier
logisticRegr = LogisticRegression(solver="liblinear", max_iter=500)
logisticRegr.fit(train_feature_vectors, train_target_labels)
lr_predictions = logisticRegr.predict(test_feature_vectors)

accuracy = accuracy_score(test_target_labels, lr_predictions)
precision = precision_score(test_target_labels, lr_predictions)
recall = recall_score(test_target_labels, lr_predictions)
f1 = f1_score(test_target_labels, lr_predictions)

print("Accuracy: " + str(accuracy))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F1: " + str(f1))

# Top 20 NSFW + Top 20 SFW, 1000 posts each, full count BoW with LR:
# Accuracy: 0.899370226450489
# Precision: 0.8677929739876643
# Recall: 0.9261591299370349
# F1: 0.8960265817527344

# ######################################################################### SVM Classifier
svm = LinearSVC(max_iter=2000)
svm.fit(train_feature_vectors, train_target_labels)
svm_predictions = svm.predict(test_feature_vectors)

accuracy = accuracy_score(test_target_labels, svm_predictions)
precision = precision_score(test_target_labels, svm_predictions)
recall = recall_score(test_target_labels, svm_predictions)
f1 = f1_score(test_target_labels, svm_predictions)

print("Accuracy: " + str(accuracy))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F1: " + str(f1))

# Top 20 NSFW + Top 20 SFW, 1000 posts each, full count BoW with LinearSVM (did not converge):
# Accuracy: 0.8926705078386707
# Precision: 0.8743397275507367
# Recall: 0.9001144819690898
# F1: 0.8870399097447468

# #################################################################### MLP Classifier
# Load the word embeddings data into w
w = models.KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin', binary=True)

wlen = w["word"].size


def getAverageWordEmbeddings(review):
    length_of_review = len(review)
    result = [w[word] for word in review if word in w]
#     print(np.sum(result, axis=0)/len(result))
    return np.sum(result, axis=0) / len(result)


word2vec_train_data = []
word2vec_train_labels = []
word2vec_test_data = []
word2vec_test_labels = []


for post in train_set:
    avg_word_embedding = getAverageWordEmbeddings(post['text'])
    if avg_word_embedding.size == 1:
        continue
#     if avg_word_embedding.size!=300:
#         print(avg_word_embedding.size)
    word2vec_train_data.append(avg_word_embedding)
    word2vec_train_labels.append(post['nsfw'])
word2vec_train_data = np.array(word2vec_train_data)
word2vec_train_labels = np.array(word2vec_train_labels)

for post in test_set:
    avg_word_embedding = getAverageWordEmbeddings(post['text'])
    if avg_word_embedding.size == 1:
        continue
#     print(avg_word_embedding.size)
    word2vec_test_data.append(avg_word_embedding)
    word2vec_test_labels.append(post['nsfw'])
word2vec_test_data = np.array(word2vec_test_data)
word2vec_test_labels = np.array(word2vec_test_labels)

net = MLPClassifier(activation='relu', hidden_layer_sizes=(100, 50, 10), max_iter=300)
net.fit(word2vec_train_data, word2vec_train_labels)
net_test_predictions = net.predict(word2vec_test_data)

accuracy = accuracy_score(word2vec_test_labels, net_test_predictions)
precision = precision_score(word2vec_test_labels, net_test_predictions)
recall = recall_score(word2vec_test_labels, net_test_predictions)
f1 = f1_score(word2vec_test_labels, net_test_predictions)

print("Accuracy: " + str(accuracy))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F1: " + str(f1))

# ############################################################# SVD - LSA for topic modelling


vectorizer = TfidfVectorizer()

all_post_text = []
stop_words = set(stopwords.words('english'))
for post in all_processed_posts:
    filtered_post = [w for w in post['text'].split() if not w in stop_words]
    all_post_text.append(' '.join(filtered_post))

X = vectorizer.fit_transform(all_post_text)
tfidf_vocabulary = vectorizer.vocabulary_

svd_model = TruncatedSVD(n_components=30, algorithm='randomized', n_iter=100, random_state=122)
svd_model.fit(X)


terms = vectorizer.get_feature_names()

for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic {}: {}".format(str(i),[w[0] for w in sorted_terms]))
    for t in sorted_terms:
        print(t[0])