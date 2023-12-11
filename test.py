from elasticsearch import Elasticsearch
from sklearn.feature_extraction.text import TfidfVectorizer
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import distance_metric, type_metric
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def pre_processing_text(text):
    lower_text = text.lower()
    tokens = nltk.word_tokenize(lower_text)
    stop_words = nltk.corpus.stopwords.words("english")
    filtered_tokens = [token for token in tokens if token not in stop_words]
    stemmer = nltk.PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    lemmatizer = nltk.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return ' '.join(lemmatized_tokens)


# Connect to Elasticsearch
client = Elasticsearch(
    hosts=["http://localhost:9200"]
)

# Define the search query
query = {
    "multi_match": {
        "query": "java  error",
        "fields": ["title", "comment_text", "tags"],
    }
}

# Perform the search and retrieve the documents
response = client.search(index="mine_index", query=query, size=500)
documents = [hit["_source"] for hit in response["hits"]["hits"]]
# Extract text content from each document
texts = [" ".join([doc["title"], doc["comment_text"], " ".join(doc["tags"])]) for doc in documents]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
data = X.toarray().tolist()
metric = distance_metric(type_metric.EUCLIDEAN_SQUARE)

n_clusters = 10 if len(documents) >= 10 else len(documents)

# Perform k-means clustering
initial_centers = data[:n_clusters]
kmeans_instance = kmeans(data, initial_centers, metric=metric)
kmeans_instance.process()

# Get the centroids of each cluster
centroids = kmeans_instance.get_centers()

# Find the nearest document to each centroid
nearest_documents = []
for i in range(n_clusters):
    # Calculate distances from all documents to the current centroid
    distances = [metric(data[j], centroids[i]) for j in range(len(data))]
    # Find the document with the minimum distance (nearest)
    nearest_document_index = distances.index(min(distances))
    # Get the actual document
    nearest_document = documents[nearest_document_index]
    # Add the nearest document to the list
    nearest_documents.append(nearest_document)

# Return the nearest documents as search results
print("Nearest documents to cluster centroids:")
for i, doc in enumerate(nearest_documents):
    print(f" - Cluster {i + 1}: {doc['title']}")
    # print(f" - Cluster {i + 1}, text: {doc['comment_text']}")
    print(f"   Distance: {metric(data[nearest_documents.index(doc)], centroids[i])}")

