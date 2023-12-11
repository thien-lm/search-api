from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
from sklearn.feature_extraction.text import TfidfVectorizer
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import distance_metric, type_metric
import nltk
import json

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

@app.route('/cluster', methods=['POST'])
def cluster():
    # Get the query text from the POST request
    print("hi client")
    query_text = request.json['query']
    options = request.json['option']
    # Connect to Elasticsearch
    client = Elasticsearch(
        hosts=["http://localhost:9200"]
    )

    # Define the search query
    query = {
        "multi_match": {
            "query": query_text,
            "fields": ["title", "comment_text", "tags"],
        }
    }

    # Perform the search and retrieve the documents
    response = client.search(index="mine_index", query=query, size=200)
    documents = [hit["_source"] for hit in response["hits"]["hits"]]
    if options == True:
        results = []
        for doc in documents[0:5]:
            results.append({
                "title": doc["title"],
                "body": doc["comment_text"],
                "distance": None,
            })
        return json.dumps(results)
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
        distances = [metric(data[j], centroids[i]) for j in range(len(data))]
        nearest_document_index = distances.index(min(distances))
        nearest_document = documents[nearest_document_index]
        nearest_documents.append(nearest_document)
    results = []

    # Return the nearest documents as search results
    for i, doc in enumerate(nearest_documents):
        results.append({
            "title": doc["title"],
            "body": doc["comment_text"],
            "distance": metric(data[nearest_documents.index(doc)], centroids[i]),
        })

    # Convert the results dictionary to a JSON string
    results_json = json.dumps(results)

    return results_json

if __name__ == '__main__':
    app.run(debug=False)
