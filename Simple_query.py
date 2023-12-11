from elasticsearch import Elasticsearch

# Replace with your Elasticsearch endpoint and credentials
client = Elasticsearch(
    hosts=["http://localhost:9200"]
)


# Define the search query
query = {
    "multi_match": {
        "query": "php",  # Replace with your actual search term
        "fields": ["title", "comment_text", "tags"],  # Fields to search within
    }
}

response = client.search(index="mine_index", query=query)
print(response['hits']['hits'])
