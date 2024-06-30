from flask import Flask, request, jsonify
import json
import boto3
import numpy as np
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
endpoint_name = ''
region = 'ap-south-1'
mongo_uri = ""
mongo_db_name = ""
mongo_collection_name = ""

sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=region)
client = MongoClient(mongo_uri)
db = client[mongo_db_name]
collection = db[mongo_collection_name]

app = Flask(__name__)

def generate_embedding(text):
    data = {"inputs": text}
    data_json = json.dumps(data)
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=data_json
    )
    result = json.loads(response['Body'].read().decode())
    
    if isinstance(result, list) and len(result) > 0:
        embeddings = result[0]
    else:
        raise ValueError("Unexpected response format from SageMaker endpoint")
    
    embeddings_list = embeddings if isinstance(embeddings, list) else embeddings.tolist()
    return np.array(embeddings_list)

def pad_embeddings(embeddings, max_length):
    padded_embeddings = []
    for embedding in embeddings:
        if embedding.shape[0] < max_length:
            padding = np.zeros((max_length - embedding.shape[0], embedding.shape[1]))
            padded_embedding = np.vstack((embedding, padding))
        else:
            padded_embedding = embedding
        padded_embeddings.append(padded_embedding)
    return np.array(padded_embeddings)

def store_embedding_in_mongo(query, embedding, response=None):
    document = {"query": query, "embedding": embedding.tolist()}
    if response:
        document["response"] = response
    collection.insert_one(document)

def get_all_embeddings():
    cursor = collection.find({})
    queries = []
    embeddings = []
    responses = []
    for document in cursor:
        queries.append(document['query'])
        embeddings.append(np.array(document['embedding']))
        if 'response' in document:
            responses.append(document['response'])
        else:
            responses.append(None)
    
    max_length = max(embedding.shape[0] for embedding in embeddings)
    padded_embeddings = pad_embeddings(embeddings, max_length)
    
    return queries, padded_embeddings, responses

def compute_cosine_similarity(query_embedding, all_embeddings):
    cosine_sim = cosine_similarity([query_embedding], all_embeddings)
    return cosine_sim[0]

def search_similar_queries(query, top_k=5, threshold=0.80):
    query_embedding = generate_embedding(query)
    query_embedding_mean = query_embedding.mean(axis=0)
    
    all_queries, all_embeddings, _ = get_all_embeddings()
    all_embeddings_mean = np.array([embedding.mean(axis=0) for embedding in all_embeddings])
    
    similarities = compute_cosine_similarity(query_embedding_mean, all_embeddings_mean)
    
    filtered_indices = [i for i, score in enumerate(similarities) if score > threshold and all_queries[i].strip().lower() != query.strip().lower()]
    filtered_queries = [(all_queries[i], similarities[i]) for i in filtered_indices]
    
    filtered_queries.sort(key=lambda x: x[1], reverse=True)
    
    return filtered_queries[:top_k]

def generate_embeddings_and_store_responses(context):
    responses = []
    if 'q_1' in context:
        first_query = context['q_1']
        first_query_embedding = generate_embedding(first_query)
        first_query_response = fetch_response(first_query)
        store_embedding_in_mongo(first_query, first_query_embedding, first_query_response)
        responses.append({"query": first_query, "response": first_query_response})
        
        if 'follow_up' in context:
            follow_up_queries = context['follow_up']
            for follow_up_query in follow_up_queries:
                follow_up_embedding = generate_embedding(follow_up_query)
                follow_up_response = fetch_response(follow_up_query)
                store_embedding_in_mongo(follow_up_query, follow_up_embedding, follow_up_response)
                responses.append({"query": follow_up_query, "response": follow_up_response})
    else:
        print("No initial query found in context")
    return responses

def fetch_response(query):
    response = {
        "results": [
            {
                "title": "Machine Learning Engineer Job Description",
                "snippet": "A Machine Learning Engineer is responsible for creating programs and algorithms that enable machines to take actions without being directed...",
                "link": "https://example.com/machine-learning-engineer-job-description"
            },
            {
                "title": "How to Become a Machine Learning Engineer",
                "snippet": "To become a Machine Learning Engineer, you typically need a background in computer science, mathematics, or a related field, along with hands-on experience...",
                "link": "https://example.com/how-to-become-a-machine-learning-engineer"
            }
        ]
    }
    return response

@app.route('/generate_embeddings', methods=['POST'])
def generate_embeddings():
    data = request.get_json()
    context_data = data.get('context')
    if context_data:
        responses = generate_embeddings_and_store_responses(context_data)
        return jsonify({
            "message": "Embeddings and responses stored successfully",
            "responses": responses
        })
    else:
        return jsonify({"error": "No context provided"}), 400

@app.route('/search_similar_queries', methods=['POST'])
def search_similar_queries_endpoint():
    data = request.get_json()
    query = data.get('query')
    if query:
        similar_queries = search_similar_queries(query)
        return jsonify({"similar_queries": similar_queries})
    else:
        return jsonify({"error": "No query provided"}), 400

@app.route('/get_responses', methods=['GET'])
def get_responses():
    cursor = collection.find({})
    responses = []
    for document in cursor:
        responses.append({
            "query": document['query'],
            "response": document.get('response', None)
        })
    return jsonify({"responses": responses})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
