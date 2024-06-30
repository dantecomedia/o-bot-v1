# O BOT

## Overview
O BOT is a Streamlit-based application that leverages OpenAI's GPT-4 and Google's search API to fetch, filter, and summarize information based on user queries. It provides an interactive and responsive interface for users to get concise and relevant information in real-time.

## Libraries
The code imports necessary libraries:
- **Streamlit**: For the web interface.
- **Requests**: For making HTTP requests.
- **JSON**: For JSON handling.
- **BeautifulSoup**: For HTML parsing.
- **Concurrent.futures**: For asynchronous execution.
- **re**: For regular expressions.
- **os**: For environment variables.
- **time**: For handling sleep.

## Setup
### Environment Variables
Fetches API keys for OpenAI and Serp API from environment variables:

## Approach
User Interaction: The user interacts with a Streamlit-based web interface, providing queries and receiving responses.
Authentication: A simple login mechanism ensures only authenticated users can use the bot.
Search and Fetch: The bot fetches search results using the Serp API and filters the content relevant to the user's query.
Summarization: The filtered content is summarized using OpenAI's GPT-4, and the summary is formatted as JSON.
Context Management: The bot maintains a context of previous queries and responses to provide coherent and contextually relevant answers.
Asynchronous Execution: Concurrent futures are used to fetch and process search results in parallel, improving the efficiency of the bot.
How to Run
Set up the environment variables oai and serp with your OpenAI and Serp API keys.
Install the required libraries:

pip install streamlit requests beautifulsoup4


Embedding and Query Similarity Service
This project provides a service for generating text embeddings using AWS SageMaker, storing them in MongoDB, and searching for similar queries using cosine similarity. It uses Boto3 for interacting with SageMaker, MongoDB for storage, and scikit-learn for computing similarities.

Features
Generate Embeddings: Convert text queries into embeddings using an AWS SageMaker endpoint.
Store Embeddings: Store embeddings and responses in MongoDB.
Search Similar Queries: Find similar queries based on cosine similarity of embeddings.
Context Management: Handle context for multiple queries and responses.
Setup
Prerequisites
AWS account with SageMaker endpoint setup for generating embeddings.
MongoDB instance for storing embeddings and query responses.
Python 3.x environment with necessary libraries installed.
Environment Variables
Ensure the following environment variables are set:

oai: OpenAI API key.
serp: Serp API key.
Libraries
The code uses the following libraries:

python
Copy code
import json
import boto3
import numpy as np
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
Configuration
Replace the placeholders with your actual configuration details:

python
Copy code
endpoint_name = ''  # Replace with your actual SageMaker endpoint name
region = 'ap-south-1'  # Replace with your AWS region
mongo_uri = "" # MongoDB URI
mongo_db_name = "" # MongoDB database name
mongo_collection_name = "" # MongoDB collection name
Functions
generate_embedding
Generates an embedding for the given text using the SageMaker endpoint.

python
Copy code
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
pad_embeddings
Pads each embedding to the maximum length.

python
Copy code
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
store_embedding_in_mongo
Stores the query and its embedding in MongoDB.

python
Copy code
def store_embedding_in_mongo(query, embedding, response=None):
    document = {"query": query, "embedding": embedding.tolist()}
    if response:
        document["response"] = response
    collection.insert_one(document)
get_all_embeddings
Retrieves all stored embeddings from MongoDB.

python
Copy code
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
compute_cosine_similarity
Computes cosine similarity between the query embedding and all stored embeddings.

python
Copy code
def compute_cosine_similarity(query_embedding, all_embeddings):
    cosine_sim = cosine_similarity([query_embedding], all_embeddings)
    return cosine_sim[0]
search_similar_queries
Searches for queries similar to the input query based on cosine similarity.

python
Copy code
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
generate_embeddings_and_store_responses
Generates embeddings for context queries and stores them in MongoDB.

python
Copy code
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
fetch_response
Mocks an API response for testing purposes.

python
Copy code
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
handler
Handles incoming events and routes them to the appropriate action.

python
Copy code
def handler(event, context):
    k1 = json.dumps(event)
    k = json.loads(k1)
    event = json.loads(k['body'])
    action = event.get('action')
    
    if action == 'generate_embeddings':
        context_data = event.get('context')
        if context_data:
            responses = generate_embeddings_and_store_responses(context_data)
            return {
                'statusCode': 200,
                'body': json.dumps({
                    "message": "Embeddings and responses stored successfully",
                    "responses": responses
                })
            }
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({"error": "No context provided"})
            }
    
    elif action == 'search_similar_queries':
        query = event.get('query')
        if query:
            similar_queries = search_similar_queries(query)
            return {
                'statusCode': 200,
                'body': json.dumps({"similar_queries": similar_queries})
            }
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({"error": "No query provided"})
            }
    
    else:
        return {
            'statusCode': 400,
            'body': json.dumps({"error": "Invalid action"})
        }
Example Usage for Local Testing
You can test the functionality locally using the following code:

python
Copy code
if __name__ == "__main__":
    example_event = {
        'body': json.dumps({
            'action': 'generate_embeddings',
            'context': {
                'q_1': 'Machine Learning Engineer',
                'follow_up': [
                    'What are the requirements for a Machine Learning Engineer?',
                    'How to prepare for a Machine Learning Engineer interview?'
                ]
            }
        })
    }
    
    response = handler(example_event, None)
    print(response)

    example_event_search = {
        'body': json.dumps({
            'action': 'search_similar_queries',
            'query': 'Machine Learning Engineer'
        })
    }
    
    response_search = handler(example_event_search, None)
    print(response_search)
