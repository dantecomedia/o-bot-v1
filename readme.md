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
Fetches API keys for OpenAI and Serp API from environment variables.

### How to Run
1. Set up the environment variables `oai` and `serp` with your OpenAI and Serp API keys.
2. Install the required libraries:
    ```bash
    pip install streamlit requests beautifulsoup4
    ```
3. Run the application:
    ```bash
    streamlit run your_script.py
    ```

## Approach
- **User Interaction**: The user interacts with a Streamlit-based web interface, providing queries and receiving responses.
- **Authentication**: A simple login mechanism ensures only authenticated users can use the bot.
- **Search and Fetch**: The bot fetches search results using the Serp API and filters the content relevant to the user's query.
- **Summarization**: The filtered content is summarized using OpenAI's GPT-4, and the summary is formatted as JSON.
- **Context Management**: The bot maintains a context of previous queries and responses to provide coherent and contextually relevant answers.
- **Asynchronous Execution**: Concurrent futures are used to fetch and process search results in parallel, improving the efficiency of the bot.

# Embedding and Query Similarity Service

This project provides a service for generating text embeddings using AWS SageMaker, storing them in MongoDB, and searching for similar queries using cosine similarity. It uses Boto3 for interacting with SageMaker, MongoDB for storage, and scikit-learn for computing similarities.

## Features
- **Generate Embeddings**: Convert text queries into embeddings using an AWS SageMaker endpoint.
- **Store Embeddings**: Store embeddings and responses in MongoDB.
- **Search Similar Queries**: Find similar queries based on cosine similarity of embeddings.
- **Context Management**: Handle context for multiple queries and responses.

## Setup

### Prerequisites
- AWS account with SageMaker endpoint setup for generating embeddings.
- MongoDB instance for storing embeddings and query responses.
- Python 3.x environment with necessary libraries installed.

### Environment Variables
Ensure the following environment variables are set:
- `oai`: OpenAI API key.
- `serp`: Serp API key.


