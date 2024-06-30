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
```python
openai_api_key = os.environ.get('oai')
serp_api = os.environ.get('serp')
Headers and Helper Functions
Sets up headers for the OpenAI API requests:

python
Copy code
headers = {
    'Authorization': 'Bearer ' + openai_api_key,
    'Content-Type': 'application/json',
}
to_json
Cleans up and converts a string into JSON format by removing tags and extracting the JSON substring:

python
Copy code
def to_json(string):
    string = string.strip().replace('<json>', '').replace('</json>', '')
    start_index = string.find('{')
    end_index = string.rfind('}')
    json_string = string[start_index:end_index + 1]
    data = json.loads(json_string, strict=False)
    return data
summarizer_stream
Creates a prompt for GPT-4 based on the user's query, context, and search results. It then streams the response from the OpenAI API and yields the parsed JSON result:

python
Copy code
def summarizer_stream(context, query, outcome):
    ...
update_context
Updates the context with the latest query and summary. It also ensures the context length does not exceed a specified limit by summarizing the context if needed:

python
Copy code
def update_context(context, query, summary, max_context_length=5000):
    ...
summarize_context
Summarizes the context using a sliding window approach to ensure the context remains within the specified length:

python
Copy code
def summarize_context(context, window_size=3):
    ...
fetch_and_filter_text
Fetches and filters relevant text from a URL based on the user's query:

python
Copy code
def fetch_and_filter_text(url, query, max_length=1000):
    ...
fetch_search_results
Fetches search results from Google's search API using the provided query:

python
Copy code
def fetch_search_results(query):
    ...
Main Application Logic
Streamlit State Initialization
Initializes session state variables to manage context, authentication, chat history, status, and query:

python
Copy code
def main():
    st.title("O BOT")
    if 'context' not in st.session_state:
        st.session_state.context = {}
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'status' not in st.session_state:
        st.session_state.status = 'green'
    if 'query' not in st.session_state:
        st.session_state.query = ""
Authentication
Implements a simple authentication mechanism. If the username and password are correct, the user is authenticated and the page reloads:

python
Copy code
    if not st.session_state.authenticated:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "admin" and password == "password":
                st.session_state.authenticated = True
                st.success("Logged in successfully")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")
Display Chat History and Status
Displays previous chat history and the current API status:

python
Copy code
    if st.session_state.authenticated:
        for chat in st.session_state.chat_history:
            st.markdown(...)
            if chat['follow_up']:
                with st.expander("Follow Up Questions", expanded=True):
                    ...
            if chat['prequisiter']:
                with st.expander("Prerequisite Information", expanded=True):
                    ...
        status = st.session_state.status
        if status == 'green':
            st.markdown('<div style="background-color:green;color:white;padding:10px;">API Status: Success</div>', unsafe_allow_html=True)
        elif status == 'yellow':
            st.markdown('<div style="background-color:yellow;color:black;padding:10px;">API Status: Processing</div>', unsafe_allow_html=True)
        elif status == 'red':
            st.markdown('<div style="background-color:red;color:white;padding:10px;">API Status: Error</div>', unsafe_allow_html=True)
Query Input
Takes the user query and triggers a search when the button is clicked:

python
Copy code
        query = st.text_input("Enter your query", value="", key="query_input")
        submit_button = st.button("Search")
        if submit_button and query:
            st.session_state.query = query
            st.session_state.status = 'yellow'
            st.experimental_rerun()
Fetch and Filter Search Results
Uses a thread pool to concurrently fetch and filter text from the search results:

python
Copy code
        if st.session_state.status == 'yellow' and st.session_state.query:
            search_results = fetch_search_results(st.session_state.query)
            urls = [result.get('href') for result in search_results]
            results_list = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_url = {executor.submit(fetch_and_filter_text, url, st.session_state.query): url for url in urls}
                for future in concurrent.futures.as_completed(future_to_url):
                    try:
                        result = future.result()
                        results_list.append(result)
                    except Exception as exc:
                        st.session_state.status = 'red'
                        st.error(f"Error fetching text: {exc}")
                        st.experimental_rerun()
Generate and Display Summary
Streams the summary from the OpenAI API and updates the chat history with the new query and response. It also handles displaying the response in real-time:

python
Copy code
            response_placeholder = st.empty()
            response_text = ""
            for summary in summarizer_stream(context, st.session_state.query, results_list):
                if summary:
                    context = update_context(context, st.session_state.query, summary)
                    st.session_state.context = context
                    st.session_state.chat_history.append({
                        "query": st.session_state.query,
                        "response": summary['res']['outcome'],
                        "follow_up": summary['res'].get('follow_up', []),
                        "prequisiter": summary['res'].get('prequisiter', [])
                    })
                    st.session_state.status = 'green'
                    st.session_state.query = ""
                    response_text = ""
                    for char in summary['res']['outcome']:
                        response_text += char
                        response_placeholder.markdown(f"<div style='text-align: left; border: 1px dotted #000; padding: 10px; margin: 5px; border-radius: 10px; white-space: pre-wrap;'>{response_text}</div>", unsafe_allow_html=True)
                        time.sleep(0.001)
                else:
                    st.session_state.status = 'red'
            st.experimental_rerun()
Main Execution
Ensures the main function is called when the script is run:

python
Copy code
if __name__ == "__main__":
    main()
Approach
User Interaction: The user interacts with a Streamlit-based web interface, providing queries and receiving responses.
Authentication: A simple login mechanism ensures only authenticated users can use the bot.
Search and Fetch: The bot fetches search results using the Serp API and filters the content relevant to the user's query.
Summarization: The filtered content is summarized using OpenAI's GPT-4, and the summary is formatted as JSON.
Context Management: The bot maintains a context of previous queries and responses to provide coherent and contextually relevant answers.
Asynchronous Execution: Concurrent futures are used to fetch and process search results in parallel, improving the efficiency of the bot.
How to Run
Set up the environment variables oai and serp with your OpenAI and Serp API keys.
Install the required libraries:
bash
Copy code
pip install streamlit requests beautifulsoup4
Run the application:
bash
Copy code
streamlit run your_script.py
Open the Streamlit app in your browser, authenticate using the credentials (username: admin, password: password), and start interacting with O BOT.
