import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import numpy as np
import torch
import faiss
import streamlit as st
from rank_bm25 import BM25Okapi
from transformers import BertTokenizer, BertModel, DPRContextEncoder, DPRContextEncoderTokenizer
from gensim import corpora, models
from gensim.similarities import MatrixSimilarity
from groq import Groq
from constants import api_key
# Define TreeNode and helper functions

api_key=api_key
class TreeNode:
    def __init__(self, id, title, content=None):
        self.id = id
        self.title = title
        self.content = content
        self.children = []
        self.level = 0

    def add_child(self, node):
        self.children.append(node)
        node.level = self.level + 1

def dict_to_tree(tree_dict: dict) -> TreeNode:
    node = TreeNode(tree_dict["id"], tree_dict["title"], tree_dict.get("content"))
    node.children = [dict_to_tree(child) for child in tree_dict["children"]]
    return node

def load_json(file_path: str) -> TreeNode:
    with open(file_path, "r", encoding="utf-8") as file:
        tree_dict = json.load(file)
    return dict_to_tree(tree_dict)

def initialize_bm25(tree_root):
    documents = []
    node_to_doc_id = {}

    def traverse_tree(node):
        if node.content:
            documents.append(node.content.split())
            node_to_doc_id[len(documents) - 1] = node
        for child in node.children:
            traverse_tree(child)

    traverse_tree(tree_root)
    bm25 = BM25Okapi(documents)
    return bm25, node_to_doc_id

def initialize_bert():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

def initialize_dpr():
    dpr_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    dpr_model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    return dpr_tokenizer, dpr_model

def hybrid_retrieval(query, bm25, node_to_doc_id, bert_tokenizer, bert_model, bert_embeddings, dpr_tokenizer, dpr_model, dpr_embeddings, faiss_index, gensim_index, dictionary, tfidf_model):
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    inputs = bert_tokenizer(query, return_tensors='pt', truncation=True, padding=True, max_length=512)
    bert_query_embedding = bert_model(**inputs).last_hidden_state.mean(dim=1).squeeze().detach().numpy()

    dpr_inputs = dpr_tokenizer(query, return_tensors='pt', truncation=True, padding=True, max_length=512)
    dpr_query_embedding = dpr_model(**dpr_inputs).pooler_output.squeeze().detach().numpy()

    combined_query_embedding = np.concatenate((bert_query_embedding, dpr_query_embedding), axis=0).astype('float32')

    _, faiss_indices = faiss_index.search(np.array([combined_query_embedding]), k=5)
    faiss_indices = faiss_indices.flatten()

    gensim_query_bow = dictionary.doc2bow(tokenized_query)
    gensim_similarities = gensim_index[tfidf_model[gensim_query_bow]]

    scores = []
    for doc_id, bm25_score in enumerate(bm25_scores):
        faiss_score = 1.0 if doc_id in faiss_indices else 0.0
        gensim_score = gensim_similarities[doc_id]
        combined_score = bm25_score + faiss_score + gensim_score
        scores.append((combined_score, doc_id))

    scores.sort(reverse=True, key=lambda x: x[0])
    top_nodes = [node_to_doc_id[doc_id] for _, doc_id in scores[:5]]
    return top_nodes

# Function to call Groq model
def generate_groq_response(query):
    # Example placeholder for Groq model client initialization
    # Make sure to replace this with actual Groq model initialization
    # Replace with actual Groq client import
    
    # Initialize the Groq client
    client = Groq(api_key=api_key)  # Replace with your API key or initialization
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ],
            model="llama3-70b-8192"  # Replace with your Groq model name
        )
        response = chat_completion.choices[0].message.content.strip()
        return response
    except Exception as e:
        st.error(f"An error occurred with the Groq model: {e}")
        return "There was an error processing your request."

def main():
    st.title("Hybrid Retrieval System")

    tree_root = load_json("hierarchical_tree.json")

    bm25, node_to_doc_id = initialize_bm25(tree_root)
    bert_tokenizer, bert_model = initialize_bert()
    dpr_tokenizer, dpr_model = initialize_dpr()

    bert_embeddings = np.load("bert_embeddings.npy", allow_pickle=True).item()
    dpr_embeddings = np.load("dpr_embeddings.npy", allow_pickle=True).item()

    faiss_index = faiss.read_index("faiss_index.index")
    dictionary = corpora.Dictionary.load("gensim_dictionary.dict")
    tfidf_model = models.TfidfModel.load("tfidf_model.model")
    gensim_index = MatrixSimilarity.load("gensim_index.index")

    query = st.text_input("Enter your query:")

    if query:
        top_nodes = hybrid_retrieval(
            query, bm25, node_to_doc_id, bert_tokenizer, bert_model, bert_embeddings,
            dpr_tokenizer, dpr_model, dpr_embeddings, faiss_index, gensim_index,
            dictionary, tfidf_model
        )

        st.write("Top matching nodes:")
        for node in top_nodes:
            st.write(f"Title: {node.title}")
            st.write(f"Content: {node.content[:200]}...")

        # Generate Groq response
        response = generate_groq_response(query)
        st.subheader("Response from Groq:")
        st.write(response)

if __name__ == "__main__":
    main()
