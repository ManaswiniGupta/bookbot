import json
import logging
import numpy as np
import torch
import faiss
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi
from transformers import BertTokenizer, BertModel, DPRContextEncoder, DPRContextEncoderTokenizer
from gensim import corpora, models
from gensim.similarities import MatrixSimilarity
import PyPDF2
from toc import toc

logging.basicConfig(level=logging.INFO)

# Define TreeNode and helper functions
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

def extract_text_from_page(pdf_reader, page_num):
    page = pdf_reader.pages[page_num]
    return page.extract_text()

def create_hierarchical_tree(toc, pdf_reader):
    root = TreeNode(0, "Root")
    node_map = {0: root}
    node_counter = 1

    last_nodes_at_level = {0: root}

    for level, title, page_num in toc:
        parent_level = level - 1
        parent_node = last_nodes_at_level.get(parent_level, root)

        text_content = extract_text_from_page(pdf_reader, page_num - 1)
        new_node = TreeNode(node_counter, title, text_content)
        parent_node.add_child(new_node)
        node_map[node_counter] = new_node
        last_nodes_at_level[level] = new_node
        node_counter += 1

    return root

def display_tree(node, level=0):
    print("  " * level + f"{node.title} (ID: {node.id}, Level: {node.level})")
    for child in node.children:
        display_tree(child, level + 1)

def tree_to_dict(node: TreeNode) -> dict:
    return {
        "id": node.id,
        "title": node.title,
        "content": node.content,
        "children": [tree_to_dict(child) for child in node.children]
    }

def replace_empty_content(node, replacement_value="No content"):
    def traverse_and_replace(node):
        if not node.content:
            node.content = replacement_value
        for child in node.children:
            traverse_and_replace(child)

    traverse_and_replace(node)

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

def generate_bert_embeddings(tree_root, tokenizer, model):
    node_to_embedding = {}

    def traverse_tree(node):
        if node.content:
            inputs = tokenizer(node.content, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            node_to_embedding[node.id] = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        for child in node.children:
            traverse_tree(child)

    traverse_tree(tree_root)
    return node_to_embedding

def generate_dpr_embeddings(tree_root, dpr_tokenizer, dpr_model):
    node_to_embedding = {}

    def traverse_tree(node):
        if node.content:
            inputs = dpr_tokenizer(node.content, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = dpr_model(**inputs)
            node_to_embedding[node.id] = outputs.pooler_output.squeeze().detach().numpy()
        for child in node.children:
            traverse_tree(child)

    traverse_tree(tree_root)
    return node_to_embedding

def remove_surrogates(text):
    return text.encode('utf-8', 'replace').decode('utf-8')

def process_tree_dict(d):
    if isinstance(d, dict):
        return {k: process_tree_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [process_tree_dict(i) for i in d]
    elif isinstance(d, str):
        return remove_surrogates(d)
    else:
        return d



# Update the example usage section
pdf_path = r"E:\aaaaa\bbot1\uploads\b11.pdf"
toc_path = "uploads/toc.txt"


toc = toc

with open(pdf_path, "rb") as pdf_file:
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    root = create_hierarchical_tree(toc, pdf_reader)

display_tree(root)
replace_empty_content(root)

tree_dict = tree_to_dict(root)
clean_tree_dict = process_tree_dict(tree_dict)
with open("hierarchical_tree.json", "w", encoding="utf-8") as file:
    json.dump(clean_tree_dict, file, ensure_ascii=False, indent=4)

# Initialize BM25, BERT, and DPR models
bm25, node_to_doc_id = initialize_bm25(root)
bert_tokenizer, bert_model = initialize_bert()
dpr_tokenizer, dpr_model = initialize_dpr()

# Generate embeddings
bert_embeddings = generate_bert_embeddings(root, bert_tokenizer, bert_model)
dpr_embeddings = generate_dpr_embeddings(root, dpr_tokenizer, dpr_model)

# Save embeddings
np.save("bert_embeddings.npy", bert_embeddings)
np.save("dpr_embeddings.npy", dpr_embeddings)

# FAISS initialization
dimension = list(bert_embeddings.values())[0].shape[0] + list(dpr_embeddings.values())[0].shape[0]
faiss_index = faiss.IndexFlatL2(dimension)

faiss_vectors = []
for node_id, node in node_to_doc_id.items():
    bert_embedding = bert_embeddings.get(node.id)
    dpr_embedding = dpr_embeddings.get(node.id)
    if bert_embedding is not None and dpr_embedding is not None:
        concatenated_embeddings = np.concatenate([bert_embedding, dpr_embedding]).astype('float32')
        faiss_vectors.append(concatenated_embeddings)

faiss_index.add(np.array(faiss_vectors))
faiss.write_index(faiss_index, "faiss_index.index")

# Gensim initialization
texts = [[word for word in node.content.split()] for node in node_to_doc_id.values() if node.content]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf_model = models.TfidfModel(corpus)
gensim_index = MatrixSimilarity(tfidf_model[corpus])

dictionary.save("gensim_dictionary.dict")
tfidf_model.save("tfidf_model.model")
gensim_index.save("gensim_index.index")
