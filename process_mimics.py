import json
import pickle
from collections import defaultdict
import random
from tqdm import tqdm
import numpy as np

mimics_path = "/Users/sylviadeng/Downloads/MIMICS-master/data/MIMICS-ClickExplore.tsv"
serps_path = "/Users/sylviadeng/Downloads/MIMICS-BingAPI.result"


def generate_query2intents():
    # Set fixed random seed
    random.seed(42)
    
    query2intents = {}
    with open(mimics_path, "r") as f:
        next(f)
        for line in f:
            fields = line.strip().split('\t')
            query = fields[0]
            intents = [opt for opt in fields[2:7] if opt]
            
            # Only keep queries with less than 10 intents
            if len(intents) < 10:
                if query not in query2intents:
                    query2intents[query] = set()
                query2intents[query].update(intents)

    query2intents = {k: list(v) for k, v in query2intents.items() if len(v) <= 10}
    
    print(f"Number of queries with less than 10 intents: {len(query2intents)}")
    
    # Random sampling 10% of queries
    all_queries = list(query2intents.keys())
    sample_size = int(len(all_queries) * 0.1)
    sampled_queries = random.sample(all_queries, sample_size)
    
    # Keep only sampled queries
    sampled_query2intents = {k: list(query2intents[k]) for k in sampled_queries}
    
    return sampled_query2intents

def format_serps(sampled_query2intents):
    id2doc = {}  # Store mapping from doc_id to doc content
    serps = defaultdict(list)  # Store mapping from query to doc_id list
    doc_counter = defaultdict(int)  # Counter for generating doc_id for each query
    
    # Get sampled queries set for quick lookup
    sampled_queries = set(sampled_query2intents.keys())
    
    with open(serps_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            data = json.loads(line)
            query = data['queryContext']['originalQuery']
            
            # Only process sampled queries
            if query not in sampled_queries:
                continue
                
            # Get web search results
            if 'webPages' in data and 'value' in data['webPages']:
                documents = data['webPages']['value']
                
                # Process each document
                for doc in documents:
                    # Generate doc_id: query-number
                    doc_counter[query] += 1
                    doc_id = f"{'_'.join(query.split())}-{doc_counter[query]}"
                    
                    # Combine title and snippet
                    doc_content = f"{doc.get('name', '')} {doc.get('snippet', '')}"
                    
                    # Store document content
                    id2doc[doc_id] = doc_content
                    
                    # Store query-doc_id mapping
                    serps[query].append(doc_id)
            else:
                del(sampled_query2intents[query])
    
    with open('data/id2doc.pkl', 'wb') as f:
        pickle.dump(id2doc, f)
    
    with open('data/serps.pkl', 'wb') as f:
        pickle.dump(dict(serps), f)

    with open('data/query2intents.json', 'w', encoding='utf-8') as f:
        json.dump(sampled_query2intents, f, ensure_ascii=False, indent=4)

    # Create query to qid mapping
    query2qid = {query: f"{idx+1}" for idx, query in enumerate(sampled_query2intents.keys())}
    with open('data/query2qid.json', 'w', encoding='utf-8') as f:
        json.dump(query2qid, f, ensure_ascii=False, indent=4)
    
    # Save all qids
    all_qids = np.array(list(query2qid.values()))
    np.save('data/all_qids.npy', all_qids)
    
    return id2doc, serps

def check_empty_serps(serps, sampled_query2intents):
    # Check queries with empty document list
    empty_queries = [query for query, docs in serps.items() if len(docs) == 0]
    
    # Check queries without SERP
    missing_queries = set(sampled_query2intents.keys()) - set(serps.keys())
    
    print(f"Total sampled queries: {len(sampled_query2intents)}")
    print(f"Queries with SERP: {len(serps)}")
    print(f"Queries with empty document list: {len(empty_queries)}")
    print(f"Queries without SERP: {len(missing_queries)}")
    
    if empty_queries:
        print("\nExample queries with empty document list:")
        for query in empty_queries[:5]:  # Show first 5
            print(f"- {query}")
    
    if missing_queries:
        print("\nExample queries without SERP:")
        for query in list(missing_queries)[:5]:  # Show first 5
            print(f"- {query}")

def format_judgement(sampled_query2intents, id2doc, serps):
    def check_relevance(doc_content, intent):
        # Convert document content and intent to lowercase
        doc_content = doc_content.lower()
        intent_terms = intent.lower().split()
        # Check if all intent terms are in the document
        return all(term in doc_content for term in intent_terms)
    
    # Track progress
    total_pairs = sum(len(intents) * len(serps[query]) 
                     for query, intents in sampled_query2intents.items() 
                     if query in serps)
    processed = 0
    
    with open('data/judgement.tsv', 'w', encoding='utf-8') as f:
        # Write header
        f.write("query\tintent\tdoc_id\tjudgement\n")
        
        # Iterate through each query
        for query, intents in sampled_query2intents.items():
            # Skip queries without search results
            if query not in serps:
                continue
            
            doc_ids = serps[query]
            # Iterate through all intents for this query
            for intent in intents:
                # Iterate through all documents for this query
                for doc_id in doc_ids:
                    doc_content = id2doc[doc_id]
                    # Judge relevance
                    judgement = 1 if check_relevance(doc_content, intent) else 0
                    # Write result
                    f.write(f"{query}\t{intent}\t{doc_id}\t{judgement}\n")
                    
                    processed += 1
                    if processed % 1000 == 0:
                        print(f"Processed: {processed}/{total_pairs} pairs")
    
    print(f"Judgement completed, processed {processed} query-intent-document pairs")


def split_train_test_qid():
    qid_list = np.load('data/all_qids.npy')
    train_qid_list = qid_list[:int(len(qid_list) * 0.8)]
    test_qid_list = qid_list[int(len(qid_list) * 0.8):]
    np.save('data/train_qids.npy', train_qid_list)
    np.save('data/test_qids.npy', test_qid_list)    


if __name__ == "__main__":
    sampled_query2intents = generate_query2intents()
    id2doc, serps = format_serps(sampled_query2intents)
    check_empty_serps(serps, sampled_query2intents)

    judgement = format_judgement(sampled_query2intents, id2doc, serps)

    split_train_test_qid()

    
    