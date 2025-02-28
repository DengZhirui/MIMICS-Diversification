import os
import math
import copy
import pickle
import numpy as np
import multiprocessing
import xml.dom.minidom
from tqdm import tqdm
from div_type import *
import json
from rank_bm25 import BM25Okapi

MAXDOC = 200
REL_LEN = 18


def get_query_dict():
    # Load MIMICS data
    with open('data/query2intents.json', 'r', encoding='utf-8') as f:
        query2intents = json.load(f)
    
    # Create query to qid mapping
    query2qid = {query: f"{idx+1}" for idx, query in enumerate(query2intents.keys())}
    
    # Save all qids
    all_qids = np.array(list(query2qid.values()))
    np.save('data/all_qids.npy', all_qids)
    
    # Create div_query dictionary
    dq_dict = {}
    for query, intents in tqdm(query2intents.items()):
        qid = query2qid[query]
        # Create subtopic IDs (1-based indexing for each query)
        subtopic_id_list = [f"{qid}.{i+1}" for i in range(len(intents))]
        
        # Create div_query object
        dq = div_query(
            qid=qid,
            query=query,
            subtopic_id_list=subtopic_id_list,
            subtopic_list=intents
        )
        dq_dict[str(qid)] = dq
    return dq_dict


def get_docs_dict():
    '''
    Calculate BM25 scores for documents and sort them
    docs_dict[qid] = [doc_id, ...]
    docs_rel_score_dict[qid] = [score, ...]
    '''
    # Load processed data
    with open('data/id2doc.pkl', 'rb') as f:
        id2doc = pickle.load(f)
    with open('data/serps.pkl', 'rb') as f:
        serps = pickle.load(f)
    
    # Load query mapping
    with open('data/query2intents.json', 'r') as f:
        query2intents = json.load(f)
    query2qid = {query: f"{idx+1}" for idx, query in enumerate(query2intents.keys())}
    
    docs_dict = {}
    docs_rel_score_dict = {}
    
    # Process each query
    for query, doc_ids in tqdm(serps.items()):
        if not doc_ids:  # Skip queries without documents
            continue
            
        qid = query2qid[query]
        
        # Get documents for this query
        query_docs = [id2doc[doc_id] for doc_id in doc_ids]
        
        # Tokenize documents
        tokenized_docs = [doc.lower().split() for doc in query_docs]
        
        # Create BM25 object
        bm25 = BM25Okapi(tokenized_docs)
        
        # Calculate BM25 scores
        tokenized_query = query.lower().split()
        doc_scores = bm25.get_scores(tokenized_query)
        
        # Sort documents by BM25 scores
        sorted_indices = np.argsort(-doc_scores)  # Sort in descending order
        
        # Store sorted doc_ids and scores
        docs_dict[qid] = [doc_ids[i] for i in sorted_indices]
        docs_rel_score_dict[qid] = [doc_scores[i] for i in sorted_indices]
        
        # Normalize scores
        max_score = docs_rel_score_dict[qid][0]  # First score is the highest
        docs_rel_score_dict[qid] = [score/max_score for score in docs_rel_score_dict[qid]]
    
    return docs_dict, docs_rel_score_dict


def get_doc_judge(qd, dd, ds):
    '''
    Load document list and relevance score list for the corresponding query
    qd : query dictionary
    dd : document dictionary
    ds : document relevance score dictionary
    '''
    # First add documents and their relevance scores
    for key in qd:
        qd[key].add_docs(dd[key])
        qd[key].add_docs_rel_score(ds[key])
    
    # Load judgements from TSV file
    with open('data/judgement.tsv', 'r', encoding='utf-8') as f:
        # Skip header
        next(f)
        for line in tqdm(f):
            query, intent, doc_id, judge = line.strip().split('\t')
            judge = int(judge)
            
            # Get qid and subtopic_id
            for qid, dq in qd.items():
                if dq.query == query:
                    # Find corresponding subtopic_id
                    for idx, subtopic in enumerate(dq.subtopic_list):
                        if subtopic == intent:
                            subtopic_id = dq.subtopic_id_list[idx]
                            # Update judgement if document exists
                            if doc_id in dq.subtopic_df.index.values:
                                dq.subtopic_df[subtopic_id][doc_id] = judge
                            break
                    break
    
    return qd


def get_stand_best_metric(qd, alpha=0.5):
    '''
    Calculate best alpha-nDCG for each query using greedy strategy
    '''
    print("Calculating best metrics for each query...")
    
    metrics_dict = {}
    for qid, query_obj in tqdm(qd.items()):
        # Calculate best ranking and metric using greedy strategy
        query_obj.get_best_rank(alpha=alpha)
        best_metric = query_obj.best_metric
        
        # Store the best metric
        metrics_dict[qid] = best_metric
        
        # Set the standard metric
        query_obj.set_std_metric(best_metric)
    
    # Save metrics dictionary with original file name and format
    with open('data/stand_metrics.data', 'wb') as f:
        pickle.dump(metrics_dict, f)


def data_process_worker(task):
    for item in task:
        qid = item[0]
        dq = item[1]
        ''' get the best ranking for the top 50 relevant documents '''
        dq.get_best_rank(MAXDOC)
        pickle.dump(dq, open('./data/best_rank/'+str(qid)+'.data', 'wb'), True)


def split_list(origin_list, n):
    res_list = []
    L = len(origin_list)
    N = int(math.ceil(L / float(n)))
    begin = 0
    end = begin + N
    while begin < L:
        if end < L:
            temp_list = origin_list[begin:end]
            res_list.append(temp_list)
            begin = end
            end += N
        else:
            temp_list = origin_list[begin:]
            res_list.append(temp_list)
            break
    return res_list


def calculate_best_rank(qd):
    data_dir = 'data/best_rank/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    q_list = []
    for key in qd:
        x = copy.deepcopy(qd[key])
        q_list.append((str(key), x))
    jobs = []
    task_list = split_list(q_list, 8)
    for task in task_list:
        p = multiprocessing.Process(target = data_process_worker, args = (task, ))
        jobs.append(p)
        p.start()
    for job in jobs:
        job.join()


def data_process():
    ''' get subtopics for each query '''
    # qd[qid] = class(qid, query, subtopic_id_list, [class(subtopic_id, subtopic), ...])
    qd = get_query_dict()
    ''' get documents dictionary '''
    dd, ds = get_docs_dict()
    ''' get diversity judge for documents '''
    qd = get_doc_judge(qd, dd, ds)
    ''' get the stand best alpha-nDCG from DSSA '''
    get_stand_best_metric(qd)
    ''' get the best ranking for top n relevant documents and save as files'''
    calculate_best_rank(qd)


def generate_qd():
    ''' generate diversity_query file from data_dir '''
    data_dir = './data/best_rank/'
    files = os.listdir(data_dir)
    files.sort(key = lambda x:int(x[:-5]))
    query_dict = {}
    for f in files:
        file_path = os.path.join(data_dir, f)
        temp_q = pickle.load(open(file_path, 'rb'))
        query_dict[str(f[:-5])] = temp_q
    pickle.dump(query_dict, open('./data/div_query.data', 'wb'), True)
    return query_dict


if __name__ == "__main__":
    data_process()
    generate_qd()

