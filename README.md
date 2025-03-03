# MIMICS-Diversification

This repository builds a search result diversification dataset based on the MIMICS-ClickExplore dataset. This dataset is constructed based on the [dataset](https://github.com/PxYu/LiEGe-SIGIR2022?tab=readme-ov-file) used by [LiEGe](https://dl.acm.org/doi/abs/10.1145/3477495.3532067) and the dataset processed strategy in [DUB](https://dl.acm.org/doi/10.1145/3583780.3615050).

## Data Collection

Download the MIMICS dataset from [MIMICS](https://github.com/castorini/mimics) and the search engine result page from [SERP](http://ciir.cs.umass.edu/downloads/mimics-serp/MIMICS-BingAPI-results.zip).

## Data Processing
You can build the dataset by running the following command:
```
python process_mimics.py
```

We also provide code to process the data same with [FairDiverse benchmark](https://github.com/XuChen0427/FairDiverse/tree/master):
```
python fd_data_process.py
```

## Description of the dataset

Since the document from the same URL contains different content for different queries, following the strategy in [LiEGe](https://dl.acm.org/doi/abs/10.1145/3477495.3532067), doc-id is named with *query*-number.

- `query2intents.json`: The mapping from query to potential user intents, {query: [intents, intents, ...]}. Each candidate answers in MIMICS for a queryclarification pair is considered as a potential user intents.
- `query2qid.json`: The mapping from query to query id, {query: qid}.
- `all_qids.npy`: [qid1, qid2, ...], The list of query ids.
- `serps.pkl`: {query: list} distionary, where `list` is the list of document ids retrieved by Bing for the query.
- `id2doc.pkl`: {doc_id: doc_content} dictionary, where `doc_content` is the concatenation of the document name and the document snippet.
- `judgement.tsv`: Each line contains `query\tintent\tdoc_id\tjudgement`. The judgement reveals the relevance between the document and the intent under the query.

We also provide a 8:2 split of the query ids for training and testing.
- `train_qids.npy`: [qid1, qid2, ...], The list of query ids for training.
- `test_qids.npy`: [qid1, qid2, ...], The list of query ids for testing.

For the `fd_dsata_process.py` file, it will generate the following files:
- `div_query.data`: Data structure used by [FairDiverse benchmark](https://github.com/XuChen0427/FairDiverse/tree/master).
- `stand_metrics.data`: The ideal alpha-DCG for each query.


## Data Statistics
Run `data_info.py` to get the statistics of the dataset.
```
python data_info.py
```

### Intent and Document Distribution

| Metric | Intents | Documents |
|--------|---------|-----------|
| Maximum | 10 | 10 |
| Minimum | 2 | 4 |
| Average | 6.06 | 9.08 |

### Example Queries

#### Queries with Maximum Intents (10)

1. **Query**: "deadly nightshade"  
   **Intents**: deadly nightshade side effects, deadly nightshade for sale, deadly nightshade flower, deadly nightshade facts, berries, seeds, vegetables, deadly nightshade benefits, for sale, deadly nightshade family

2. **Query**: "how to make a box"  
   **Intents**: metal, glass, paper, fabric, violin, bass, wood, guitar, drum, didgeridoo

3. **Query**: "tinea corporis"  
   **Intents**: diflucan for, causes, diet, treatment, ketoconazole, terbinafine, symptom, tinea corporis diet, nystatin for, clotrimazole

#### Queries with Minimum Intents (2)

1. **Query**: "blackmarket"  
   **Intents**: blackmarket game, blackmarket show

2. **Query**: "write a letter on my computer"  
   **Intents**: need, want to

3. **Query**: "steeped tea"  
   **Intents**: steeped tea usa, steeped tea canada


