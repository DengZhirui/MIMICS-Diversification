import json
import pickle
import numpy as np

# Load data
with open('data/query2intents.json', 'r', encoding='utf-8') as f:
    query2intents = json.load(f)

with open('data/serps.pkl', 'rb') as f:
    serps = pickle.load(f)

with open('data/id2doc.pkl', 'rb') as f:
        id2doc = pickle.load(f)

print("query number: {}, document number: {}".format(len(query2intents), len(id2doc)))

# Calculate the number of intents for each query
intent_counts = [len(intents) for intents in query2intents.values()]
max_intents = max(intent_counts)
min_intents = min(intent_counts)
avg_intents = np.mean(intent_counts)

# Calculate the number of documents for each query
doc_counts = [len(docs) for docs in serps.values()]
max_docs = max(doc_counts)
min_docs = min(doc_counts)
avg_docs = np.mean(doc_counts)

print(f"Intent Statistics:")
print(f"Maximum number of intents: {max_intents}")
print(f"Minimum number of intents: {min_intents}")
print(f"Average number of intents: {avg_intents:.2f}")
print(f"\nDocument Statistics:")
print(f"Maximum number of documents: {max_docs}")
print(f"Minimum number of documents: {min_docs}")
print(f"Average number of documents: {avg_docs:.2f}")

# Output examples of queries with the most and least intents
max_intent_queries = [q for q, i in query2intents.items() if len(i) == max_intents]
min_intent_queries = [q for q, i in query2intents.items() if len(i) == min_intents]

print(f"\nQueries with most intents ({max_intents}):")
for q in max_intent_queries[:3]:  # Show top 3
    print(f"- {q}")
    print(f"  Intents: {query2intents[q]}")

print(f"\nQueries with least intents ({min_intents}):")
for q in min_intent_queries[:3]:  # Show top 3
    print(f"- {q}")
    print(f"  Intents: {query2intents[q]}")
