# BM25 & Filtering im RAG-System  
Verbesserung von Retrieval-Genauigkeit in Retrieval-Augmented Generation (RAG)

Dieses Repository erklärt und demonstriert, welche Probleme durch BM25 (Keyword Retrieval) und Filtering (Metadaten-Filterung) in RAG-Systemen gelöst werden und warum die reine semantische Vektor-Suche nicht ausreicht.

---

## Warum BM25 & Filtering im RAG wichtig sind

Viele RAG-Projekte verlassen sich ausschließlich auf semantische Embeddings.  
Das führt zu Problemen, insbesondere bei technischen, juristischen oder sehr spezifischen Daten.

BM25 und Filtering ergänzen die Vektor-Suche und sorgen für:  
- höhere Präzision  
- bessere Kontrolle über Quellen  
- weniger Halluzinationen  
- mehr Robustheit in realen Anwendungen

---

# 1. Welches Problem löst BM25?

Semantische Embeddings sind stark in der Bedeutungsähnlichkeit, aber schlecht bei:

- exakten Begriffen  
- juristischen Paragraphen (§4 DSGVO)  
- IDs, Produktcodes, Zahlen  
- technischen Fachwörtern  
- Abkürzungen  
- Querys mit wenigen Tokens

Reine Embedding-Retrieval-Systeme übersehen diese Begriffe häufig oder ordnen sie falsch ein.

## Lösung: BM25 (Keyword-Retrieval)

BM25 ist eine klassische, bewährte Suchtechnik, optimiert für:

- exakte Worttreffer  
- kurze Suchanfragen  
- technische Formulierungen  
- juristische Texte  
- numerische und spezifische Daten

Beispiel:  
Eine Suche nach „§4 Datenschutz“ wird durch BM25 zuverlässig gefunden Embeddings jedoch scheitern oft oder liefern nur semantisch ähnliche Stellen.

---

# 2. Welches Problem löst Filtering?

In RAG-Systemen mit vielen Dokumenten (über 100 oder mehr) entstehen zusätzliche Probleme:

- irrelevante Treffer aus anderen Dokumenten  
- vermischte Quellen  
- unpräzise kontextbasierte Antworten  
- unnötig lange Prompts und höhere Tokenkosten

## Lösung: Metadata Filtering

Filtering ermöglicht es, Retrieval-Ergebnisse auf bestimmte Dokumente oder Kategorien zu beschränken, zum Beispiel:

- `source = "embeddings.md"`
- `category = "finance"`
- `date > 2023-01-01"`

Filtering sorgt dafür, dass nur relevante Dokumente durchsucht werden und verbessert die Präzision deutlich.

---

# Warum Hybrid Retrieval sinnvoll ist

Professionelle RAG-Systeme kombinieren:

| Technik     | Vorteil                          |
|-------------|-----------------------------------|
| BM25        | exakte Begriffe, Keywords         |
| Embeddings  | semantische Ähnlichkeit           |
| Filtering   | Kontrolle über Dokumentenquellen  |

Die Kombination ergibt ein robustes, skalierbares und präzises Retrieval-System.

---

# Fazit

BM25 und Filtering adressieren die größten Schwächen reiner Embeddings:

- exakte Begriffe und Fachvokabular

- technische Dokumentation

- juristische Referenzen

- große Datenmengen

- gezielte Quellensteuerung

Hybrid Retrieval ist ein wichtiger Bestandteil jeder produktiven RAG-Architektur.
Durch die Kombination aus Keyword-Suche, semantischer Suche und Metadaten-Filterung wird Retrieval deutlich präziser, stabiler und sicherer.

---

# Beispiel: Hybrid Retrieval (BM25 + Embeddings)

Weiterführende Links

Hier finden Sie vertiefende Ressourcen zu BM25, Hybrid Retrieval und Filtering:

- LlamaIndex BM25 Retriever & Metadata Filtering
https://developers.llamaindex.ai/python/examples/retrievers/bm25_retriever/#bm25-retriever--metadatafiltering

- LangChain BM25 Retriever
https://docs.langchain.com/oss/python/integrations/retrievers/bm25

- GeeksForGeeks: BM25 Erklärung
https://www.geeksforgeeks.org/nlp/what-is-bm25-best-matching-25-algorithm/

- Hybrid Search kombiniert BM25 & Embeddings
https://medium.com/etoai/hybrid-search-combining-bm25-and-semantic-search-for-better-results-with-lan-1358038fe7e6

- HuggingFace BM25S
https://huggingface.co/blog/xhluca/bm25s

- Rankify BM25 Retriever
https://rankify.readthedocs.io/en/stable/api/retrievers/bm25/

---

# Beispiel: Hybrid Retrieval (BM25 + Embeddings)

```python
from bm25_search import build_bm25_index, search_bm25
from openai import OpenAI
import chromadb

client = OpenAI()
chroma_client = chromadb.Client()
collection = chroma_client.get_collection("multi_rag")

# Alle Texte aus Chroma extrahieren
all_docs = collection.get()["documents"]

# BM25 Index bauen
ix = build_bm25_index(all_docs)

def hybrid_search(query, top_k=3):
    # Embedding Search
    emb = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    ).data[0].embedding

    vec_res = collection.query(
        query_embeddings=[emb],
        n_results=top_k
    )["documents"][0]

    # BM25 Search
    keyword_res = search_bm25(ix, query, top_k)

    # Kombinieren und Duplikate entfernen
    combined = list(dict.fromkeys(vec_res + keyword_res))

    return combined

print(hybrid_search("Was ist ein Embedding?"))

