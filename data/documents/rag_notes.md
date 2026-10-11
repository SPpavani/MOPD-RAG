# Notes on retrieval-augmented generation (sample document)

Retrieval-augmented generation (RAG) gives a language model access to documents it was not trained on. The system splits documents into chunks, finds the chunks most relevant to a question, and puts them in the prompt so the model can answer from them.

Chunk size matters. Small chunks are precise but can lose context; large chunks keep context but dilute the match. Overlap between chunks reduces the chance that an answer is cut in half.

TF-IDF scores a chunk by how often the query words appear in it, weighted by how rare those words are across all chunks. It needs no model and works well when questions share exact words with the document.

A good RAG system says when the documents do not contain the answer, instead of guessing.
