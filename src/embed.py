from typing import List

import numpy as np
from openai import OpenAI

from summary import Summary


def compute_embeddings(client: OpenAI, summaries: list[Summary], model: str) -> List[List[float]]:
	texts = [f"{s.topic}: {s.info}" for s in summaries]

	response = client.embeddings.create(input=texts, model=model)

	embeddings = [item.embedding for item in response.data]

	return embeddings


def search_embeddings(
	client: OpenAI, query: str, summaries: list[Summary], embeddings, top_k: int, model: str
) -> List[tuple[Summary, float]]:
	embeddings_array = np.array(embeddings)

	response = client.embeddings.create(input=[query], model=model)
	query_embedding = np.array(response.data[0].embedding)

	similarities = np.dot(embeddings_array, query_embedding) / (
		np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_embedding)
	)

	top_indices = np.argsort(similarities)[-top_k:][::-1]

	results = [(summaries[i], float(similarities[i])) for i in top_indices]
	return results
