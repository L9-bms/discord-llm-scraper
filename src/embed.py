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


def find_similar_pairs(embeddings: List[List[float]], threshold: float) -> List[tuple[int, int, float]]:
	"""returns a list of tuples of (index1, index2, similarity) in descending order"""
	embeddings_array = np.array(embeddings)

	# cosine similarity matrix
	norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
	normalized = embeddings_array / norms
	similarity_matrix = np.dot(normalized, normalized.T)

	similar_pairs = []
	n = len(embeddings)
	for i in range(n):
		for j in range(i + 1, n):
			if similarity_matrix[i, j] >= threshold:
				similar_pairs.append((i, j, float(similarity_matrix[i, j])))

	similar_pairs.sort(key=lambda x: x[2], reverse=True)
	return similar_pairs


def merge_summaries(client: OpenAI, summary1: Summary, summary2: Summary, model: str) -> Summary:
	response = client.responses.parse(
		model=model,
		instructions="You are a knowledge consolidation system.",
		input=f"""Merge these two similar summaries into one:

Summary 1:
Topic: {summary1.topic}
Info: {summary1.info}

Summary 2:
Topic: {summary2.topic}
Info: {summary2.info}""",
		text_format=Summary,
	)

	merged = response.output_parsed
	if not merged:
		return summary1

	return Summary(topic=merged.topic, info=merged.info)
