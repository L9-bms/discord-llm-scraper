from typing import List

from openai import OpenAI

from summarise import Summary


def answer_query_streaming(
	client: OpenAI,
	query: str,
	results: List[tuple[Summary, float]],
	model: str,
):
	context_parts = []
	for i, (summary, score) in enumerate(results, 1):
		context_parts.append(f"{i}. Topic: {summary.topic}. Info: {summary.info}")

	context = "\n".join(context_parts)

	# Stream the response
	stream = client.responses.create(
		model=model,
		instructions="You are an assistant that answers questions based on given knowledge.",
		input=f"""Use the knowledge below to answer the subsequent question. If the answer cannot be found in the knowledge, write "I could not find an answer.

Knowledge snippets:
{context}

Question: {query}""",
		stream=True,
	)

	for event in stream:
		if event.type == "response.output_text.delta":
			yield event.delta
