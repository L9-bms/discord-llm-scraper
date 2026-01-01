from openai import OpenAI
from pydantic import BaseModel
from rich import print

from search import Chain


class Summary(BaseModel):
	topic: str
	info: str


class Output(BaseModel):
	summaries: list[Summary]


def summarise_chain(client: OpenAI, chain: Chain, model: str) -> list[Summary]:
	conversation = ""
	for message in chain.messages:
		conversation += f"\n{message.author}: {message.content}"

	response = client.responses.parse(
		model=model,
		instructions="You are a knowledge extraction system. Return only valid JSON.",
		input=f"""Extract facts that will be useful to retrieve later from the conversation about "{chain.topic}" below.

Rules:
1. EXCLUDE all questions, requests for help, user goals, or descriptions of what someone is trying to do.
2. ONLY include definitive statements, answers, or confirmed details about how "{chain.topic}" functions.
3. If the conversation only consists of people asking how to do something, return an empty array: {{"summaries": []}}
4. Write in concise third-person. Do not invent details.

CONVERSATION: {conversation}
""",
		text_format=Output,
	)

	output = response.output_parsed
	if not output:
		return []

	print(f"extracted {len(output.summaries)} summaries")

	return output.summaries
