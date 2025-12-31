import json
from typing import Annotated

import requests
import typer
from dotenv import load_dotenv
from openai import OpenAI
from rich import print
from rich.live import Live
from rich.markdown import Markdown

from discord import Chain, fetch_messages, traverse_reply_chain
from embedding import compute_embeddings, search_embeddings
from query import answer_query_streaming
from summarise import Summary, summarise_chain
from util import extend_array_json, load_seen_ids

load_dotenv()
app = typer.Typer()


@app.command(
	help="Searches Discord server for messages containing content, traverses the reply chain and then exports the chains to messages.json"
)
def search(
	guild_id: str,
	content: Annotated[str, typer.Argument(help="Content to filter messages by")],
	token: Annotated[str, typer.Option(help="Discord authentication token", prompt=True, envvar="DISCORD_TOKEN")],
	channel_id: Annotated[list[str] | None, typer.Option(help="Filter messages by these channel IDs")] = None,
	limit: Annotated[int, typer.Option(help="Number of messages to search for", prompt=True, max=10000)] = 25,
	slop: Annotated[int, typer.Option(help="Max number of words to skip between matching tokens", max=100)] = 2,
	context_amount: Annotated[int, typer.Option(help="Number of context messages to fetch at each chain step")] = 10,
	include_bots: Annotated[bool, typer.Option()] = False,
	include_webhooks: Annotated[bool, typer.Option()] = False,
	oldest_first: Annotated[bool, typer.Option()] = False,
) -> None:
	session = requests.Session()
	session.headers.update({"Authorization": token})

	print(f"searching for {content} in {guild_id}...")
	messages = fetch_messages(
		session,
		guild_id,
		content,
		channel_id if channel_id else [],
		limit,
		slop,
		include_bots,
		include_webhooks,
		oldest_first,
	)
	if not messages:
		print("no messages found :(")
		return

	print("processing messages and traversing reply chains...")
	seen_ids = load_seen_ids("messages.json")
	if seen_ids:
		print(f"loaded {len(seen_ids)} existing message ids")

	chains: list[Chain] = []
	for message in messages:
		if message.id in seen_ids:
			print(f"message {message.id} already exists")
			continue
		reply_chain_messages = traverse_reply_chain(session, message, context_amount, seen_ids)
		chain = Chain(topic=content, messages=reply_chain_messages)
		chains.append(chain)

	print("done!")

	extend_array_json("messages.json", [c.to_dict() for c in chains])
	print("wrote to messages.json")


@app.command(
	help="Reads conversations from messages.json, summarises them into knowledge chunks and then computes embeddings for each summary, then saves into embedded.json"
)
def extract(
	openrouter_key: Annotated[str, typer.Option(envvar="OPENROUTER_API_KEY", prompt=True)],
	summary_model: Annotated[str, typer.Option()] = "openai/gpt-oss-120b",
	embedding_model: Annotated[str, typer.Option()] = "qwen/qwen3-embedding-8b",
) -> None:
	file = open("messages.json", "r")
	threads = json.load(file)
	print("loaded threads from messages.json")

	client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)

	summaries: list[Summary] = []
	for i, thread in enumerate(threads):
		print(f"[{i}] ", end="")
		summaries.extend(summarise_chain(client, Chain.model_validate(thread), summary_model))

	if not summaries:
		print("no summaries generated :(")

	print(f"computing embeddings for {len(summaries)} summaries...")
	embeddings = compute_embeddings(client, summaries, embedding_model)
	print("done!")

	data = {
		"summaries": [{"topic": s.topic, "info": s.info, "embedding": emb} for s, emb in zip(summaries, embeddings)],
		"metadata": {"count": len(summaries), "embedding_dim": len(embeddings[0]) if embeddings else 0},
	}

	file = open("embedded.json", "w")
	json.dump(data, file)
	print("wrote to embedded.json")


@app.command(
	help="Takes a query, embeds it, retrieves the top-k ranked results which are used as context to answer the query."
)
def ask(
	query: str,
	openrouter_key: Annotated[str, typer.Option(envvar="OPENROUTER_API_KEY", prompt=True)],
	top_k: Annotated[int, typer.Option()] = 5,
	embedding_model: Annotated[str, typer.Option()] = "qwen/qwen3-embedding-8b",
	answer_model: Annotated[str, typer.Option()] = "openai/gpt-4o-mini",
) -> None:
	file = open("embedded.json", "r")
	data = json.load(file)
	print("loaded embeddings from embedded.json")

	summaries = [Summary(topic=item["topic"], info=item["info"]) for item in data["summaries"]]
	embeddings = [item["embedding"] for item in data["summaries"]]

	client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)

	results = search_embeddings(client, query, summaries, embeddings, top_k, embedding_model)
	print(f"found {len(results)} results with relatedness {['{:.3f}'.format(r[1]) for r in results]}")

	print("\n[bold]answer:[/bold]\n")
	full_response = ""
	with Live(Markdown(""), refresh_per_second=10) as live:
		for chunk in answer_query_streaming(client, query, results, answer_model):
			full_response += chunk
			live.update(Markdown(full_response))
	print()


if __name__ == "__main__":
	app()
