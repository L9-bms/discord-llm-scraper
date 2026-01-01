from typing import Annotated

import requests
import typer
from dotenv import load_dotenv
from openai import OpenAI
from rich import print
from rich.live import Live
from rich.markdown import Markdown

from embed import compute_embeddings, find_similar_pairs, merge_summaries, search_embeddings
from query import answer_query_streaming, create_prompt
from search import Chain, fetch_messages, traverse_reply_chain
from summary import Summary, summarise_chain
from util import extend_array_json, load_json, load_seen_ids, save_json

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
	threads = load_json("messages.json")
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

	save_json("embedded.json", data)
	print("wrote to embedded.json")


@app.command(
	help="Takes a query, embeds it, retrieves the top-k ranked results which are used as context to answer the query."
)
def ask(
	query: str,
	openrouter_key: Annotated[str, typer.Option(envvar="OPENROUTER_API_KEY", prompt=True)],
	top_k: Annotated[int, typer.Option(help="How many results are used as context to answer the query")] = 5,
	embedding_model: Annotated[str, typer.Option()] = "qwen/qwen3-embedding-8b",
	answer_model: Annotated[str, typer.Option()] = "openai/gpt-4o-mini",
	show_prompt: Annotated[bool, typer.Option(help="Show the prompt before it's fed into the answer model")] = False,
) -> None:
	data = load_json("embedded.json")
	print("loaded embeddings from embedded.json")

	summaries = [Summary.model_validate(item) for item in data["summaries"]]
	embeddings = [item["embedding"] for item in data["summaries"]]

	client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)

	results = search_embeddings(client, query, summaries, embeddings, top_k, embedding_model)
	print(f"found {len(results)} results with relatedness {['{:.3f}'.format(r[1]) for r in results]}")

	prompt = create_prompt(query, results)
	if show_prompt:
		print("\n[bold]prompt:[/bold]\n")
		print(prompt)

	print("\n[bold]answer:[/bold]\n")
	full_response = ""
	with Live(Markdown(""), refresh_per_second=10) as live:
		for chunk in answer_query_streaming(client, prompt, answer_model):
			full_response += chunk
			live.update(Markdown(full_response))
	print()


@app.command(help="Merges similar summaries in embedded.json")
def merge(
	openrouter_key: Annotated[str, typer.Option(envvar="OPENROUTER_API_KEY", prompt=True)],
	threshold: Annotated[float, typer.Option(help="Similarity threshold for merging", min=0.5, max=1.0)] = 0.9,
	embedding_model: Annotated[
		str, typer.Option(help="Model used to re-embed the merged summaries")
	] = "qwen/qwen3-embedding-8b",
	merge_model: Annotated[str, typer.Option()] = "openai/gpt-4o-mini",
	dry_run: Annotated[bool, typer.Option(help="Show what would be merged without actually merging")] = False,
) -> None:
	data = load_json("embedded.json")
	print("loaded embeddings from embedded.json")

	summaries = [Summary.model_validate(item) for item in data["summaries"]]
	embeddings = [item["embedding"] for item in data["summaries"]]

	print(f"finding similar pairs with threshold >= {threshold}...")
	similar_pairs = find_similar_pairs(embeddings, threshold)

	if not similar_pairs:
		print("no similar summaries found :)")
		return

	print(f"found {len(similar_pairs)} similar pairs")

	if dry_run:
		for i, j, score in similar_pairs:
			print(f"\nSimilarity: {score:.3f}")
			print(f"  [{i}] Topic: {summaries[i].topic}")
			print(f"      Info: {summaries[i].info[:100]}...")
			print(f"  [{j}] Topic: {summaries[j].topic}")
			print(f"      Info: {summaries[j].info[:100]}...")
		return

	client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)

	to_remove = set()
	merged_summaries = list(summaries)
	merged_embeddings = list(embeddings)

	for idx, (i, j, score) in enumerate(similar_pairs):
		if i in to_remove or j in to_remove:
			continue

		print(f"[{idx + 1}/{len(similar_pairs)}] merging summaries {i} and {j} (similarity: {score:.3f})")

		merged_summaries[i] = merge_summaries(client, summaries[i], summaries[j], merge_model)
		to_remove.add(j)

	final_summaries = [s for idx, s in enumerate(merged_summaries) if idx not in to_remove]
	final_embeddings = [e for idx, e in enumerate(merged_embeddings) if idx not in to_remove]

	print(f"\nreduced from {len(summaries)} to {len(final_summaries)} summaries")

	print("recomputing embeddings for merged summaries...")
	indices_to_recompute = [i for i, j, _ in similar_pairs if i not in to_remove]
	if indices_to_recompute:
		old_to_new = {}
		new_idx = 0
		for old_idx in range(len(summaries)):
			if old_idx not in to_remove:
				old_to_new[old_idx] = new_idx
				new_idx += 1

		for old_idx in set(indices_to_recompute):
			new_idx = old_to_new[old_idx]
			new_emb = compute_embeddings(client, [final_summaries[new_idx]], embedding_model)
			final_embeddings[new_idx] = new_emb[0]

	data = {
		"summaries": [
			{"topic": s.topic, "info": s.info, "embedding": emb} for s, emb in zip(final_summaries, final_embeddings)
		],
		"metadata": {
			"count": len(final_summaries),
			"embedding_dim": len(final_embeddings[0]) if final_embeddings else 0,
		},
	}

	save_json("embedded.json", data)
	print("wrote to embedded.json")


if __name__ == "__main__":
	app()
