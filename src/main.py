from typing import Annotated

import requests
import typer
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from requests import Session
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


class SearchTerms(BaseModel):
	terms: list[str]


def load_embeddings() -> tuple[list[Summary], list[list[float]]]:
	data = load_json("embedded.json")
	print("loaded embeddings from embedded.json")
	summaries = [Summary.model_validate(item) for item in data["summaries"]]
	embeddings = [item["embedding"] for item in data["summaries"]]
	return summaries, embeddings


def save_embeddings(summaries: list[Summary], embeddings: list[list[float]]) -> None:
	data = {
		"summaries": [{"topic": s.topic, "info": s.info, "embedding": emb} for s, emb in zip(summaries, embeddings)],
		"metadata": {"count": len(summaries), "embedding_dim": len(embeddings[0]) if embeddings else 0},
	}
	save_json("embedded.json", data)
	print("wrote to embedded.json")


def search_and_traverse(
	session: Session,
	guild_id: str,
	search_terms: list[str],
	channel_ids: list[str],
	limit: int,
	slop: int,
	include_bots: bool,
	include_webhooks: bool,
	oldest_first: bool,
	context_amount: int,
	chain_depth: int,
	seen_ids: set[str] | None = None,
) -> list[Chain]:
	if seen_ids is None:
		seen_ids = set()

	all_chains: list[Chain] = []

	for i, content in enumerate(search_terms, 1):
		print(f"\n[{i}/{len(search_terms)}] searching for {content}")
		messages = fetch_messages(
			session,
			guild_id,
			content,
			channel_ids,
			limit,
			slop,
			include_bots,
			include_webhooks,
			oldest_first,
		)

		if not messages:
			print("no messages found :(")
			continue

		print(f"found {len(messages)} messages")
		for message in messages:
			if message.id in seen_ids:
				continue

			reply_chain_messages = traverse_reply_chain(session, message, context_amount, seen_ids, chain_depth)
			chain = Chain(topic=content, messages=reply_chain_messages)
			all_chains.append(chain)

	return all_chains


def extract_summaries(client: OpenAI, chains: list[Chain], model: str) -> list[Summary]:
	summaries: list[Summary] = []
	for i, chain in enumerate(chains):
		print(f"[{i + 1}/{len(chains)}] ", end="")
		summaries.extend(summarise_chain(client, chain, model))
	return summaries


def merge_similar_summaries(
	client: OpenAI,
	summaries: list[Summary],
	embeddings: list[list[float]],
	threshold: float,
	merge_model: str,
	embedding_model: str,
	num_passes: int,
) -> tuple[list[Summary], list[list[float]]]:
	for pass_num in range(num_passes):
		print(f"\npass {pass_num + 1}/{num_passes}:")

		similar_pairs = find_similar_pairs(embeddings, threshold)

		if not similar_pairs:
			print(f"no summaries found with similarity >= {threshold}")
			break

		print(f"found {len(similar_pairs)} similar pairs")

		to_remove: set[int] = set()
		merged_summaries = list(summaries)
		merged_embeddings = list(embeddings)

		for idx, (i, j, score) in enumerate(similar_pairs):
			if i in to_remove or j in to_remove:
				continue

			print(f"[{idx + 1}/{len(similar_pairs)}] merging {i} and {j} (similarity: {score:.3f})")

			merged_summaries[i] = merge_summaries(client, summaries[i], summaries[j], merge_model)
			to_remove.add(j)

		summaries = [s for idx, s in enumerate(merged_summaries) if idx not in to_remove]
		embeddings = [e for idx, e in enumerate(merged_embeddings) if idx not in to_remove]

		print(f"reduced to {len(summaries)} summaries")

		indices_to_recompute = [i for i, j, _ in similar_pairs if i not in to_remove]
		if indices_to_recompute:
			print("recomputing embeddings for merged summaries...")

			old_to_new: dict[int, int] = {}
			new_idx = 0
			for old_idx in range(len(merged_summaries)):
				if old_idx not in to_remove:
					old_to_new[old_idx] = new_idx
					new_idx += 1

			for old_idx in set(indices_to_recompute):
				new_idx = old_to_new[old_idx]
				new_emb = compute_embeddings(client, [summaries[new_idx]], embedding_model)
				embeddings[new_idx] = new_emb[0]

		print("done!")

	return summaries, embeddings


def answer_query(
	client: OpenAI,
	query: str,
	summaries: list[Summary],
	embeddings: list[list[float]],
	top_k: int,
	embedding_model: str,
	answer_model: str,
	show_prompt: bool = False,
) -> None:
	results = search_embeddings(client, query, summaries, embeddings, top_k, embedding_model)
	print(f"found {len(results)} results with relatedness {['{:.3f}'.format(r[1]) for r in results]}")

	prompt = create_prompt(query, results)
	if show_prompt:
		print("\n[bold]Prompt:[/bold]\n")
		print(prompt)

	print("\n[bold]Answer:[/bold]\n")
	full_response = ""
	with Live(Markdown(""), refresh_per_second=10) as live:
		for chunk in answer_query_streaming(client, prompt, answer_model):
			full_response += chunk
			live.update(Markdown(full_response))
	print()


def generate_search_terms(client: OpenAI, query: str, num_terms: int, model: str) -> list[str] | None:
	response = client.responses.parse(
		model=model,
		instructions="You are a search term generator.",
		input=f"Generate {num_terms} specific search terms (1-2 keywords) that would appear in Discord messages relevant to answering this question: {query}",
		text_format=SearchTerms,
		extra_body={"plugins": [{"id": "response-healing"}]},
	)

	if not response.output_parsed or not response.output_parsed.terms:
		return None

	return response.output_parsed.terms[:num_terms]


@app.command(
	help="Searches Discord server for messages containing content, traverses the reply chain and then exports the chains to messages.json"
)
def search(
	guild_id: str,
	content: Annotated[str, typer.Argument(help="Content to filter messages by")],
	token: Annotated[str, typer.Option(help="Discord authentication token", prompt=True, envvar="DISCORD_TOKEN")],
	channel_id: Annotated[list[str] | None, typer.Option(help="Filter messages by these channel IDs")] = None,
	limit: Annotated[int, typer.Option(help="Number of messages to search for", prompt=True, max=10000)] = 10,
	slop: Annotated[int, typer.Option(help="Max number of words to skip between matching tokens", max=100)] = 2,
	context_amount: Annotated[int, typer.Option(help="Number of context messages to fetch at each chain step")] = 10,
	chain_depth: Annotated[
		int, typer.Option(help="Max depth for recursive reply traversal (1 = main only, 2 = 1 level side chains, etc.)")
	] = 2,
	include_bots: Annotated[bool, typer.Option()] = False,
	include_webhooks: Annotated[bool, typer.Option()] = False,
	oldest_first: Annotated[bool, typer.Option()] = False,
) -> None:
	session = requests.Session()
	session.headers.update({"Authorization": token})
	seen_ids = load_seen_ids("messages.json")
	if seen_ids:
		print(f"loaded {len(seen_ids)} existing message ids")

	print(f"searching for {content} in {guild_id}...")
	chains = search_and_traverse(
		session=session,
		guild_id=guild_id,
		search_terms=[content],
		channel_ids=channel_id if channel_id else [],
		limit=limit,
		slop=slop,
		include_bots=include_bots,
		include_webhooks=include_webhooks,
		oldest_first=oldest_first,
		context_amount=context_amount,
		chain_depth=chain_depth,
		seen_ids=seen_ids,
	)

	if not chains:
		print("no messages found :(")
		return

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
	chains = [Chain.model_validate(thread) for thread in threads]

	summaries = extract_summaries(client, chains, summary_model)
	if not summaries:
		print("no summaries generated :(")
		return

	print(f"computing embeddings for {len(summaries)} summaries...")
	embeddings = compute_embeddings(client, summaries, embedding_model)
	print("done!")

	save_embeddings(summaries, embeddings)


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
	summaries, embeddings = load_embeddings()
	client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)
	answer_query(client, query, summaries, embeddings, top_k, embedding_model, answer_model, show_prompt)


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
	summaries, embeddings = load_embeddings()

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
	original_count = len(summaries)

	summaries, embeddings = merge_similar_summaries(
		client, summaries, embeddings, threshold, merge_model, embedding_model, 1
	)

	print(f"\nreduced from {original_count} to {len(summaries)} summaries")
	save_embeddings(summaries, embeddings)


@app.command(
	help="""All-in-one: generates search terms from query, then performs search, extract, merge, and ask.
By default, will store the generated embeddings to use `ask` command later."""
)
def auto(
	guild_id: str,
	query: str,
	token: Annotated[str, typer.Option(help="Discord authentication token", prompt=True, envvar="DISCORD_TOKEN")],
	openrouter_key: Annotated[str, typer.Option(envvar="OPENROUTER_API_KEY", prompt=True)],
	channel_id: Annotated[list[str] | None, typer.Option(help="Filter messages by these channel IDs")] = None,
	search_limit: Annotated[int, typer.Option(help="Number of messages to search for per term", max=10000)] = 5,
	slop: Annotated[int, typer.Option(help="Max number of words to skip between matching tokens", max=100)] = 2,
	context_amount: Annotated[int, typer.Option(help="Number of context messages to fetch at each chain step")] = 10,
	chain_depth: Annotated[
		int, typer.Option(help="Max depth for recursive reply traversal (1 = main only, 2 = 1 level side chains, etc.)")
	] = 2,
	include_bots: Annotated[bool, typer.Option()] = False,
	include_webhooks: Annotated[bool, typer.Option()] = False,
	oldest_first: Annotated[bool, typer.Option()] = False,
	num_search_terms: Annotated[int, typer.Option(help="Number of search terms to generate")] = 3,
	top_k: Annotated[int, typer.Option(help="How many results are used as context to answer the query")] = 5,
	merge_threshold: Annotated[
		float, typer.Option(help="Similarity threshold for merging summaries", min=0.5, max=1.0)
	] = 0.9,
	merge_passes: Annotated[int, typer.Option(help="Number of merge passes to perform", min=0, max=10)] = 1,
	search_term_model: Annotated[str, typer.Option()] = "openai/gpt-4o-mini",
	summary_model: Annotated[str, typer.Option()] = "openai/gpt-oss-120b",
	embedding_model: Annotated[str, typer.Option()] = "qwen/qwen3-embedding-8b",
	merge_model: Annotated[str, typer.Option()] = "openai/gpt-4o-mini",
	answer_model: Annotated[str, typer.Option()] = "openai/gpt-4o-mini",
	show_prompt: Annotated[bool, typer.Option(help="Show the prompt before it's fed into the answer model")] = False,
	save: Annotated[bool, typer.Option(help="Save the generated embeddings to embedded.json")] = True,
) -> None:
	client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)
	session = requests.Session()
	session.headers.update({"Authorization": token})

	search_terms = generate_search_terms(client, query, num_search_terms, search_term_model)
	if not search_terms:
		print("[red]failed to generate search terms[/red]")
		return
	print(f"generated search terms: {search_terms}")

	chains = search_and_traverse(
		session=session,
		guild_id=guild_id,
		search_terms=search_terms,
		channel_ids=channel_id if channel_id else [],
		limit=search_limit,
		slop=slop,
		include_bots=include_bots,
		include_webhooks=include_webhooks,
		oldest_first=oldest_first,
		context_amount=context_amount,
		chain_depth=chain_depth,
	)
	if not chains:
		print("\n[red]no messages found from all chains :([/red]")
		return

	summaries = extract_summaries(client, chains, summary_model)

	embeddings = compute_embeddings(client, summaries, embedding_model)

	if merge_passes > 0:
		summaries, embeddings = merge_similar_summaries(
			client, summaries, embeddings, merge_threshold, merge_model, embedding_model, merge_passes
		)

	if save:
		save_embeddings(summaries, embeddings)

	answer_query(client, query, summaries, embeddings, top_k, embedding_model, answer_model, show_prompt)


if __name__ == "__main__":
	app()
