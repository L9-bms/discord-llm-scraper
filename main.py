import math
from time import sleep
import typer
from rich import print
from typing import Annotated
import requests
import json

app = typer.Typer()

DISCORD_BASE_URL = "https://discord.com/api/v9"


def fetch_messages(
	session: requests.Session,
	guild_id: str,
	content: str,
	amount: int,
	slop: int,
	include_bots: bool,
	include_webhooks: bool,
	oldest_first: bool,
) -> list[dict[str, str]]:
	author_type: list[str] = []
	if not include_bots:
		author_type.append("-bot")
	if not include_webhooks:
		author_type.append("-webhook")

	params: dict[str, str | list[str]] = {
		"content": content,
		"slop": str(slop),
		"author_type": author_type,
		"sort_order": "asc" if oldest_first else "desc",
	}

	messages: list[dict[str, str]] = []

	for i in range(math.ceil(amount / 25)):
		payload = params.copy()
		payload["limit"] = str(min(25, amount - i * 25))
		payload["offset"] = str(i * 25)

		r = session.get(f"{DISCORD_BASE_URL}/guilds/{guild_id}/messages/search", params=payload)

		match r.status_code:
			case 200:
				for message_group in r.json()["messages"]:
					for message in message_group:
						fetched_message: dict[str, str] = {
							"id": message["id"],
							"content": message["content"],
							"author": message["author"]["username"],
						}
						if "message_reference" in message:
							fetched_message["referenced_channel_id"] = message["message_reference"]["channel_id"]
							fetched_message["referenced_message_id"] = message["message_reference"]["message_id"]
						messages.append(fetched_message)

				print(f"{i * 25 + min(25, amount - i * 25)} ", end="")
			case 202:
				print(f"not yet indexed, retry after {r.json()['retry_after']} seconds")
				return []
			case _:
				print(f"Unexpected status code {r.status_code}")
				print(r.json())
				return []

		# todo: properly handle rate limiting
		sleep(1)

	print("done!")
	return messages


def traverse_reply_chain(session: requests.Session, root_message: dict[str, str]) -> list[dict[str, str]]:
	chain: list[dict[str, str]] = [{"content": root_message["content"], "author": root_message["author"]}]
	current_message = root_message

	while "referenced_message_id" in current_message and "referenced_channel_id" in current_message:
		print(f"[{root_message['id']}] fetching message {current_message['referenced_message_id']}...")
		r = session.get(
			f"{DISCORD_BASE_URL}/channels/{current_message['referenced_channel_id']}/messages",
			params={"around": current_message["referenced_message_id"], "limit": 1},
		)

		fetched = r.json()[0]
		current_message = {
			"id": fetched["id"],
			"content": fetched["content"],
			"author": fetched["author"]["username"],
		}
		if "message_reference" in fetched:
			current_message["referenced_channel_id"] = fetched["message_reference"]["channel_id"]
			current_message["referenced_message_id"] = fetched["message_reference"]["message_id"]

		chain.insert(0, {"content": current_message["content"], "author": current_message["author"]})

		# todo: rate limiting
		sleep(1)

	print(f"[{root_message['id']}] reply chain ended")

	# todo: rate limiting
	sleep(1)

	return chain


@app.command(
	help="Searches Discord server for messages containing content, traverses the reply chain and then exports the chains to messages.json"
)
def search(
	guild_id: str,
	content: Annotated[str, typer.Argument(help="Content to filter messages by")],
	token: Annotated[str, typer.Option(help="Discord authentication token", prompt=True)],
	amount: Annotated[int, typer.Option(help="Number of messages to search for", prompt=True, max=10000)] = 25,
	slop: Annotated[int, typer.Option(help="Max number of words to skip between matching tokens", max=100)] = 2,
	include_bots: Annotated[bool, typer.Option()] = False,
	include_webhooks: Annotated[bool, typer.Option()] = False,
	oldest_first: Annotated[bool, typer.Option()] = False,
):
	session = requests.Session()
	session.headers.update({"Authorization": token})

	print(f"searching for {content} in {guild_id}...")
	messages = fetch_messages(session, guild_id, content, amount, slop, include_bots, include_webhooks, oldest_first)
	if not messages:
		return

	print("traversing reply chains...")
	chains = [traverse_reply_chain(session, msg) for msg in messages]

	print("done!")

	with open("messages.json", "w") as file:
		json.dump(chains, file)
	print("wrote to messages.json")


if __name__ == "__main__":
	app()
