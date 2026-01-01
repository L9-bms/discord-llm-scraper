import math
from time import sleep
from typing import Optional

import requests
from pydantic import BaseModel
from rich import print

DISCORD_BASE_URL = "https://discord.com/api/v9"


class Message(BaseModel):
	id: str
	content: str
	author: str
	referenced_channel_id: Optional[str] = None
	referenced_message_id: Optional[str] = None

	@classmethod
	def from_api_response(cls, discord_msg: dict) -> Message:
		referenced_channel_id = None
		referenced_message_id = None

		if "message_reference" in discord_msg:
			referenced_channel_id = discord_msg["message_reference"].get("channel_id")
			referenced_message_id = discord_msg["message_reference"].get("message_id")

		return cls(
			id=discord_msg["id"],
			content=discord_msg["content"],
			author=discord_msg["author"]["username"],
			referenced_channel_id=referenced_channel_id,
			referenced_message_id=referenced_message_id,
		)

	def has_reference(self) -> bool:
		return self.referenced_message_id is not None and self.referenced_channel_id is not None


class Chain(BaseModel):
	topic: str
	messages: list[Message] = []

	def to_dict(self) -> dict:
		return {
			"topic": self.topic,
			"messages": [{"id": msg.id, "content": msg.content, "author": msg.author} for msg in self.messages],
		}


def fetch_messages(
	session: requests.Session,
	guild_id: str,
	content: str,
	channel_ids: list[str],
	limit: int,
	slop: int,
	include_bots: bool,
	include_webhooks: bool,
	oldest_first: bool,
) -> list[Message]:
	"""https://docs.discord.food/resources/message#search-guild-messages"""
	author_type: list[str] = []
	if not include_bots:
		author_type.append("-bot")
	if not include_webhooks:
		author_type.append("-webhook")

	params: dict[str, str | list[str]] = {
		"content": content,
		"channel_id": channel_ids,
		"slop": str(slop),
		"author_type": author_type,
		"sort_order": "asc" if oldest_first else "desc",
	}

	messages: list[Message] = []

	for i in range(math.ceil(limit / 25)):
		payload = params.copy()
		payload["limit"] = str(min(25, limit - i * 25))
		payload["offset"] = str(i * 25)

		r = session.get(f"{DISCORD_BASE_URL}/guilds/{guild_id}/messages/search", params=payload)

		match r.status_code:
			case 200:
				for message_group in r.json()["messages"]:
					for message in message_group:
						messages.append(Message.from_api_response(message))

				print(f"{i * 25 + min(25, limit - i * 25)} ", end="")
			case 202:
				print(f"not yet indexed, retry after {r.json()['retry_after']} seconds")
				return []
			case _:
				print(f"unexpected status code {r.status_code}")
				print(r.json())
				return []

		# todo: properly handle rate limiting
		sleep(1)

	print("done!")
	return messages


def traverse_reply_chain(
	session: requests.Session,
	root_message: Message,
	context_messages: int,
	seen_ids: set[str],
	chain_depth: int,
	chain_path: Optional[list[str]] = None,
) -> list[Message]:
	"""https://docs.discord.food/resources/message#get-messages"""
	chain: list[Message] = []
	seen_ids.add(root_message.id)
	chain.append(root_message)
	current_message = root_message

	if chain_path is None:
		chain_path = [root_message.id]
		prefix = f"[{root_message.id}]"
	else:
		chain_path = chain_path + [root_message.id]
		prefix = f"[{'->'.join(chain_path)}]"

	if not current_message.has_reference():
		print(f"{prefix} no message reference, adding as-is")
		return chain

	while current_message.has_reference():
		print(f"{prefix} fetching {context_messages} messages around {current_message.referenced_message_id}...")
		r = session.get(
			f"{DISCORD_BASE_URL}/channels/{current_message.referenced_channel_id}/messages",
			params={"around": current_message.referenced_message_id, "limit": context_messages},
		)

		fetched_messages = r.json()
		next_message = None

		for fetched in fetched_messages:
			if fetched["id"] == current_message.referenced_message_id:
				next_message = fetched
				break

		message_block: list[Message] = []
		for fetched in fetched_messages:
			if fetched["id"] not in seen_ids:
				seen_ids.add(fetched["id"])
				message_block.append(Message.from_api_response(fetched))

		if chain_depth > 1:
			for msg in message_block:
				if msg.has_reference() and msg.id != current_message.referenced_message_id:
					print(f"{prefix} traversing side chain to {msg.id}")
					side_chain = traverse_reply_chain(
						session, msg, context_messages, seen_ids, chain_depth - 1, chain_path
					)
					for side_msg in side_chain:
						if side_msg.id not in [m.id for m in chain] and side_msg.id not in [
							m.id for m in message_block
						]:
							message_block.append(side_msg)

		# should already be returned in reverse chronological order by discord
		message_block.reverse()
		chain = message_block + chain

		if next_message is None:
			print(f"{prefix} could not find next message to continue chain!")
			break

		current_message = Message.from_api_response(next_message)

		sleep(1)

	print(f"{prefix} reply chain ended")

	sleep(1)

	return chain
