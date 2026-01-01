import json
import os
from typing import Any

from search import Chain


def load_json(filename: str) -> Any:
	with open(filename, "r") as f:
		return json.load(f)


def save_json(filename: str, data: Any) -> None:
	with open(filename, "w") as f:
		json.dump(data, f)


def load_seen_ids(filename: str) -> set[str]:
	seen_ids: set[str] = set()
	try:
		existing_chains = load_json(filename)
		for chain_data in existing_chains:
			chain = Chain.model_validate(chain_data)
			for msg in chain.messages:
				seen_ids.add(msg.id)
	except FileNotFoundError:
		pass

	return seen_ids


def extend_array_json(filename: str, new_data) -> None:
	if os.path.exists(filename) and os.path.getsize(filename) > 0:
		data = load_json(filename)
	else:
		data = []

	data.extend(new_data)

	save_json(filename, data)
