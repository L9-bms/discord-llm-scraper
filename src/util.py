import json
import os

from discord import Chain


def load_seen_ids(filename: str) -> set[str]:
	seen_ids: set[str] = set()
	try:
		with open(filename, "r") as f:
			existing_chains = json.load(f)
			for chain_data in existing_chains:
				chain = Chain.model_validate(chain_data)
				for msg in chain.messages:
					seen_ids.add(msg.id)
	except FileNotFoundError:
		pass

	return seen_ids


def extend_array_json(filename: str, new_data) -> None:
	if os.path.exists(filename) and os.path.getsize(filename) > 0:
		with open(filename, "r") as file:
			data = json.load(file)
	else:
		data = []

	data.extend(new_data)

	with open(filename, "w") as file:
		json.dump(data, file)
