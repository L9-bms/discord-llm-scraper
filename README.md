# discord-llm-scraper

creates a queryable knowledgebase based on a specific topic that was talked about in a discord channel

## why

often i will try to contribute to an open source project by first reading the docs, and then the source, and then joining the discord to understand why the code is written the way it is, and finally writing my PR.

i find the most time consuming part is searching through the discord and traversing though reply chains, threads and conversations to gain knowledge that never made it into the docs ("tribal knowledge") and discover previous attempts on implementing the feature.

## how

1. use the discord user api to search for messages containing relevant strings
2. traverse reply chains
3. extract knowledge
4. maybe integrate with code and documentation
5. build vector database (ingestion)
6. now able to "query the conversation" through RAG + reranking -> prompt
7. maybe make it agentic RAG?

## problems

unable to discard incorrect information that has been corrected
