# blooapp-api

This is the API server for the blooapp; to return the recycling instructions for items in image using Retrieval Augmented Generation (RAG) with Langchain.

## Requirements

- > = Python 3.10
- Poetry
- Linux (ubuntu) environment
- Docker

## Setup (Linux)

Linux is used as the development environment as the `jq` package have issues when installing via poetry in windows.
Use windows subsystem for linux (WSL) for development.

```bash
sudo apt update && sudo apt upgrade

```

Install poetry in wsl

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Install dependencies

```bash
poetry shell
poetry install
```

## Running the server in Poetry Shell (launch LangServe)

```bash
langchain serve --port 8080
```

### Exiting poetry shell

```bash
exit
```

## Running the server in Docker Compose

```bash
docker-compose up --build -d
```

## Teardown

```bash
docker-compose down
```
