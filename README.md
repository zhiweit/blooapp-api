# blooapp-api

This is the API server for the blooapp; to return the recycling instructions for items in image using Retrieval Augmented Generation (RAG) with Langchain.


## Chat architecture
![chat-architecture](https://github.com/user-attachments/assets/cd0a8ea0-1615-46e8-b6c3-07a177f8076a)

## Chat Feature
Demo of the chat feature

https://github.com/user-attachments/assets/529da66c-3e0c-4655-a114-bc6718fbded0




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

<a href="https://mightymagnus.notion.site/Bloo-API-Documentation-d9fd5834240346b9acc5274b0112e8d5" target="_blank">Further documentation in Notion</a>

