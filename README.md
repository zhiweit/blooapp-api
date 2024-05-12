# blooapp-api

## Setup

Install [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)

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
docker-compose up -d
```

## Teardown

```bash
docker-compose down
```
