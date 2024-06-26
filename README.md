# blooapp-api

## Setup (Linux)

Linux is used as the development environment as the `jq` package have issues when installing via poetry in windows.

```bash
sudo apt update && sudo apt upgrade

```

Install python3.11 in wsl

```bash
sudo apt update
sudo apt install software-properties-common
```

2. Add the Deadsnakes PPA:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
```

Install Python 3.11

```bash
sudo apt install python3.11
```

1. Add Python 3.11 to update alternatives:

```bash
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
```

2. Configure the default Python version:

```bash
sudo update-alternatives --set python3 /usr/bin/python3.11
```

4. Verify the Installation

```bash
python3 --version
```

This should output Python 3.11.x.

5. Install pip for Python 3.11
   Ensure that pip is installed for the new Python version:

```bash
sudo apt install python3.11-distutils
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
```

Install poetry in wsl

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

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
docker-compose up --build -d
```

## Teardown

```bash
docker-compose down
```
