# Search Engine Ranking

## Python Environment
To run the code, you need to create a virtual environment and install the required packages. 
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Running Tests
To run the tests, you can use the following command:
```bash
python -m unittest
```
To run the tests with verbose output, you can use the following command:
```bash
python -m unittest -v
```
To run the tests with coverage, you can use the following command:
```bash
coverage run -m unittest
```

## Create cache
To create a Redis cache instance via Docker, you can use the following command:
```bash
docker run -p 6379:6379 -it redis/redis-stack:latest
```
