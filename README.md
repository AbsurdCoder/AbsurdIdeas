# AbsurdIdeas


Create .env file:

OPENAI_API_KEY=sk-your-key-here

Build and run docker:

docker build -t aiaas .
docker run -p 8000:8999 --env-file .env aiaas

