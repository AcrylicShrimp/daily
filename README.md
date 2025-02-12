# daily

My toy project to make an agent that answers questions based on web search.

## Features

- Search the web for information
- Index documents
- Answer questions based on indexed documents

## How it indexes documents

1. Fetch URLs from the duckduckgo search results
1. Download the HTML body of the URLs
1. Remove unnecessary tags like `script`, `style`, etc.
1. Run LLM with the URL, title, description, and first 1024 characters of text of the HTML body to generate some piece of metadata
1. Extract text from web pages using [DOM Based Content Extraction via Text Density](https://ofey.me/assets/pdf/cetd-sigir11.pdf)
1. Chunk the text into smaller chunks
1. Format all the chunks to contain the URL, title, description, and metadata
1. Store the chunks in a vector database

## Usage

First, install the dependencies:

```bash
pip install -r requirements.txt
```

Then, run the script:

```bash
python src/main.py
```

### Environment variables

- `OPENAI_API_KEY`: The API key for the OpenAI API
- `ANTHROPIC_API_KEY`: The API key for the Anthropic API
- `VOYAGE_API_KEY`: The API key for the Voyage API
- `CHROMA_DIR`: The directory to store the vector database
