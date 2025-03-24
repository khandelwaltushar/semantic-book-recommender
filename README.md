# Semantic Book Recommender

A sophisticated book recommendation system that uses semantic search and emotional analysis to suggest books based on user preferences. This project combines natural language processing, sentiment analysis, and vector search to provide personalized book recommendations.

## Features

- **Semantic Search**: Uses advanced language models to understand the meaning behind book descriptions and user queries
- **Emotional Analysis**: Recommends books based on emotional tones (Happy, Surprising, Angry, Suspenseful, Sad)
- **Category Filtering**: Filter recommendations by book categories
- **Interactive UI**: Built with Gradio for a user-friendly interface
- **Visual Recommendations**: Displays book covers with descriptions in a gallery format

## Prerequisites

- Python 3.8+
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/khandelwaltushar/semantic-book-recommender.git
cd semantic-book-recommender
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with your API keys and configurations.

## Usage

1. Start the application:
```bash
python gradio_app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:7860)

3. Enter a description of the kind of book you're looking for, select a category and emotional tone, and click "Find Recommendations"

## Project Structure

- `gradio_app.py`: Main application file with the Gradio interface
- `books_with_emotions.csv`: Dataset containing books with emotional analysis
- `books_with_categories.csv`: Dataset containing books with category information
- `tagged_description.txt`: Processed book descriptions for semantic search
- Various Jupyter notebooks for data exploration and model development:
  - `data-exploration.ipynb`
  `text-classification.ipynb`
  - `sentiment-analysis.ipynb`
  - `vector-search.ipynb`

## Technical Details

- Uses the `sentence-transformers/all-MiniLM-L6-v2` model for semantic embeddings
- Implements vector search using Chroma for efficient similarity search
- Processes book descriptions and metadata for enhanced recommendations
- Combines semantic similarity with emotional analysis for personalized results


## Acknowledgments

- Built with [Gradio](https://www.gradio.app/)
- Uses [LangChain](https://www.langchain.com/) for document processing
- Powered by [Hugging Face](https://huggingface.co/) models 