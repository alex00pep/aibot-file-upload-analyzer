# Pandas DataFrame Analyzer - Prompt-driven analysis with PandasAI

## Dependencies install 

```bash
poetry add PyPDF2 langchain chromadb llama-cpp-python pdfminer.six InstructorEmbedding sentence-transformers faiss-cpu huggingface_hub
transformers streamlit streamlit-extras python-dotenv pandasai
```

## Run the application
Udate the file .env with the proper HUGGINGFACE_API_KEY value

```bash
streamlit run main.py
```

The browser instance will be opened with Pandas Dataframe Analyzer app running

## Application Usage
Write different questions to PandasAI to analyze dataset imported about Titanic passengers:

What was the average Fare price?
Give me a summary of this dataset
Who was more likely to survive, males or females? How much more likely?
Plot the survival counts for males and females