# AI-Powered answer Generator

## Project Overview
This project implements an AI-powered answer generator . It utilizes advanced language models and vector stores to understand and process user queries efficiently.


## Tech Stack
- **Language**: Python
- **AI Model**: LLaMa 3 70B (via Gemini API)
- **Vector Store**: ChromaDB
- **API Framework**: Flask

## Implementation Steps

### Setup

1. **CSV Data**: 
   - Utilize CSV files containing mock data, including:
     - Sales data (e.g., `sales_data.csv`)
     - Inventory data (e.g., `inventory.csv`)
   - These CSV files will be used for training and as sample data sources.

### Business Logic
1. **LLM Integration**: 
   - Utilize the Geminni API to access the LLaMa 3 70B model for natural language processing and query generation.

2. **Vector Store Setup**: 
   - Implement ChromaDB as the vector store for efficient similarity searches and data retrieval.


3. **Training**: 
   - Implement a training process that can use either CSV data or database schema information.

### REST API
- **Endpoints**: 
  - **GET /train**: Initiates the training process
  - **POST /ask**: Accepts a question and returns the generated answer and results

## Key Components

### ChromaDB (Vector Store)
- Manages collections for answers and CSV metadata.
- Provides methods for adding and retrieving similar questions, related DDL, and documentation.

### gemini API Integration
- Utilizes the Gemini API to access the LLaMa 3 70B model.
- Handles prompt submission and response processing.

### CSV Integration
- Processes CSV files to extract metadata and store it in the vector database.
- Allows for querying and retrieval of CSV-related information.

## Getting Started
### Setup and Running the Application

1. **Install Dependencies**:
   - Run the following command to install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```


2. **Run the Application**:
   - Navigate to the project folder and run:
     ```bash
     python app.py
     ```

3. **Using the API**:
   - To train the model:
     ```bash
     GET /train?csv=path_to_csv_file  # For CSV-based training
     GET /train  # For database schema-based training
     ```
   - To ask a question:
     ```bash
     POST /ask
     Content-Type: application/json
     
     {
         "question": "Your natural language question here"
     }
     ```

   Example using curl:

   For training:
   ```bash
   curl "http://localhost:5000/train?csv=path/to/your/csvfile.csv"
   ```
   or
   ```bash
   curl "http://localhost:5000/train"
   ```

   For asking a question:
   ```bash
   curl -X POST "http://localhost:5000/ask" \
        -H "Content-Type: application/json" \
        -d '{"question": "What is the total sales for the last quarter?"}'
   ```

   The response will be a JSON object containing the generated SQL query and the query results.
