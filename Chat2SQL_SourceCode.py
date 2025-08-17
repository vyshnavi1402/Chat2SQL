#!/usr/bin/env python
# coding: utf-8

# #### **Installations**

# ##### **General Installations**

# In[1]:


pip install sentence_transformers transformers


# In[2]:


pip install faiss-cpu numpy datasets huggingface_hub


# In[10]:


pip install google


# In[11]:


pip install google-cloud


# In[12]:


pip install google-api-python-client


# In[3]:


pip install google-generativeai


# ##### **Installations for DuckDBNSQLModel**

# In[3]:


pip install torch torchvision torchaudio


# #### **Dataset**

# In[15]:


from huggingface_hub import login
login(token="")


# In[16]:


from datasets import load_dataset
ds = load_dataset("motherduckdb/duckdb-text2sql-25k")
ds


# In[17]:


prompts = ds['train']['prompt']
sql = ds['train']['query']

print(prompts[11])
print(sql[11])


# #### **RAG Architecture Implementation**

# In[18]:


from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
import numpy as np

embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Sample vector store (pretend these are few-shot training examples)
examples = prompts
example_sql = sql

# Embed and build FAISS index
example_embeddings = embedder.encode(examples)
index = faiss.IndexFlatL2(example_embeddings.shape[1])
index.add(np.array(example_embeddings))

def retrieve_examples(user_query, top_k=2):
    query_vec = embedder.encode([user_query])
    D, I = index.search(np.array(query_vec), top_k)
    return [(examples[i], example_sql[i]) for i in I[0]]

# RAG Prompt Construction
def build_rag_prompt(user_query):
    retrieved = retrieve_examples(user_query)
    prompt = ""
    for q, sql in retrieved:
        prompt += f"Q: {q}\nSQL: {sql}\n"

    prompt += f"\nQ: {user_query}\nSQL:"
    return prompt


# In[19]:


# Test RAG Pipeline
user_query = "Create me a MYSQL Command to list all patients who were diagnosed with 'Hypertension' and have follow-up appointments scheduled"
prompt = build_rag_prompt(user_query)

print("RAG-Enhanced Prompt:\n", prompt)


# #### **Gemini Model Implementation**

# In[ ]:


import google.generativeai as genai

# Configure API key
genai.configure(api_key=" ")

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# Use a concise prompt and specify desired output format
prompt = """
You are a helpful assistant. Only return the SQL query. Do not explain anything.

User: Give an SQL query to find patients diagnosed with Hypertension who have a follow-up appointment scheduled. Assume a table named 'patients' with columns: patient_id, diagnosis, followup_appointment.
"""

# Generate output
response = model.generate_content(prompt)

# Print just the SQL
print(response.text.strip())


# #### **DuckDBNSQL Model**

# In[40]:


from transformers import AutoTokenizer, AutoModelForCausalLM

duckdb_model_id = "motherduckdb/DuckDB-NSQL-7B-v0.1"
duckdb_tokenizer = AutoTokenizer.from_pretrained(duckdb_model_id)
duckdb_model = AutoModelForCausalLM.from_pretrained(duckdb_model_id)


# prompt = '''Q: Generate a SQL query that pivots the ⁠ class_details ⁠ column in the ⁠ course_arrange ⁠ table by ⁠ grade ⁠ and the ⁠ name ⁠ and ⁠ hometown ⁠ columns in the ⁠ teacher ⁠ table by ⁠ age ⁠, then performs a right join on ⁠ teacher_id ⁠ and filters for ⁠ age ⁠ between 20 and 30.
# SQL: WITH pivot_course AS (
#     PIVOT course_arrange ON Grade USING SUM(course_arrange.Class_Details->'Room') AS room, SUM(course_arrange.Class_Details->'Time') AS time GROUP BY course_arrange.Course_ID, course_arrange.Teacher_ID
# ), pivot_teacher AS (
#     PIVOT teacher ON Age USING MAX(Name) AS name, AVG(Hometown) AS hometown GROUP BY teacher.Teacher_ID
# )
# SELECT pivot_course., pivot_teacher. FROM pivot_course RIGHT JOIN pivot_teacher ON pivot_course.Teacher_ID = pivot_teacher.Teacher_ID WHERE pivot_teacher.Age BETWEEN 20 AND 30;
# Q: Generate a list of all the education values from the Students table.
# SQL: SELECT UNNEST(education) as student_education FROM Students;

# Q: Create me a MYSQL Command for List all the students in the school who all were there in 6th class ,with more than B grade
# SQL:'''


# #### **Connection to Database and Query Execution**

# In[9]:


pip install mysql-connector-python tabulate


# In[72]:


import mysql.connector
from mysql.connector import Error
from tabulate import tabulate

def execute_query_on_mysql(query):
    try:
        # Step 1: Connect to the MySQL database
        mydb = mysql.connector.connect(
            host="",
            port=3306,
            user="root",
            password="",
            database=""
        )

        if mydb.is_connected():
            print('Connected to MySQL')
            cursor = mydb.cursor()
            cursor.execute(query)

            # Step 2: Print tabular result if SELECT query
            if query.strip().lower().startswith("select"):
                columns = [desc[0] for desc in cursor.description]
                results = cursor.fetchall()
                table_data = tabulate(results, headers=columns, tablefmt='psql')
                return table_data
            else:
                mydb.commit()
                print(f"Query executed successfully: {cursor.rowcount} row(s) affected.")
                return ""

    except Error as e:
        print("Error while connecting or executing query:", e)
        return ""

    finally:
        if 'mydb' in locals() and mydb.is_connected():
            print('Closing the connection')
            cursor.close()
            mydb.close()

# Example usage
gemini_generated_query = "SELECT * FROM Perscription;"
tabular_data = execute_query_on_mysql(gemini_generated_query)
print(tabular_data)


# #### **Results to Natural Language**

# In[12]:


import google.generativeai as genai

genai.configure(api_key="")
sql_to_nlp_model = genai.GenerativeModel("gemini-1.5-flash")

# table_data = """
# | patient_id | first_name | last_name | gender | appointment_date |
# |------------|------------|-----------|--------|------------------|
# | 1          | John       | Doe       | M      | 2024-06-10       |
# | 2          | Jane       | Smith     | F      | 2024-06-12       |
# """

table_data = tabular_data

def getTableEmbeddedPrompt(table_data):
  prompt = f'''You are a medical assistant analyzing hospital data.

  Below is a table extracted from a hospital management system. Your task is to:
  1. Summarize key insights, patterns, or statistics from the data.
  2. Highlight notable entries if any (e.g., patients with multiple visits, critical prescriptions, etc.).
  3. Provide an overall narrative or observation in natural language.
  4. Do not list each row individually unless it is an outlier or important.

  Table data:
  {table_data}

  Write a clear, human-readable summary with bullet points followed by a short paragraph conclusion.'''
  return prompt

prompt = getTableEmbeddedPrompt(table_data)
response = sql_to_nlp_model.generate_content(prompt)
print(response.text)


# ### **Comparative Study**

# In[60]:


# Considering this query to evaluate all the models
test_query = "I want to find patients whose latest diagnosis was about diabetes and who received insulin medication"
test_query_rag_prompt = build_rag_prompt(test_query)
print(test_query_rag_prompt)


# In[48]:


def getCleanQuery(query):
    return query.strip().removeprefix("```sql").removesuffix("```").strip()


# #### **Model 1: User Prompt -> Gemini API**

# In[50]:


# Query Generation
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
gemini_response = model.generate_content(test_query)
gemini_query = getCleanQuery(gemini_response.text)
print(gemini_query)


# In[51]:


# Execution on DB
gemini_query_response = execute_query_on_mysql(gemini_query)
print(gemini_query_response)


# In[52]:


# Data response in Natural Language
gemini_prompt = getTableEmbeddedPrompt(gemini_query_response)
gemini_nl_response = sql_to_nlp_model.generate_content(gemini_prompt)
print(gemini_nl_response.text)


# #### **Model 2: User Prompt -> RAG -> Gemini API**

# In[53]:


# Query Generation
rag_gemini_response = model.generate_content(test_query_rag_prompt)
rag_gemini_query = getCleanQuery(rag_gemini_response.text)
print(rag_gemini_query)


# In[54]:


# Execution on DB
rag_gemini_query_response = execute_query_on_mysql(rag_gemini_query)
print(rag_gemini_query_response)


# In[55]:


# Data response in Natural Language
rag_gemini_prompt = getTableEmbeddedPrompt(rag_gemini_query_response)
rag_gemini_nl_response = sql_to_nlp_model.generate_content(rag_gemini_prompt)
print(rag_gemini_nl_response.text)


# #### **Model 3: User Prompt -> DuckDBNSQL Model**

# In[ ]:


# Query Generation
ddb_inputs = duckdb_tokenizer(test_query, return_tensors="pt")
ddb_outputs = duckdb_model.generate(**ddb_inputs, max_new_tokens=50)

ddb_query = duckdb_tokenizer.decode(ddb_outputs[0], skip_special_tokens=True)


# In[65]:


print(ddb_query)


# In[66]:


# Execution on DB
ddb_query_response = execute_query_on_mysql(ddb_query)
print(ddb_query_response)


# In[67]:


# Data response in Natural Language
ddb_prompt = getTableEmbeddedPrompt(ddb_query_response)
ddb_nl_response = sql_to_nlp_model.generate_content(ddb_prompt)
print(ddb_nl_response.text)


# #### **Model 4: User Prompt -> RAG -> DuckDBNSQL Model**

# In[ ]:


# Query Generation
rag_ddb_inputs = duckdb_tokenizer(test_query_rag_prompt, return_tensors="pt")
rag_ddb_outputs = duckdb_model.generate(**rag_ddb_inputs, max_new_tokens=50)

rag_ddb_query = duckdb_tokenizer.decode(rag_ddb_outputs[0], skip_special_tokens=True)


# In[75]:


print(rag_ddb_query)


# In[76]:


# Execution on DB
rag_ddb_query_response = execute_query_on_mysql(rag_ddb_query)
print(rag_ddb_query_response)


# In[77]:


# Data response in Natural Language
rag_ddb_prompt = getTableEmbeddedPrompt(rag_ddb_query_response)
rag_ddb_nl_response = sql_to_nlp_model.generate_content(rag_ddb_prompt)
print(rag_ddb_nl_response.text)


# #### **Results Analysis**

# The evaluation of output responses across Gemini, RAG-augmented Gemini, DuckDB-NSQL, and RAG-augmented DuckDB models reveals meaningful insights into the behavior and effectiveness of each system when presented with a minimal dataset comprising only patient IDs and names.
# 
# Despite the simplicity of the dataset, RAG-based pipelines consistently demonstrated an ability to contextualize the limitations of the data and communicate the need for additional fields (e.g., diagnosis, medication, visit frequency) to enable richer analysis. Notably:
# 
# RAG + Gemini produced a structured and clear response, identifying both the lack of clinical detail and highlighting sequential ID patterns, which is a subtle but insightful observation not made by base Gemini alone.
# 
# DuckDB-NSQL, although not a language model, showed commendable performance in understanding the dataset's scope and returning appropriate observations. The model effectively interpreted the scale of the dataset, absence of outliers, and potential sample source (small clinic or department).
# 
# **Key Takeaways**
# 
# - RAG frameworks successfully enhance the baseline performance by adding context-driven interpretation, especially when datasets are sparse.
# 
# - DuckDBNSQL, even as a structured SQL-focused model, matched LLMs in basic analytical reasoning when combined with RAG, and in some cases, surpassed them with schema-specific deductions.
# 
# - The RAG + DuckDBNSQL setup appears highly promising for domain-specific structured tasks like healthcare data analytics, as it leverages retrieval and precise generation effectively.
# 
# **Conclusion**
# 
# Overall, both RAG-augmented architectures—with Gemini and DuckDB—demonstrated clear advantages in output depth, specificity, and relevance. The RAG + DuckDB pipeline, in particular, shows potential to outperform generic LLMs in structured, schema-bound data environments by blending contextual awareness with SQL fluency.
# 
# 

# In[ ]:





# In[ ]:




