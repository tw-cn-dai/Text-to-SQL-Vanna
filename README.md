# text2sql-vanna
Text to SQL generation via Vanna.   

This repo works in two easy steps - train a RAG "model" on your data, and then ask questions which will return SQL queries that can be set up to automatically run on your database.

Please refer to  https://github.com/vanna-ai/vanna  for information related to Vanna.


# **Getting started**

# **Install**
    pip install -r requirements.txt
    cd src/vanna 
    pip install -e .

# **Import**

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)
        
vn = MyVanna(config={
                'base_url':"http://bj.private.gluon-meson.tech:11000/model-proxy/v1", 
                'api_key': 'sk-...', 
                'model': 'gpt-4-...',
                'path': dir_to_save_vector_db
                        }
                )
# **Training**
## **Train with DDL Statements**
DDL statements contain information about the table names, columns, data types, and relationships in your database.

vn.train(ddl="""
        CREATE TABLE IF NOT EXISTS my-table (
            id INT PRIMARY KEY,
            name VARCHAR(100),
            age INT
            )
     """)
## **Train with Documentation**
Sometimes you may want to add documentation about your business terminology or definitions.

vn.train(documentation="Our business defines XYZ as ...")

## **Train with SQL**

vn.train(sql="SELECT name, age FROM my-table WHERE name = 'John Doe'")

## **Train with SQL-Question pairs**
vn.train(
        question="Name movie titles released in year 1945. Sort the listing by the descending order of movie popularity",
         sql="SELECT movie_title FROM movies WHERE movie_release_year = 1945 ORDER BY movie_popularity DESC LIMIT 1"
         )
## **Train plan**
The information schema query may need some tweaking depending on your database. This is a good starting point.

df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")

This will break up the information schema into bite-sized chunks that can be referenced by the LLM
plan = vn.get_training_plan_generic(df_information_schema)

If you like the plan, then uncomment this and run it to train
vn.train(plan=plan)


# **Demo**

Please refer to infer.py, which trains RAG based on the train set of the bird dataset, with california_schools from the dev set used as the test set, and chromaDB+GPT4 as the backend.
After running the infer.py code, a web application will be created, allowing for SQL generation tasks to be performed on the web interface.

# **DataBase**

## **Spider**
Please refer to  [Spider](https://yale-lily.github.io/spider) for information related to Spyider database.

## **Bird**
Please refer to  [Bird](https://bird-bench.github.io)  for information related to Bird database.
