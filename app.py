import os
import logging
import sys
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from openai import OpenAI
from dotenv import load_dotenv
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    SQLDatabase,
)
from llama_index.indices.struct_store.sql_query import (
    SQLTableRetrieverQueryEngine,
)
from llama_index.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from sqlalchemy import select, create_engine, MetaData, Table
from llama_index.indices.struct_store.sql_query import NLSQLTableQueryEngine

# load .env file
load_dotenv()

# Logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# DB setup and indexing
db_uri = os.getenv('DB_URI')
db_engine = create_engine(db_uri)
sql_database = SQLDatabase(db_engine)

table_schema_objs = []
for tn in sql_database.get_usable_table_names():
    table_schema_objs.append(SQLTableSchema(table_name = tn))

obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    SQLTableNodeMapping(sql_database),
    VectorStoreIndex,
)

query_engine = SQLTableRetrieverQueryEngine(
    sql_database, obj_index.as_retriever(similarity_top_k=1)
)

# Create web service and api
client = OpenAI()
client.models.list()

app = Flask(__name__)
api = Api(app)

# Endpoint handler
class ChatGpt(Resource):
    def get(self):
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
                    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            return e.message
        
    def post(self):
        try:
            query_str = request.json['query']
            response = query_engine.query(query_str)
            return response.response
        except Exception as e:
            return e.message

api.add_resource(ChatGpt, '/')