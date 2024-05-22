import vanna
import os
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore
from vanna.flask import VannaFlaskApp
import json



os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)



def train_ddl(vn):
    df_ddl = vn.run_sql("SELECT type, sql FROM sqlite_master WHERE sql is not null")

    for ddl in df_ddl['sql'].to_list():
        vn.train(ddl=ddl)


def remove_trained_data(vn):
    training_data = vn.get_training_data()
    ids = training_data['id']
    for id_ in ids:
        vn.remove_training_data(id=id_)

def remove_specific_trained_data(vn):
    training_data = vn.get_training_data()
    for index, row in training_data.iterrows():
        training_data_type = row['training_data_type']
        if training_data_type != 'sql':
            vn.remove_training_data(id=row['id'])



def train_model(vn):
    train_json_path = 'data/train_data/bird/train.json'
    f1 = open(train_json_path, 'r')
    json_list = json.load(f1)
    f1.close()

    length = len(json_list)
    for i,pairs in enumerate(json_list):
        #documentation = "database:{}; ".format(pairs['db_id'])+pairs["evidence"]
        vn.train(question=pairs["question"], sql=pairs["SQL"])
        print("start training model {}/{} data".format(i, length))
    


def train_document(vn):
    dev_json_path = 'data/test_data/dev.json'
    f1 = open(dev_json_path, 'r')
    json_list = json.load(f1)
    f1.close()
    db_name = 'california_schools'
    length = len(json_list)
    for i,pairs in enumerate(json_list):
        db_id = pairs['db_id']
        if db_id == db_name:
            evidences = pairs["evidence"].split(';')
            if len(evidences) == 0 :
                continue
            else:
                for evidence in evidences:
                    vn.train(documentation = evidence)
                    print("start training document {} data".format(i))




def run():

    vn = MyVanna(config={
                'api_key' : 'eyJhbGciOiJIUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICI3Mzc5ZDJkOC00MTBlLTRhMjMtOWQzNS1iYTE5Y2RmMGRjNzIifQ.eyJpYXQiOjE3MTYyNzMzNTQsImp0aSI6ImMyMGM0M2EyLThkM2YtNDc5NC04ZGExLTM2Y2U3MjJmYzkyZCIsImlzcyI6Imh0dHA6Ly9iai5wcml2YXRlLmdsdW9uLW1lc29uLnRlY2g6ODAwNi9yZWFsbXMvZ2x1b24tbWVzb24iLCJhdWQiOiJodHRwOi8vYmoucHJpdmF0ZS5nbHVvbi1tZXNvbi50ZWNoOjgwMDYvcmVhbG1zL2dsdW9uLW1lc29uIiwic3ViIjoiNDhiOWFjM2MtODVhNy00YTUxLWI1MjMtMDU2NTUyNTBiZDg0IiwidHlwIjoiT2ZmbGluZSIsImF6cCI6ImNvbnNvbGUtdWkiLCJzZXNzaW9uX3N0YXRlIjoiZjgzYjc0ZTEtNGQ1ZS00MDU3LWE4ODUtNDI2YTYyYjZkMGJiIiwic2NvcGUiOiJwcm9maWxlIGVtYWlsIG9mZmxpbmVfYWNjZXNzIHJvbGVzIiwic2lkIjoiZjgzYjc0ZTEtNGQ1ZS00MDU3LWE4ODUtNDI2YTYyYjZkMGJiIn0.ZcH9UeEE6dJaXs8zsxdHUDbGMZQUdf5QMCbCFLlf90I',
                'base_url' : 'http://bj.private.gluon-meson.tech:11000/model-proxy/v1',
                'model' : 'gpt-4',
                'path':'data/chromadb'
                })
    vn.connect_to_sqlite('./data/test_data/california_schools.sqlite')
    remove_trained_data(vn)
    train_ddl(vn)
    train_model(vn)
    train_document(vn)
    suggested_db_name = 'california_schools'
    suggested_db_path =  'data/test_data/dev.json'
    app = VannaFlaskApp(vn, suggested_db_name=suggested_db_name, suggested_db_path=suggested_db_path)
    host = '0.0.0.0'
    port = '42210'
    app.run(host=host, port=port)
        
run() 
