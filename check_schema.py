import boto3
import os
from dotenv import load_dotenv

load_dotenv()

REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
DDB_TABLE = os.getenv("DDB_TABLE", "deteccion_eventos")

session = boto3.session.Session(region_name=REGION)
ddb = session.client("dynamodb")

try:
    resp = ddb.describe_table(TableName=DDB_TABLE)
    print("KeySchema:", resp["Table"]["KeySchema"])
    print("AttributeDefinitions:", resp["Table"]["AttributeDefinitions"])
except Exception as e:
    print(e)
