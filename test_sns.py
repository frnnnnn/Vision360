import boto3
import os
from dotenv import load_dotenv

load_dotenv()

SNS_TOPIC_ARN = os.getenv("SNS_TOPIC_ARN")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

print(f"Region: {REGION}")
print(f"Topic ARN: {SNS_TOPIC_ARN}")

if not SNS_TOPIC_ARN:
    print("Error: SNS_TOPIC_ARN is empty")
    exit(1)

try:
    sns = boto3.client("sns", region_name=REGION)
    response = sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject="Test Vision360 Alert",
        Message="This is a test message from the Vision360 debugging script."
    )
    print("Success! Message ID:", response["MessageId"])
except Exception as e:
    print(f"Error sending SNS: {e}")
