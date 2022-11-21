import logging
import boto3
from botocore.exceptions import ClientError
import os

def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        print(f'Uploading {file_name} to s3://{bucket}/{object_name} in PROGRESS', end='\r')
        
        response = s3_client.upload_file(file_name, bucket, object_name)
        print(f'Uploading {file_name} to s3://{bucket}/{object_name} COMPLETED. ')
    except ClientError as e:
        logging.error(e)

    return

def download_file(bucket, object_name, fname):
    s3 = boto3.client('s3')
    s3.download_file(bucket, object_name, fname)