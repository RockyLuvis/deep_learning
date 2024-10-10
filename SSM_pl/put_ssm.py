import boto3

client = boto3.client('ssm')

response = client.put_parameter(
    Name='staging_password',
    Description='Staging Password',
    Value='test123',
    Type='SecureString',
    Overwrite=True,
    AllowedPattern='string',
    Tags=[
        {
            'Key': 'env',
            'Value': 'staging'
        },
    ],
    Tier='Standard'
)