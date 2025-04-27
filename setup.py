from setuptools import setup, find_packages

setup(
    name='pytorch_db_checkpoint',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch>=2.6.0',
        'psycopg2>=2.9.10'
    ]
)