import os
from dotenv import load_dotenv

os.environ['RUN_TIMEZONE_CHECK'] = '0'

from .db import init_db

load_dotenv()

print(f"POSTGRES_HOST: {os.getenv('POSTGRES_HOST')}")

if __name__ == "__main__":
    print("Initializing database...")
    init_db()