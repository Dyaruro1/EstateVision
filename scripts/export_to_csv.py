"""
Export property data from PostgreSQL to CSV for Kaggle upload.
"""

import psycopg2
import pandas as pd
from pathlib import Path

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "real_estate",
    "user": "admin",
    "password": "admin123"
}

# Query to fetch all properties
query = """
SELECT 
    p.id,
    p.bedrooms,
    p.bathrooms,
    p.area,
    p.zipcode,
    p.price
FROM propiedades p
ORDER BY p.id;
"""

# Connect to database
conn = psycopg2.connect(**DB_CONFIG)
df = pd.read_sql_query(query, conn)
conn.close()

# Save to CSV
output_path = Path("../data/propiedades.csv")
df.to_csv(output_path, index=False)

print(f"✓ Exported {len(df)} properties to {output_path}")
print(f"\nFirst few rows:")
print(df.head())
