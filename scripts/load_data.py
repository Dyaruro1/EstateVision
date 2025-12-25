import psycopg2
from pathlib import Path

# Paths
DATA_FILE = Path("../data/HousesInfo.txt")
IMAGES_DIR = Path("../data/images")

# DB config
DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "real_estate",
    "user": "admin",
    "password": "admin123"
}

conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

with open(DATA_FILE, "r") as f:
    for idx, line in enumerate(f, start=1):
        bedrooms, bathrooms, area, zipcode, price = line.split()

        cursor.execute(
            """
            INSERT INTO propiedades (bedrooms, bathrooms, area, zipcode, price)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id;
            """,
            (int(bedrooms), float(bathrooms), int(area), int(zipcode), int(price))
        )

        propiedad_id = cursor.fetchone()[0]

        for tipo in ["kitchen", "bathroom", "bedroom", "frontal"]:
            image_path = IMAGES_DIR / f"{idx}_{tipo}.jpg"

            cursor.execute(
                """
                INSERT INTO metadata_imagenes (propiedad_id, tipo_imagen, ruta_imagen)
                VALUES (%s, %s, %s);
                """,
                (propiedad_id, tipo, str(image_path))
            )

conn.commit()
cursor.close()
conn.close()

print("Datos cargados correctamente")