CREATE TABLE propiedades (
    id SERIAL PRIMARY KEY,
    bedrooms INTEGER,
    bathrooms FLOAT,
    area INTEGER,
    zipcode INTEGER,
    price BIGINT
);

CREATE TABLE metadata_imagenes (
    id SERIAL PRIMARY KEY,
    propiedad_id INTEGER REFERENCES propiedades(id),
    tipo_imagen VARCHAR(50),
    ruta_imagen TEXT
);

SELECT * FROM metadata_imagenes;