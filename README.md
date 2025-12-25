# EstateVision: Multimodal Real Estate Pricing

This project is an exploratory analysis and modeling initiative for real estate pricing using a housing dataset. The goal is to predict property prices based on tabular features (bedrooms, bathrooms, area, location) and visual data (property images).

## Project Structure

```
├── data/               # Data folder (Ignored in git)
│   ├── HousesInfo.txt  # Tabular dataset
│   └── images/         # House images
├── db/                 # Database initialization scripts
├── models/             # Trained models
├── notebooks/          # Jupyter Notebooks for EDA and modeling
├── scripts/            # Utility scripts (data loading, etc.)
├── docker-compose.yml  # Docker services configuration (PostgreSQL)
└── README.md           # Project documentation
```

## Requirements

*   Python 3.8+
*   Docker & Docker Compose
*   Python Libraries (see `requirements.txt` or install manually: pandas, psycopg2, scikit-learn, matplotlib, seaborn, jupyter)

## Setup & Execution

### 1. Database (Docker)

The project uses PostgreSQL running in a Docker container to store structured housing data.

To start the database:

```bash
docker-compose up -d
```

This will spin up a Postgres service on port `5433` (user: `admin`, password: `admin123`, db: `real_estate`).

### 2. Data Setup

**Security Note:** The original data is not included in this repository for privacy/size reasons.

To reproduce the environment:
1.  Place the `HousesInfo.txt` file in the `data/` folder.
2.  Place the images in the `data/images/` folder.

### 3. Data Loading

Once the database is running and files are in place, run the ETL script to populate the database:

```bash
cd scripts
python load_data.py
```

### 4. Analysis & Modeling

Explore the notebooks in the `notebooks/` folder:
*   `01_eda_tabular.ipynb`: Exploratory Data Analysis (EDA) of the tabular data.
*   `03_baseline.ipynb`: Baseline price prediction model (XGBoost).

## Project Status

🚧 **Under Development** 🚧

*   [x] Database Configuration (Docker)
*   [x] ETL Script (Data Loading)
*   [x] Initial EDA
*   [ ] Advanced Feature Engineering
*   [ ] Deep Learning Modeling (CNNs for images)
*   [ ] Inference API

## Author

Daniel Esteban Yaruro Contreras
