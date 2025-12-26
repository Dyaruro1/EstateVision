"""
Data loader for real estate dataset combining tabular data and images.
Creates TensorFlow-ready dataset for multi-modal neural networks.
"""

import os
import numpy as np
import pandas as pd
import psycopg2
from pathlib import Path
from typing import Tuple, Dict, List
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import StandardScaler
import pickle


class RealEstateDataLoader:
    """Loads and preprocesses real estate data combining tabular features and images."""
    
    def __init__(
        self,
        db_config: Dict[str, str],
        images_dir: Path,
        img_size: Tuple[int, int] = (224, 224),
        normalize_images: bool = True
    ):
        """
        Initialize data loader.
        
        Args:
            db_config: Database connection configuration
            images_dir: Path to images directory
            img_size: Target size for image resizing
            normalize_images: Whether to normalize images to [0, 1]
        """
        self.db_config = db_config
        self.images_dir = Path(images_dir)
        self.img_size = img_size
        self.normalize_images = normalize_images
        self.scaler = StandardScaler()
        self.image_types = ["kitchen", "bathroom", "bedroom", "frontal"]
        
    def _connect_db(self) -> psycopg2.extensions.connection:
        """Establish database connection."""
        return psycopg2.connect(**self.db_config)
    
    def load_data_from_db(self) -> pd.DataFrame:
        """
        Load tabular data from PostgreSQL database.
        
        Returns:
            DataFrame with property features and image metadata
        """
        query = """
        SELECT 
            p.id,
            p.bedrooms,
            p.bathrooms,
            p.area,
            p.zipcode,
            p.price,
            m.tipo_imagen,
            m.ruta_imagen
        FROM propiedades p
        LEFT JOIN metadata_imagenes m ON p.id = m.propiedad_id
        ORDER BY p.id, m.tipo_imagen;
        """
        
        conn = self._connect_db()
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def _load_single_image(self, image_path: Path) -> np.ndarray:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        try:
            img = load_img(image_path, target_size=self.img_size)
            img_array = img_to_array(img)
            
            if self.normalize_images:
                img_array = img_array / 255.0
                
            return img_array
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return blank image on error
            return np.zeros((*self.img_size, 3), dtype=np.float32)
    
    def _get_image_paths(self, property_id: int) -> Dict[str, Path]:
        """
        Get file paths for all images of a property.
        
        Args:
            property_id: ID of the property
            
        Returns:
            Dictionary mapping image type to file path
        """
        image_paths = {}
        for img_type in self.image_types:
            filename = f"{property_id}_{img_type}.jpg"
            image_paths[img_type] = self.images_dir / filename
        return image_paths
    
    def load_property_images(self, property_id: int) -> np.ndarray:
        """
        Load all 4 images for a property.
        
        Args:
            property_id: ID of the property
            
        Returns:
            Array of shape (4, height, width, 3) containing all images
        """
        image_paths = self._get_image_paths(property_id)
        images = []
        
        for img_type in self.image_types:
            img_array = self._load_single_image(image_paths[img_type])
            images.append(img_array)
        
        return np.array(images)
    
    def prepare_tabular_features(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Prepare tabular features and labels (with separate zipcode for embedding).
        
        Args:
            df: DataFrame with property data
            fit_scaler: Whether to fit the scaler (True for train, False for test)
            
        Returns:
            Tuple of (numeric_features, zipcode, labels, n_zipcodes)
        """
        # Get unique properties
        properties_df = df.groupby('id').first().reset_index()
        
        # Feature engineering: room_ratio (consistent with baseline)
        properties_df['room_ratio'] = properties_df['bedrooms'] / properties_df['bathrooms']
        
        # Separate numeric features from categorical
        numeric_columns = ['bedrooms', 'bathrooms', 'area', 'room_ratio']
        X_numeric = properties_df[numeric_columns].values
        
        # Zipcode as categorical for embedding
        zipcode = properties_df['zipcode'].values
        n_zipcodes = len(np.unique(zipcode))
        
        # Target with log transform
        y = properties_df['price'].values
        y_log = np.log1p(y)
        
        # Normalize numeric features only
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X_numeric)
        else:
            X_scaled = self.scaler.transform(X_numeric)
        
        return X_scaled, zipcode, y_log, n_zipcodes
    
    def create_dataset(
        self,
        batch_size: int = 16,
        shuffle: bool = True,
        validation_split: float = 0.2,
        random_seed: int = 42
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict[str, int]]:
        """
        Create TensorFlow datasets for training and validation.
        
        Args:
            batch_size: Batch size for dataset
            shuffle: Whether to shuffle the data
            validation_split: Fraction of data for validation
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, val_dataset, dataset_info)
        """
        # Load data from database
        df = self.load_data_from_db()
        
        # Prepare tabular features (separate zipcode for embedding)
        X_tabular, zipcode, y, n_zipcodes = self.prepare_tabular_features(df, fit_scaler=True)
        
        # Get unique property IDs
        property_ids = df['id'].unique()
        n_samples = len(property_ids)
        
        # Load all images
        print(f"Loading {n_samples} properties with 4 images each...")
        X_images = []
        for prop_id in property_ids:
            images = self.load_property_images(prop_id)
            X_images.append(images)
        X_images = np.array(X_images)
        
        # Split into train and validation
        if shuffle:
            np.random.seed(random_seed)
            indices = np.random.permutation(n_samples)
        else:
            indices = np.arange(n_samples)
        
        split_idx = int(n_samples * (1 - validation_split))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        # Create train dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'tabular_input': X_tabular[train_indices],
                'zipcode_input': zipcode[train_indices],
                'image_kitchen': X_images[train_indices, 0],
                'image_bathroom': X_images[train_indices, 1],
                'image_bedroom': X_images[train_indices, 2],
                'image_frontal': X_images[train_indices, 3]
            },
            y[train_indices]
        ))
        
        # Create validation dataset
        val_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'tabular_input': X_tabular[val_indices],
                'zipcode_input': zipcode[val_indices],
                'image_kitchen': X_images[val_indices, 0],
                'image_bathroom': X_images[val_indices, 1],
                'image_bedroom': X_images[val_indices, 2],
                'image_frontal': X_images[val_indices, 3]
            },
            y[val_indices]
        ))
        
        # Configure datasets
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=len(train_indices))
        
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        dataset_info = {
            'n_samples': n_samples,
            'n_train': len(train_indices),
            'n_val': len(val_indices),
            'n_features': X_tabular.shape[1],
            'n_zipcodes': n_zipcodes,
            'image_shape': (*self.img_size, 3),
            'note': 'Target (price) is log-transformed. Use np.expm1() to get original scale.'
        }
        
        print(f"Dataset created: {dataset_info['n_train']} train, {dataset_info['n_val']} val samples")
        
        return train_dataset, val_dataset, dataset_info
    
    def save_scaler(self, filepath: Path):
        """Save fitted scaler for later use."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_scaler(self, filepath: Path):
        """Load previously fitted scaler."""
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)


def main():
    """Example usage of the data loader."""
    
    # Configuration
    DB_CONFIG = {
        "host": "localhost",
        "port": 5433,
        "dbname": "real_estate",
        "user": "admin",
        "password": "admin123"
    }
    
    IMAGES_DIR = Path("data/images")
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 16
    
    # Initialize data loader
    loader = RealEstateDataLoader(
        db_config=DB_CONFIG,
        images_dir=IMAGES_DIR,
        img_size=IMG_SIZE,
        normalize_images=True
    )
    
    # Create datasets
    train_ds, val_ds, info = loader.create_dataset(
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_split=0.2,
        random_seed=42
    )
    
    # Save scaler
    loader.save_scaler(Path("../models/scaler.pkl"))
    
    # Display dataset info
    print("\n Dataset Information ")
    print(f"Total samples: {info['n_samples']}")
    print(f"Training samples: {info['n_train']}")
    print(f"Validation samples: {info['n_val']}")
    print(f"Tabular features: {info['n_features']}")
    print(f"Image shape: {info['image_shape']}")
    
    # Test batch loading
    print("\n Testing batch loading ")
    for batch_data, batch_labels in train_ds.take(1):
        print(f"Tabular input shape: {batch_data['tabular_input'].shape}")
        print(f"Kitchen image shape: {batch_data['image_kitchen'].shape}")
        print(f"Bathroom image shape: {batch_data['image_bathroom'].shape}")
        print(f"Bedroom image shape: {batch_data['image_bedroom'].shape}")
        print(f"Frontal image shape: {batch_data['image_frontal'].shape}")
        print(f"Labels shape: {batch_labels.shape}")
        print(f"\nSample label values: {batch_labels[:3].numpy()}")


if __name__ == "__main__":
    main()

