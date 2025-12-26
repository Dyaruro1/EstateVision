"""
Data loader for Kaggle environment - loads from CSV instead of PostgreSQL.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import StandardScaler
import pickle


class RealEstateDataLoaderKaggle:
    """Loads real estate data from CSV and images for Kaggle environment."""
    
    def __init__(
        self,
        csv_path: Path,
        images_dir: Path,
        img_size: Tuple[int, int] = (224, 224),
        normalize_images: bool = True
    ):
        """
        Initialize Kaggle data loader.
        
        Args:
            csv_path: Path to CSV file with property data
            images_dir: Path to images directory
            img_size: Target size for image resizing
            normalize_images: Whether to normalize images to [0, 1]
        """
        self.csv_path = Path(csv_path)
        self.images_dir = Path(images_dir)
        self.img_size = img_size
        self.normalize_images = normalize_images
        self.scaler = StandardScaler()
        self.image_types = ["kitchen", "bathroom", "bedroom", "frontal"]
        
    def load_data_from_csv(self) -> pd.DataFrame:
        """Load property data from CSV file."""
        df = pd.read_csv(self.csv_path)
        return df
    
    def _load_single_image(self, image_path: Path) -> np.ndarray:
        """Load and preprocess a single image."""
        try:
            img = load_img(image_path, target_size=self.img_size)
            img_array = img_to_array(img)
            
            if self.normalize_images:
                img_array = img_array / 255.0
                
            return img_array
        except Exception as e:
            print(f"Warning: Could not load {image_path}")
            return np.zeros((*self.img_size, 3), dtype=np.float32)
    
    def _get_image_paths(self, property_id: int) -> Dict[str, Path]:
        """Get file paths for all images of a property."""
        image_paths = {}
        for img_type in self.image_types:
            filename = f"{property_id}_{img_type}.jpg"
            image_paths[img_type] = self.images_dir / filename
        return image_paths
    
    def load_property_images(self, property_id: int) -> np.ndarray:
        """Load all 4 images for a property."""
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
        """Prepare tabular features and labels (with separate zipcode for embedding)."""
        
        # Feature engineering: room_ratio (consistent with baseline)
        df['room_ratio'] = df['bedrooms'] / df['bathrooms']
        
        # Separate numeric features from categorical
        numeric_columns = ['bedrooms', 'bathrooms', 'area', 'room_ratio']
        X_numeric = df[numeric_columns].values
        
        # Zipcode as categorical for embedding
        zipcode = df['zipcode'].values
        n_zipcodes = len(np.unique(zipcode))
        
        # Target with log transform
        y = df['price'].values
        y_log = np.log1p(y)
        
        # Normalize numeric features only
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X_numeric)
        else:
            X_scaled = self.scaler.transform(X_numeric)
        
        return X_scaled, zipcode, y_log, n_zipcodes
    
    def create_dataset(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        validation_split: float = 0.2,
        random_seed: int = 42
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict[str, int]]:
        """Create TensorFlow datasets for training and validation."""
        
        # Load data from CSV
        df = self.load_data_from_csv()
        
        # Prepare tabular features (separate zipcode for embedding)
        X_tabular, zipcode, y, n_zipcodes = self.prepare_tabular_features(df, fit_scaler=True)
        
        # Get property IDs
        property_ids = df['id'].values
        n_samples = len(property_ids)
        
        # Load all images
        print(f"Loading {n_samples} properties with 4 images each...")
        X_images = []
        for prop_id in property_ids:
            images = self.load_property_images(prop_id)
            X_images.append(images)
        X_images = np.array(X_images)
        
        # Split data
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
        
        print(f" Dataset created: {dataset_info['n_train']} train, {dataset_info['n_val']} val samples")
        
        return train_dataset, val_dataset, dataset_info
    
    def save_scaler(self, filepath: Path):
        """Save fitted scaler."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_scaler(self, filepath: Path):
        """Load previously fitted scaler."""
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
