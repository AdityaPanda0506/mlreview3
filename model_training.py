import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

class AttritionModel:
    def __init__(self):
        self.models = {}
        self.label_encoders = {}
        self.feature_names = [
            'OverTime', 'MonthlyIncome', 'StockOptionLevel', 
            'JobSatisfaction', 'WorkLifeBalance', 'YearsAtCompany'
        ]
    
    def generate_synthetic_data(self, n_samples=1500):
        """Generate realistic synthetic HR data"""
        np.random.seed(42)
        
        data = {
            'OverTime': np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65]),
            'MonthlyIncome': np.random.normal(65000, 20000, n_samples).clip(20000, 150000),
            'StockOptionLevel': np.random.choice([0, 1, 2, 3], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
            'JobSatisfaction': np.random.choice([1, 2, 3, 4], n_samples, p=[0.1, 0.2, 0.4, 0.3]),
            'WorkLifeBalance': np.random.choice([1, 2, 3, 4], n_samples, p=[0.15, 0.25, 0.4, 0.2]),
            'YearsAtCompany': np.random.exponential(4, n_samples).clip(0, 25),
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic target variable with non-linear relationships
        attrition_proba = (
            (df['OverTime'] == 'Yes') * 0.25 +
            (df['MonthlyIncome'] < 45000) * 0.30 +
            (df['JobSatisfaction'] == 1) * 0.35 +
            (df['JobSatisfaction'] == 2) * 0.15 +
            (df['WorkLifeBalance'] == 1) * 0.30 +
            (df['WorkLifeBalance'] == 2) * 0.12 +
            (df['StockOptionLevel'] == 0) * 0.10 +
            (df['YearsAtCompany'] < 1) * 0.15 +
            ((df['YearsAtCompany'] > 5) & (df['JobSatisfaction'] < 3)) * 0.20 +  # Stagnation risk
            np.random.normal(0, 0.08, n_samples)
        )
        
        df['Attrition'] = (attrition_proba > 0.4).astype(int)
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for training"""
        df_processed = df.copy()
        
        # Encode categorical variables
        if 'OverTime' in df_processed.columns:
            le = LabelEncoder()
            df_processed['OverTime'] = le.fit_transform(df_processed['OverTime'])
            self.label_encoders['OverTime'] = le
        
        return df_processed
    
    def train(self):
        """Train the ensemble model"""
        print("Generating synthetic data...")
        df = self.generate_synthetic_data(1500)
        
        print("Preprocessing data...")
        df_processed = self.preprocess_data(df)
        
        # Select features
        X = df_processed[self.feature_names]
        y = df_processed['Attrition']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("Training ensemble model...")
        
        # Train multiple models for ensemble
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.models['random_forest'].fit(X_train, y_train)
        
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.models['xgboost'].fit(X_train, y_train)
        
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        self.models['lightgbm'].fit(X_train, y_train)
        
        # Evaluate model
        accuracy = self.evaluate(X_test, y_test)
        print(f"✅ Model training completed with {accuracy:.1%} accuracy")
        
        return self
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            pred_proba = model.predict_proba(X_test)[:, 1]
            predictions[name] = pred_proba
        
        # Ensemble prediction (average)
        ensemble_proba = np.mean(list(predictions.values()), axis=0)
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, ensemble_pred)
        
        print(f"\n📊 MODEL PERFORMANCE SUMMARY")
        print(f"Ensemble Accuracy: {accuracy:.4f}")
        print(f"Classification Report:")
        print(classification_report(y_test, ensemble_pred))
        
        return accuracy
    
    def predict_proba(self, input_data):
        """Predict attrition probability"""
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Preprocess categorical variables
        for col in ['OverTime']:
            if col in input_df.columns and col in self.label_encoders:
                try:
                    input_df[col] = self.label_encoders[col].transform([input_data[col]])[0]
                except ValueError:
                    input_df[col] = 0
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        input_df = input_df[self.feature_names]
        
        # Get predictions from all models
        predictions = []
        for name, model in self.models.items():
            pred_proba = model.predict_proba(input_df)[0, 1]
            predictions.append(pred_proba)
        
        # Return ensemble probability
        return np.mean(predictions)