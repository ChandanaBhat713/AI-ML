from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def train_model(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the XGBoost model with hyperparameter tuning
    param_dist = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Use CPU for training, if you have gpu access you can use gpu
    device = 'cpu'

    random_search = RandomizedSearchCV(XGBClassifier(random_state=42, eval_metric='logloss', tree_method='hist', device=device), 
                                       param_distributions=param_dist, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)

    # Best model from random search
    best_model = random_search.best_estimator_

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save the trained model and X_test data
    joblib.dump(best_model, 'trained_model.pkl')
    joblib.dump(X_test, 'X_test.pkl')

    return best_model

# Load preprocessed data
X, y, label_encoders, scaler, categorical_cols, numerical_cols = joblib.load('preprocessed_data.pkl')

# Train the model using CPU
print("Training model using CPU...")
train_model(X, y)
