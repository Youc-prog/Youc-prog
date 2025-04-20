## Work objectives:
1. Prepare hotel booking data for classification tasks
2. Implement and compare multiple classification algorithms
3. Handle class imbalance in tourism datasets
4. Evaluate model performance with appropriate metrics
5. Interpret models for business insights


   # Import necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# For modeling
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# For evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score

# For model interpretation
import shap

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Set random seed for reproducibility
np.random.seed(42)



# Load the dataset
# Note: Assumes hotel_bookings.csv is in the current directory or path
# Dataset source: https://www.kaggle.com/jessemostipak/hotel-booking-demand
df = pd.read_csv('hotel_bookings.csv')

# Display basic information
print("Dataset shape:", df.shape)
print("\nFirst 5 rows of the dataset:")
display(df.head())



# Check for missing values
print("\nMissing values per column:")
display(df.isnull().sum().sort_values(ascending=False).head())



# Basic statistics
print("\nBasic statistics for numerical columns:")
display(df.describe().T)



## 2. Data Visualization

# Check class distribution (cancelled vs. not cancelled)
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='is_canceled', data=df)
plt.title('Distribution of Booking Cancellations')
plt.xlabel('Booking Cancelled (1 = Yes, 0 = No)')
plt.ylabel('Count')


# Add percentage labels
total = len(df)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')

plt.show()



# Cancellation rate by hotel type
plt.figure(figsize=(10, 6))
hotel_cancel = df.groupby('hotel')['is_canceled'].mean() * 100
hotel_cancel.plot(kind='bar', color=['skyblue', 'lightgreen'])
plt.title('Cancellation Rate by Hotel Type')
plt.xlabel('Hotel Type')
plt.ylabel('Cancellation Rate (%)')
plt.xticks(rotation=0)

for i, v in enumerate(hotel_cancel):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center')

plt.show()




# Cancellation rate by lead time (binned)
df['lead_time_bins'] = pd.cut(df['lead_time'], 
                              bins=[0, 7, 30, 90, 180, 365, df['lead_time'].max()],
                              labels=['0-7 days', '8-30 days', '31-90 days', 
                                     '91-180 days', '181-365 days', '365+ days'])

plt.figure(figsize=(12, 6))
lead_cancel = df.groupby('lead_time_bins')['is_canceled'].mean() * 100
lead_cancel.plot(kind='bar', color=sns.color_palette("viridis", 6))
plt.title('Cancellation Rate by Lead Time')
plt.xlabel('Lead Time')
plt.ylabel('Cancellation Rate (%)')
plt.xticks(rotation=45)

for i, v in enumerate(lead_cancel):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center')

plt.tight_layout()
plt.show()





# Cancellation rate by deposit type
plt.figure(figsize=(10, 6))
deposit_cancel = df.groupby('deposit_type')['is_canceled'].mean() * 100
deposit_cancel.plot(kind='bar', color=sns.color_palette("viridis", 3))
plt.title('Cancellation Rate by Deposit Type')
plt.xlabel('Deposit Type')
plt.ylabel('Cancellation Rate (%)')
plt.xticks(rotation=0)

for i, v in enumerate(deposit_cancel):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center')

plt.show()





# Correlation analysis for numerical features
numerical_features = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights',
                     'adults', 'children', 'babies', 'previous_cancellations',
                     'previous_bookings_not_canceled', 'booking_changes',
                     'days_in_waiting_list', 'adr', 'required_car_parking_spaces',
                     'total_of_special_requests', 'is_canceled']

plt.figure(figsize=(14, 10))
correlation_matrix = df[numerical_features].corr()
mask = np.triu(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            mask=mask, linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.show()




## 3. Feature Engineering

# Create a copy to avoid modifying the original dataframe
df_processed = df.copy()

# Create total nights feature
df_processed['total_nights'] = df_processed['stays_in_weekend_nights'] + df_processed['stays_in_week_nights']

# Create total guests feature
df_processed['total_guests'] = df_processed['adults'] + df_processed['children'] + df_processed['babies']

# Create average daily rate per person
df_processed['adr_per_person'] = df_processed['adr'] / df_processed['total_guests']
df_processed['adr_per_person'].replace([np.inf, -np.inf], np.nan, inplace=True)
df_processed['adr_per_person'].fillna(df_processed['adr'], inplace=True)

# Extract arrival month, day of week and season
df_processed['arrival_date_month'] = pd.Categorical(df_processed['arrival_date_month'],
                                                  categories=['January', 'February', 'March', 'April', 'May', 'June','July', 'August', 'September', 'October', 'November', 'December'],
                                                  ordered=True)
df_processed['arrival_month'] = df_processed['arrival_date_month'].cat.codes + 1

# Define seasons
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df_processed['season'] = df_processed['arrival_month'].apply(get_season)

# Create binary features
df_processed['is_repeated_guest'] = df_processed['is_repeated_guest'].astype(int)
df_processed['has_booking_changes'] = (df_processed['booking_changes'] > 0).astype(int)
df_processed['has_special_requests'] = (df_processed['total_of_special_requests'] > 0).astype(int)

# Handle categorical variables that need to be simplified
df_processed['market_segment'] = df_processed['market_segment'].replace('TA', 'Travel Agent')
df_processed['market_segment'] = df_processed['market_segment'].replace('TO', 'Tour Operator')

# Create a flag for high cancellation risk months (if any found from EDA)
monthly_cancellation = df_processed.groupby('arrival_month')['is_canceled'].mean()
high_cancel_months = monthly_cancellation[monthly_cancellation > monthly_cancellation.mean()].index.tolist()
df_processed['high_cancel_month'] = df_processed['arrival_month'].isin(high_cancel_months).astype(int)

df_processed


# Drop columns we don't need
columns_to_drop = ['reservation_status', 'reservation_status_date', 'arrival_date_year',
                   'arrival_date_week_number', 'arrival_date_day_of_month', 'agent', 'company',
                   'lead_time_bins']
df_processed = df_processed.drop(columns_to_drop, axis=1, errors='ignore')



# Handle missing values
df_processed['children'].fillna(0, inplace=True)
df_processed['country'].fillna('Unknown', inplace=True)

# Keep only essential columns for modeling
essential_columns = ['hotel', 'is_canceled', 'lead_time', 'arrival_month', 'season',
                    'stays_in_weekend_nights', 'stays_in_week_nights', 'total_nights',
                    'adults', 'children', 'babies', 'total_guests', 'meal',
                    'market_segment', 'distribution_channel', 'is_repeated_guest',
                    'previous_cancellations', 'previous_bookings_not_canceled',
                    'reserved_room_type', 'assigned_room_type', 'booking_changes',
                    'deposit_type', 'days_in_waiting_list', 'customer_type',
                    'adr', 'adr_per_person', 'required_car_parking_spaces',
                    'total_of_special_requests', 'has_booking_changes', 'has_special_requests',
                    'high_cancel_month']

df_model = df_processed[essential_columns]

# Display processed data
print("Processed dataframe shape:", df_model.shape)
df_model.head()


## 4. Data Preparation for Modeling

# Define features and target
X = df_model.drop('is_canceled', axis=1)
y = df_model['is_canceled']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("\nCategorical columns:", categorical_cols)
print("\nNumerical columns:", numerical_cols)

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ])



  ##5. Model Building: Logistic Regression

  # Create a Logistic Regression pipeline
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Train the model
lr_pipeline.fit(X_train, y_train)


# Predict on test set
y_pred_lr = lr_pipeline.predict(X_test)
y_pred_proba_lr = lr_pipeline.predict_proba(X_test)[:, 1]

# Evaluate performance
print("\n--- Logistic Regression Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_lr):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_lr):.4f}")



# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Logistic Regression')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()



# Plot ROC curve
plt.figure(figsize=(10, 8))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_lr)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_pred_proba_lr):.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.show()



## 6. Model Building: Decision Tree

# Create a Decision Tree pipeline
dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Train the model
dt_pipeline.fit(X_train, y_train)



# Predict on test set
y_pred_dt = dt_pipeline.predict(X_test)
y_pred_proba_dt = dt_pipeline.predict_proba(X_test)[:, 1]

# Evaluate performance
print("\n--- Decision Tree Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_dt):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_dt):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_dt):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_dt):.4f}")




# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Decision Tree')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()



# Plot ROC curve
plt.figure(figsize=(10, 8))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_dt)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_pred_proba_dt):.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree')
plt.legend(loc='lower right')
plt.show()


# Create a Random Forest pipeline
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
rf_pipeline.fit(X_train, y_train)


# Predict on test set
y_pred_rf = rf_pipeline.predict(X_test)
y_pred_proba_rf = rf_pipeline.predict_proba(X_test)[:, 1]

# Evaluate performance
print("\n--- Random Forest Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_rf):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")



## 8. Model Building: Gradient Boosting

# Create a Gradient Boosting pipeline
gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
])

# Train the model
gb_pipeline.fit(X_train, y_train)




# Predict on test set
y_pred_gb = gb_pipeline.predict(X_test)
y_pred_proba_gb = gb_pipeline.predict_proba(X_test)[:, 1]

# Evaluate performance
print("\n--- Gradient Boosting Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_gb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_gb):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_gb):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_gb):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_gb):.4f}")



## 9. Model Building: XGBoost

# Create an XGBoost pipeline
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=100, random_state=42))
])

# Train the model
xgb_pipeline.fit(X_train, y_train)



# Predict on test set
y_pred_xgb = xgb_pipeline.predict(X_test)
y_pred_proba_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]

# Evaluate performance
print("\n--- XGBoost Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_xgb):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_xgb):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_xgb):.4f}")



## 10. Model Comparison

# Compare all models
models = {
    'Logistic Regression': (y_pred_lr, y_pred_proba_lr),
    'Decision Tree': (y_pred_dt, y_pred_proba_dt),
    'Random Forest': (y_pred_rf, y_pred_proba_rf),
    'Gradient Boosting': (y_pred_gb, y_pred_proba_gb),
    'XGBoost': (y_pred_xgb, y_pred_proba_xgb)
}

# Create a comparison dataframe
comparison = pd.DataFrame(index=models.keys(), 
                         columns=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'])

for model_name, (y_pred, y_pred_proba) in models.items():
    comparison.loc[model_name, 'Accuracy'] = accuracy_score(y_test, y_pred)
    comparison.loc[model_name, 'Precision'] = precision_score(y_test, y_pred)
    comparison.loc[model_name, 'Recall'] = recall_score(y_test, y_pred)
    comparison.loc[model_name, 'F1 Score'] = f1_score(y_test, y_pred)
    comparison.loc[model_name, 'ROC AUC'] = roc_auc_score(y_test, y_pred_proba)

# Sort by F1 Score
comparison = comparison.sort_values('F1 Score', ascending=False)

# Display comparison
print("\n--- Model Comparison ---")
print(comparison)


# convert comparison dataframe to float format
comparison = comparison.astype(float)

# Visualize comparison
plt.figure(figsize=(12, 8))
sns.heatmap(comparison, annot=True, cmap='viridis', fmt='.4f')
plt.title('Model Comparison')
plt.tight_layout()
plt.show()


# Plot ROC curves for all models
plt.figure(figsize=(12, 8))
for model_name, (y_pred, y_pred_proba) in models.items():
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - All Models')
plt.legend(loc='lower right')
plt.show()



## 11. Handling Class Imbalance with SMOTE

from imblearn.over_sampling import SMOTE

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)

# Get preprocessed data
X_train_preprocessed = preprocessor.fit_transform(X_train)

# Apply SMOTE
X_train_smote, y_train_smote = smote.fit_resample(X_train_preprocessed, y_train)

# Display class distribution after SMOTE
print("\nClass distribution after SMOTE:")
print(pd.Series(y_train_smote).value_counts(normalize=True) * 100)



# Train XGBoost on SMOTE-resampled data
xgb_smote = XGBClassifier(n_estimators=100, random_state=42)
xgb_smote.fit(X_train_smote, y_train_smote)

# Predict on test set
X_test_preprocessed = preprocessor.transform(X_test)
y_pred_xgb_smote = xgb_smote.predict(X_test_preprocessed)
y_pred_proba_xgb_smote = xgb_smote.predict_proba(X_test_preprocessed)[:, 1]

# Evaluate performance
print("\n--- XGBoost with SMOTE Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb_smote):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb_smote):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_xgb_smote):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_xgb_smote):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_xgb_smote):.4f}")



# Compare regular XGBoost vs XGBoost with SMOTE
comparison_smote = pd.DataFrame({
    'XGBoost': [accuracy_score(y_test, y_pred_xgb), 
               precision_score(y_test, y_pred_xgb),
               recall_score(y_test, y_pred_xgb),
               f1_score(y_test, y_pred_xgb),
               roc_auc_score(y_test, y_pred_proba_xgb)],
    'XGBoost + SMOTE': [accuracy_score(y_test, y_pred_xgb_smote),
                       precision_score(y_test, y_pred_xgb_smote),
                       recall_score(y_test, y_pred_xgb_smote),
                       f1_score(y_test, y_pred_xgb_smote),
                       roc_auc_score(y_test, y_pred_proba_xgb_smote)]
}, index=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'])

print("\n--- XGBoost vs XGBoost with SMOTE ---")
print(comparison_smote)



## 12. Model Interpretation with SHAP

# Get one model for interpretation (best performing model)
best_model = xgb_pipeline

# Get preprocessed data for interpretation
X_test_preprocessed = preprocessor.transform(X_test)

# Calculate SHAP values
explainer = shap.Explainer(best_model[-1])
shap_values = explainer(X_test_preprocessed)

# Get feature names after preprocessing
preprocessor.fit(X_train)
transformed_features = []
for name, trans, cols in preprocessor.transformers_:
    if hasattr(trans, 'get_feature_names_out'):
        transformed_features.extend(trans.get_feature_names_out(cols))
    else:
        transformed_features.extend(cols)



# SHAP summary plot
shap.summary_plot(shap_values.values, X_test_preprocessed, feature_names=transformed_features)


# Dependence plots for top features
top_features = [transformed_features[i] for i in np.argsort(np.abs(shap_values.values).mean(0))[-5:]]
for feature in top_features:
    feature_idx = transformed_features.index(feature)
    shap.dependence_plot(feature_idx, shap_values.values, X_test_preprocessed, 
                        feature_names=transformed_features)




  
