import pandas as pd
from sklearn.model_selection import train_test_split

data=pd.read_csv("data/raw/disease_risk.csv")
print("Initial shape:", data.shape)
print(data.head())


##we drop these because they are derived versions of bmi and may confuse our model
drop_cols = ['survey_code', 'bmi_estimated', 'bmi_scaled', 'bmi_corrected','education_level','job_type','occupation','income','pet_owner','device_usage','gene_marker_flag','environmental_risk_score','bmi','daily_steps','healthcare_access','insurance']         
data = data.drop(columns=[c for c in drop_cols if c in data.columns])
print("Shape after dropping redundant columns:", data.shape)


# Separate numeric and categorical columns
num_cols = data.select_dtypes(include='number').columns.tolist()
cat_cols = data.select_dtypes(include='object').columns.tolist()

print("Numeric columns:")
print(num_cols)

print("\nCategorical columns:")
print(cat_cols)

##handle missing values of numeric and categorical seperately

##fill numerics with median
for col in num_cols:
    data[col] = data[col].fillna(data[col].median())

# Fill categorical columns with mode / "Unknown"
for col in cat_cols:
    data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else "Unknown")


##standardizing remove extra spaces of all categorical cols
cat_cols = [
    'gender', 'sleep_quality', 'alcohol_consumption', 'smoking_level', 
    'mental_health_support', 'education_level', 'job_type', 'occupation', 
    'diet_type', 'exercise_type', 'device_usage', 'healthcare_access', 
    'insurance', 'sunlight_exposure', 'caffeine_intake', 'family_history', 
    'pet_owner', 'target'
]

# Remove extra spaces from all categorical columns
for col in cat_cols:
    if col in data.columns:  
        data[col] = data[col].str.strip()


#handle outliers
for col in num_cols:
    lower = data[col].quantile(0.01)
    upper = data[col].quantile(0.99)
    data[col] = data[col].clip(lower=lower, upper=upper)


# Save fully preprocessed dataset

data.to_csv("data/processed/preprocessed_data.csv", index=False)
print("Preprocessed dataset saved to data/processed/preprocess.csv")

# split the data to Train/test and save seperate csv's

X = data.drop(columns=['target'])
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#combain features of x_train and y_train into one datafram to save easier
train = pd.concat([X_train, y_train], axis=1)

#combain features of x_test and y_test into one datafram to save easier
test = pd.concat([X_test, y_test], axis=1)

train.to_csv("data/processed/training_data.csv", index=False)
test.to_csv("data/processed/testing_data.csv", index=False)
print("Train and Test datasets saved to data/processed/")