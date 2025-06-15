# Student Academic Performance Analysis - Complete Script (Separated Visualizations)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set up visualizations
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12, 6)

# 1. Load Data (replace with your file path)
try:
    df = pd.read_csv('StudentsPerformance.csv')  # Kaggle's default filename
except:
    # Create synthetic data if file not found
    np.random.seed(42)
    data = {
        'gender': np.random.choice(['male', 'female'], 1000),
        'race/ethnicity': np.random.choice(['group A', 'group B', 'group C', 'group D', 'group E'], 1000),
        'parental level of education': np.random.choice([
            'some high school', 'high school', 'some college', 
            "associate's degree", "bachelor's degree", "master's degree"], 1000),
        'lunch': np.random.choice(['standard', 'free/reduced'], 1000),
        'test preparation course': np.random.choice(['none', 'completed'], 1000),
        'math score': np.random.normal(70, 15, 1000).clip(0, 100),
        'reading score': np.random.normal(75, 12, 1000).clip(0, 100),
        'writing score': np.random.normal(74, 13, 1000).clip(0, 100)
    }
    df = pd.DataFrame(data)

# 2. Data Cleaning
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
df['total_score'] = df['math_score'] + df['reading_score'] + df['writing_score']
df['average_score'] = df['total_score'] / 3

# 3. Basic EDA
print("\n=== BASIC STATISTICS ===")
print(df.describe())
print("\nMissing values:\n", df.isnull().sum())

# 4. Visualization: Score Distributions
plt.figure(figsize=(15, 5))
for i, subject in enumerate(['math_score', 'reading_score', 'writing_score'], 1):
    plt.subplot(1, 3, i)
    sns.histplot(df[subject], kde=True, bins=20)
    plt.title(f'{subject.replace("_", " ").title()} Distribution')
plt.tight_layout()
plt.show()

# 5. SEPARATE Visualizations for Each Factor
# Gender
plt.figure(figsize=(8, 6))
sns.boxplot(x='gender', y='average_score', data=df, palette='pastel')
plt.title('Average Score by Gender', pad=20)
plt.xlabel('Gender', labelpad=10)
plt.ylabel('Average Score', labelpad=10)
plt.show()

# Race/Ethnicity
plt.figure(figsize=(10, 6))
sns.boxplot(x='race/ethnicity', y='average_score', data=df, 
            order=['group A', 'group B', 'group C', 'group D', 'group E'],
            palette='Set2')
plt.title('Average Score by Race/Ethnicity', pad=20)
plt.xlabel('Race/Ethnicity Group', labelpad=10)
plt.ylabel('Average Score', labelpad=10)
plt.xticks(rotation=0)
plt.show()

# Parental Education
plt.figure(figsize=(12, 6))
order = ['some high school', 'high school', 'some college', 
         "associate's degree", "bachelor's degree", "master's degree"]
sns.boxplot(x='parental_level_of_education', y='average_score', 
            data=df, order=order, palette='coolwarm')
plt.title('Average Score by Parental Education Level', pad=20)
plt.xlabel('Parental Education', labelpad=10)
plt.ylabel('Average Score', labelpad=10)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Lunch
plt.figure(figsize=(8, 6))
sns.boxplot(x='lunch', y='average_score', data=df, palette='muted')
plt.title('Average Score by Lunch Type', pad=20)
plt.xlabel('Lunch Type', labelpad=10)
plt.ylabel('Average Score', labelpad=10)
plt.show()

# Test Preparation
plt.figure(figsize=(8, 6))
sns.boxplot(x='test_preparation_course', y='average_score', 
            data=df, palette='viridis')
plt.title('Average Score by Test Preparation', pad=20)
plt.xlabel('Test Preparation', labelpad=10)
plt.ylabel('Average Score', labelpad=10)
plt.show()

# 6. Statistical Analysis
print("\n=== STATISTICAL ANALYSIS ===")
# Gender difference
male = df[df['gender'] == 'male']['average_score']
female = df[df['gender'] == 'female']['average_score']
t_stat, p_val = stats.ttest_ind(male, female)
print(f"Gender difference p-value: {p_val:.4f} ({'Significant' if p_val < 0.05 else 'Not significant'})")

# Test prep difference
prep = df[df['test_preparation_course'] == 'completed']['average_score']
no_prep = df[df['test_preparation_course'] == 'none']['average_score']
t_stat, p_val = stats.ttest_ind(prep, no_prep)
print(f"Test prep difference p-value: {p_val:.4f} ({'Significant' if p_val < 0.05 else 'Not significant'})")

# 7. Correlation Analysis
plt.figure(figsize=(10, 8))
sns.heatmap(df[['math_score', 'reading_score', 'writing_score']].corr(), 
            annot=True, cmap='coolwarm', vmin=0, vmax=1)
plt.title('Score Correlation Matrix', pad=20)
plt.show()

# 8. Composite Visualization: Parental Education + Test Prep
plt.figure(figsize=(14, 8))
sns.violinplot(x='parental_level_of_education', y='average_score', 
               hue='test_preparation_course', data=df, order=order, 
               split=True, palette='Set2')
plt.title('Test Preparation Impact Across Parental Education Levels', pad=20)
plt.xlabel('Parental Education Level', labelpad=10)
plt.ylabel('Average Score', labelpad=10)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Test Prep Completed', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

print("\n=== ANALYSIS COMPLETE ===")