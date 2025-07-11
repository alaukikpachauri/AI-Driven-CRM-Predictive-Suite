# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 07:24:53 2025

@author: DeLL
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import streamlit as st
import sqlalchemy
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sqlparse
from io import StringIO
from streamlit_chat import message
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import uniform, randint
from xgboost import XGBClassifier
from sklearn.base import clone




model1 = None
model2 = None
model3 = None
numeric_features = None
categorical_features = None

# Function to parse SQL dump and extract data
def parse_sql_file_with_credentials(file, username, password):
    try:
        content = file.read().decode('utf-8')
        statements = sqlparse.split(content)
        data = []
        columns = None
        for stmt in statements:
            parsed = sqlparse.parse(stmt)[0]
            if parsed.get_type() == 'INSERT':
                table_name = parsed.tokens[4].get_name()
                if table_name == 'leads':
                    values = parsed.tokens[-1]
                    if isinstance(values, sqlparse.sql.Parenthesis):
                        rows = str(values).strip('()').split('),(')
                        if not columns:
                            create_stmt = next((s for s in statements if sqlparse.parse(s)[0].get_type() == 'CREATE'), None)
                            if create_stmt:
                                create_parsed = sqlparse.parse(create_stmt)[0]
                                for token in create_parsed.tokens:
                                    if isinstance(token, sqlparse.sql.Parenthesis):
                                        columns = [t.get_name() for t in token.tokens if isinstance(t, sqlparse.sql.Identifier)]
                        for row in rows:
                            row_values = row.split(',')
                            if len(row_values) == len(columns):
                                data.append(row_values)
        if data and columns:
            df = pd.DataFrame(data, columns=columns)
            return df
        # If no data found in SQL file, attempt database connection with provided credentials
        try:
            db_connection_str = f'mysql+pymysql://{username}:{password}@localhost:3306/leads_db'
            engine = sqlalchemy.create_engine(db_connection_str)
            query = "SELECT * FROM leads WHERE closed_date IS NOT NULL"
            df = pd.read_sql(query, engine)
            z = st.button('preview')
            st.write(' want to see the file preview, just click on the above button')
            if z:
                st.dataframe(' this is the data you have sellected',df)
            return df
        except Exception as e:
            st.error(f"Error connecting to database with provided credentials: {e}")
            return None
    except Exception as e:
        st.error(f"Error parsing SQL file: {e}")
        return None

# Function to load data from CSV or SQL
def load_data(uploaded_file, username=None, password=None):
    if uploaded_file is None:
        st.error("Please upload a file.")
        return None
    if uploaded_file.name.endswith('.csv'):
        
        try:
            df = pd.read_csv(uploaded_file)
            z = st.button('preview')
            st.write(' want to see the file preview, just click on the avove button')
            if z:
                st.dataframe(' this is the data you have sellected',df)
            return df
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return None
    elif uploaded_file.name.endswith('.sql'):
        username = st.text_input("MySQL Username:", "")
        password = st.text_input("MySQL Password:", type="password")
        if not username or not password:
            st.error("Please provide SQL username and password.")
            return None
        df = parse_sql_file_with_credentials(uploaded_file, username, password)
        if df is not None:
            return df
        else:
            st.error("No valid data found in SQL file or database.")
            return None
    else:
        st.error("Unsupported file type. Please upload a CSV or SQL file.")
        return None


# Function to preprocess data and create pipeline
class run_lead_scoring_prediction():
    
    numeric_features = ['how_old_days', 'untouched_since_days']
    categorical_features = [
        'lead_source', 'industry', 'customer_type', 'product_sub_category',
        'status', 'assigned_user_id', 'team_id', 'salutation', 'designation',
        'department', 'hierarchy', 'area', 'address_type', 'city', 'state', 'country'
    ]
   # if model1 is not None:
        

class run_opportunity_win_prediction():
    numeric_features = ['how_old_days', 'untouched_since_days', 'probability', 'amount', 'gross_profit']
    categorical_features = [ 'opportunity_type', 'product_sub_category', 'priority', 'customer_type', 'lead_source', 'next_followup_type']
   
    
class run_churn_risk():
    numeric_features = ['created_at', 'ratting', 'migrated_to_tally', 'updated_at']
    categorical_features = ['customer_type', 'industry' , 'latest_comment', 'assigned_user_id']

def create_pipeline(numeric_features, categorical_features, df):
    high_missing = []
    low_missing = []

    for col in numeric_features:
        missing_percent = df[col].isna().mean()
        if missing_percent > 0.35:
            st.error(f"Column '{col}' has >35% missing values.")
            continue
        elif missing_percent > 0.18:
            high_missing.append(col)
        elif missing_percent > 0.05:
            low_missing.append(col)

    scaler = RobustScaler()

    high_missing_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer(random_state=42)),
        ('scaler', scaler)
    ])

    low_missing_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', scaler)
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('high_missing_num', high_missing_transformer, high_missing),
            ('low_missing_num', low_missing_transformer, low_missing),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        ))
    ])

    return pipeline, scaler



    # Function to plot scatter plot
def plot_scatter(app_mode,df):
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='how_old_days', y='untouched_since_days', hue='converted', palette=['#FF5733', '#33FF57'])
        if app_mode == "Lead Scoring":
            plt.title('Lead Age vs. Untouched Days by Conversion Status')
                
        elif app_mode == "Opportunity Win":
            plt.tick_params('opportunity win vs. untouched  Days by Conversion Status')
            
        st.pyplot(fig)

def plot_accuracy_improvement_plotly(baseline_acc, tuned_acc):
    import pandas as pd

    df_plot = pd.DataFrame({
        'Stage': ['Before Tuning', 'After Tuning'],
        'ROC AUC Score': [baseline_acc, tuned_acc]
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_plot['Stage'],
        y=df_plot['ROC AUC Score'],
        mode='lines+markers+text',
        text=[f"{baseline_acc:.2%}", f"{tuned_acc:.2%}"],
        textposition="top center",
        marker=dict(size=12, color=['red', 'green']),
        line=dict(color='blue', dash='dash')
    ))

    improvement = tuned_acc - baseline_acc
    fig.update_layout(
        title="Model ROC AUC Score: Before vs After Hyperparameter Tuning",
        yaxis=dict(title="ROC AUC Score", range=[0, 1]),
        xaxis=dict(title="Training Stage"),
        annotations=[
            dict(
                x=1,
                y=tuned_acc,
                text=f"+{improvement:.2%} improvement",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            )
        ]
    )
    st.plotly_chart(fig, use_container_width=True)

# Streamlit app
def main():
    numeric_features = []
    categorical_features = []
    global model1 , model2 ,model3
    st.set_page_config(page_title="Choose AI features", layout="wide",)
    
    app_mode = st.selectbox("Choose App", ["Lead Scoring", "Opportunity Win", 'Churn Risk'])
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Train Model", "Predict"])
    
      # align's the message to the right
    
 
    

    if app_mode == "Lead Scoring":
        run_lead_scoring_prediction()
        if model1 is not None:
            st.error('Dislamer:')
            st.text('we are going to retrain the model again')
            
    elif app_mode == "Opportunity Win":
        run_opportunity_win_prediction()
        if model2 is not None:
            st.error('Dislamer:')
            st.text('we are going to retrain the model again')
            
    elif app_mode == "Churn Risk":
       run_churn_risk()
       if model3 is not None:
           st.error('Dislamer:')
           st.text('we are going to retrain the model again')
    
    if page == "Train Model":
        st.header(app_mode)
        uploaded_file = st.file_uploader("Upload CSV or SQL file", type=['csv', 'sql'])
        st.text('welcome to the interactive')
        st.write('hi mr.khemraj')
        dev_option = st.checkbox(" do you want to switch on developer option")
        
        with st.expander('open to chat box'):
            st. write(" this feature is currently in developing")
            message("My message") 
            message("Hello bot!", is_user=True)
        if os.path.exists('lead_scoring_model.joblib'):
            st.warning("A trained model already exists.")
            if st.button("Delete existing model and retrain"):
                os.remove('lead_scoring_model.joblib')
                st.success("Existing model deleted. You can now train a new model.")
      
        if uploaded_file is not None:
            if st.button("Load Data and Train Model"):
                with st.spinner("Loading data..."):
                    df = load_data(uploaded_file)
                    if df is not None:
                        st.info("Data loaded successfully!")
                        if 'converted' not in df.columns or 'closed_date' not in df.columns:
                            st.error("Required columns 'converted' or 'closed_date' missing.")
                            return
                        df = df[df['closed_date'].notnull()]
                        st.info("Displaying data scatter plot...")
                        
                        if app_mode == "Lead Scoring":
                           numeric_features = run_lead_scoring_prediction.numeric_features
                           categorical_features = run_lead_scoring_prediction.categorical_features


                        elif app_mode == "Opportunity Win":
                             numeric_features = run_opportunity_win_prediction.numeric_features
                             categorical_features = run_opportunity_win_prediction.categorical_features
                             
                        elif app_mode == "churn risk":
                             numeric_features = run_churn_risk().numeric_features
                             categorical_features = run_churn_risk().categorical_features



                        features_to_use = [col for col in (numeric_features + categorical_features) if col in df.columns]
                        X = df[features_to_use]
                        if app_mode =="Lead Scoring": 
                            y = df['converted']
                        elif app_mode == "Opportunity Win":
                              y = df['sales_data']
                              
                        elif app_mode == 'Churn Risk':
                            y = df ['account_Staues']
                        for col in categorical_features:
                            if col in X.columns:
                                X[col] = X[col].astype('category')

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )
                        for col in categorical_features:
                           if col in X_train.columns:
                              X_train[col] = X_train[col].astype('category')
                              X_test[col] = X_test[col].astype('category')

                        with st.spinner("Preprocessing data..."):
                            pipeline, _ = create_pipeline(numeric_features, categorical_features,df)
                        
                        if dev_option: 
                            st.write("✅ X_train shape:", X_train.shape)
                            st.write("✅ y_train shape:", y_train.shape)
                            st.write("✅ X_train columns:", X_train.columns.tolist())

                            st.write("❗ Nulls in X_train:", X_train.isnull().sum())
                            st.write("❗ Nulls in y_train:", y_train.isnull().sum())

                            st.write("🔍 First few rows of X_train:", X_train.head())
                            st.write("🔍 First few rows of y_train:", y_train.head())


                        with st.spinner("Training baseline model..."):
                            pipeline.fit(X_train, y_train)
                            baseline_accuracy = pipeline.score(X_test, y_test)
                            

# Step 1: create the baseline XGBoost classifier only
                            baseline_clf = xgb.XGBClassifier(
                                 learning_rate=0.1,
                                 max_depth=6,
                                 n_estimators=100,
                                 subsample=0.9,
                                 colsample_bytree=0.9,
                                 use_label_encoder=False,
                                 eval_metric="logloss",
                                 random_state=42
                             )

# Step 2: clone pipeline and replace classifier with baseline_clf
                            baseline_model = clone(pipeline)
                            baseline_model.steps[-1] = ('classifier', baseline_clf)
                            
                        with st.spinner("Performing hyperparameter tuning..."):
                            param_distributions = {
                               "classifier__learning_rate": uniform(0.01, 0.2),
                               "classifier__max_depth": randint(3, 10),
                               "classifier__n_estimators": randint(50, 300),
                               "classifier__subsample": uniform(0.6, 0.4),
                               "classifier__colsample_bytree": uniform(0.6, 0.4)
                                     }
                     
# Use stratified folds to avoid imbalance
                        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Use RandomizedSearch instead of GridSearch
                        search = RandomizedSearchCV(
                            pipeline,
                            param_distributions=param_distributions,
                            n_iter=30,  # Try 30 combinations
                            scoring="roc_auc",
                            cv=cv,
                            verbose=2,
                            n_jobs=-1,
                            random_state=42
                        )

                       # Step 1: Fit baseline model and score
                        baseline_model.fit(X_train, y_train)
                        baseline_proba = baseline_model.predict_proba(X_test)[:, 1]
                        baseline_accuracy = roc_auc_score(y_test, baseline_proba)

# Step 2: Fit tuned model via RandomizedSearchCV
                        search.fit(X_train, y_train)
                        tuned_model = search.best_estimator_
                        tuned_proba = tuned_model.predict_proba(X_test)[:, 1]
                        tuned_accuracy = roc_auc_score(y_test, tuned_proba)

# Step 3: Compare and select the better model
                        if tuned_accuracy >= baseline_accuracy:
                             model = tuned_model
                             final_accuracy = tuned_accuracy
                             model_source = "Tuned Model"
                        else:
                             model = baseline_model
                             final_accuracy = baseline_accuracy
                             model_source = "Baseline Model"

                        
                        
                        if app_mode == "Lead Scoring":
                            joblib.dump(model, 'lead_scoring_model.joblib')
                            model1 = joblib.load('lead_scoring_model.joblib',mmap_mode='r')
                           
                            
                        elif app_mode == "Opportunity Win":
                            joblib.dump(model, filename = 'opportunit.joblib')
                        
                           
                            model2 = joblib.load('opportunit.joblib',mmap_mode='r')
                            
                            
                        elif app_mode == "Churn Risk":
                            joblib.dump(model, filename = 'churn.joblib')
                            model3 = joblib.load('churn.joblib',mmap_mode='r')
                            
                            
                        st.success("Model trained and saved successfully!")

                        tuned_model = search.best_estimator_
                        tuned_proba = tuned_model.predict_proba(X_test)[:, 1]
                        tuned_accuracy = roc_auc_score(y_test, tuned_proba)
                        improvement = ((tuned_accuracy - baseline_accuracy) / baseline_accuracy * 100) if baseline_accuracy > 0 else 0

                        st.write(f"✅ **{model_source} selected based on higher ROC AUC**")
                        st.write(f"**Baseline Accuracy**: {baseline_accuracy:.4f}")
                        st.write(f"**Tuned Accuracy (ROC AUC)**: {tuned_accuracy:.4f}")
                        st.write(f"**Final Accuracy Used**: {final_accuracy:.4f}")
                        st.write(f"**Best Parameters (if tuned used)**: {search.best_params_}")
                        st.write(f"**CV Best ROC AUC**: {search.best_score_:.4f}")


                        plot_accuracy_improvement_plotly(baseline_accuracy, tuned_accuracy)
                        if st.checkbox("Use baseline model instead of tuned"):
                            model = baseline_model
                        else:
                            model = search.best_estimator_

                        # Update chart data
                        st.session_state['chart_data'] = {
                            "data": {
                                "labels": ["Baseline Accuracy", "Tuned Accuracy"],
                                "datasets": [{
                                    "label": "Model Accuracy",
                                    "data": [baseline_accuracy, tuned_accuracy],
                                    "backgroundColor": ["#4CAF50", "#2196F3"],
                                    "borderColor": ["#388E3C", "#1976D2"],
                                    "borderWidth": 1
                                }]
                            },
                            "options": {
                                "scales": {
                                    "y": {
                                       
                                       
                                        "max": 1
                                    },
                                    "x": {
                                      
                                    }
                                },
                                "plugins": {
                                    
                                    "annotation": {
                                        "annotations": {
                                            "improvement": {
                                                "type": "label",
                                                "xValue": 1,
                                                "yValue": tuned_accuracy,
                                                "content": [f"Improvement: {improvement:.2f}%"],
                                                "backgroundColor": "rgba(0, 0, 0, 0.7)",
                                                "color": "#FFFFFF",
                                                "padding": 6,
                                                "borderRadius": 4
                                            }
                                        }
                                    }
                                }
                            }
                        }
    
    elif page == "Predict":
       

        st.text('welcome to the interactive')
        st.write('hi mr.khemraj')
       
        with st.expander('open to chat box'):
            st. write(" this feature is currently in developing")
            message("My message") 
            message("Hello bot!", is_user=True)
            # align's the message to the right
            
     
        if app_mode == "Lead Scoring":
           st.header("Predict Lead Conversion")
           numeric_features = run_lead_scoring_prediction.numeric_features
           categorical_features = run_lead_scoring_prediction.categorical_features

           if not os.path.exists('lead_scoring_model.joblib'):
               st.error("Model not trained yet. Please go to the Train Model page.")
               if st.button("Go to Train Model"):
                   st.experimental_rerun()
           else:
               model = joblib.load('lead_scoring_model.joblib')
               st.subheader("Enter Lead Details")
               input_data = {}
               for feature in numeric_features + categorical_features:
                   if feature in numeric_features:
                       input_data[feature] = st.number_input(f"{feature}", value=0.0)
                   else:
                       input_data[feature] = st.text_input(f"{feature}", "")
               if st.button("Predict"):
                   input_df = pd.DataFrame([input_data])
                   input_df = input_df[numeric_features + categorical_features]

                   prob = model.predict_proba(input_df)[:, 1][0]
                   st.write(f"Probability of Conversion: {prob:.4f}")

            
           
           
               
        elif app_mode == "Opportunity Win":
            numeric_features = run_opportunity_win_prediction.numeric_features
            categorical_features = run_opportunity_win_prediction.categorical_features

            st.header ("oppurnity win predictor")
            if not os.path.exists('opportunit.joblib'):
                st.error("Model is not yet trained. Pleasse go to the train model page.")
                if st.buttton(" go to Train Model"):
                    st.experimental_rerun()
                else:
                    input_data = {}
                    for feature in numeric_features + categorical_features:
                        if feature in numeric_features:
                            input_data[feature] = st.number_input(f"{feature}", value=0.0)
                        else:
                            input_data[feature] = st.text_input(f"{feature}", "")
                    if st.button("Predict"):
                        input_df = pd.DataFrame([input_data])
                        input_df = input_df[numeric_features + categorical_features]
                        st.write(f"Probability of Conversion: {prob:.4f}")
                        
                        
        elif app_mode == "Churn Risk":
             numeric_features = run_opportunity_win_prediction.numeric_features
             categorical_features = run_opportunity_win_prediction.categorical_features

             st.header ("Churn Risk predictor")
             if not os.path.exists('churn.joblib'):
                 st.error("Model is not yet trained. Pleasse go to the train model page.")
                 if st.button(" go to Train Model"):
                     st.experimental_rerun()
                 else:
                     input_data = {}
                     for feature in numeric_features + categorical_features:
                         if feature in numeric_features:
                             input_data[feature] = st.number_input(f"{feature}", value=0.0)
                         else:
                             input_data[feature] = st.text_input(f"{feature}", "")
                     if st.button("Predict"):
                         input_df = pd.DataFrame([input_data])
                         input_df = input_df[numeric_features + categorical_features]
                         st.write(f"Probability of Conversion: {prob:.4f}")               

           

            
if __name__ == "__main__":
    main()

                
             
        
        
          
       
    
    


















# Function to load data from CSV or SQL with username and password for SQL
