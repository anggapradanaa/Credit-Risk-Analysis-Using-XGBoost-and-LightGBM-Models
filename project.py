import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
import lightgbm as lgb
import time

# Progress tracking
def print_progress(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

# Load dan preprocessing data yang ditingkatkan
def load_and_preprocess_data():
    print_progress("Loading data...")
    # Load data
    df = pd.read_csv(r"C:\Users\ACER\Downloads\loan_data_2007_2014.csv")
    
    # PERBAIKAN: Hapus kolom yang tidak diperlukan termasuk ID columns
    columns_to_drop = ['Unnamed: 0', 'url', 'id', 'member_id']
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_cols_to_drop:
        print_progress(f"Dropping ID columns: {existing_cols_to_drop}")
        df.drop(columns=existing_cols_to_drop, inplace=True)
    
    # Define default statuses
    default_statuses = [
        'Charged Off', 'Late (31-120 days)', 'Late (16-30 days)', 
        'Default', 'Does not meet the credit policy. Status:Charged Off'
    ]
    
    # Add is_default column
    df['is_default'] = df['loan_status'].apply(lambda x: 1 if x in default_statuses else 0)
    
    print_progress("Handling missing values...")
    # Handle missing values - keep columns with <= 40% missing (lebih strict)
    missing_percent = df.isnull().mean() * 100
    columns_to_keep = missing_percent[missing_percent <= 40].index
    df = df[columns_to_keep]
    
    # Additional preprocessing
    print_progress("Feature engineering...")
    
    # Create new features
    if 'annual_inc' in df.columns and 'loan_amnt' in df.columns:
        df['loan_to_income_ratio'] = df['loan_amnt'] / (df['annual_inc'] + 1)
    
    if 'int_rate' in df.columns and 'loan_amnt' in df.columns:
        df['interest_payment'] = df['int_rate'] * df['loan_amnt'] / 100
    
    # Remove outliers untuk numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['is_default']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df

def preprocess_features(X_train, X_test):
    print_progress("Preprocessing features...")
    
    # Process term column
    if 'term' in X_train.columns:
        X_train['term'] = X_train['term'].str.replace(' months', '').astype(int)
        X_test['term'] = X_test['term'].str.replace(' months', '').astype(int)
    
    # Process employment length dengan improvement
    if 'emp_length' in X_train.columns:
        # Convert emp_length to numeric
        emp_length_mapping = {
            '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
            '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
            '10+ years': 10
        }
        
        X_train['emp_length'] = X_train['emp_length'].map(emp_length_mapping)
        X_test['emp_length'] = X_test['emp_length'].map(emp_length_mapping)
        
        # Fill missing with median
        median_emp = X_train['emp_length'].median()
        X_train['emp_length'].fillna(median_emp, inplace=True)
        X_test['emp_length'].fillna(median_emp, inplace=True)
    
    # Process date columns
    col_date = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d']
    
    for col in col_date:
        if col in X_train.columns:
            # Fill missing values
            mode_value_train = X_train[col].mode()[0] if not X_train[col].mode().empty else 'Jan-2010'
            X_train[col].fillna(mode_value_train, inplace=True)
            X_test[col].fillna(mode_value_train, inplace=True)
            
            # Convert to datetime dan extract features
            X_train[col] = pd.to_datetime(X_train[col], format='%b-%y', errors='coerce')
            X_test[col] = pd.to_datetime(X_test[col], format='%b-%y', errors='coerce')
            
            # Extract year and month
            X_train[f'{col}_year'] = X_train[col].dt.year
            X_train[f'{col}_month'] = X_train[col].dt.month
            X_test[f'{col}_year'] = X_test[col].dt.year
            X_test[f'{col}_month'] = X_test[col].dt.month
            
            # Drop original date column
            X_train.drop(columns=[col], inplace=True)
            X_test.drop(columns=[col], inplace=True)
    
    # Fill numeric columns with median
    X_train_numeric = X_train.select_dtypes(include='number').columns
    for col in X_train_numeric:
        median_value = X_train[col].median()
        X_train[col].fillna(median_value, inplace=True)
        X_test[col].fillna(median_value, inplace=True)
    
    # PERBAIKAN UTAMA: Handle categorical columns dengan lebih agresif untuk menghindari memory error
    X_train_categorical = X_train.select_dtypes(include=['object', 'category']).columns
    print_progress(f"Processing {len(X_train_categorical)} categorical columns...")
    
    for col in X_train_categorical:
        print_progress(f"Processing column {col} with {X_train[col].nunique()} unique values...")
        
        # Fill missing values
        mode_value = X_train[col].mode()[0] if not X_train[col].mode().empty else 'Unknown'
        X_train[col].fillna(mode_value, inplace=True)
        X_test[col].fillna(mode_value, inplace=True)
        
        # PERBAIKAN: Sangat agresif dalam mengurangi kardinalitas
        # Hanya keep top 5 categories untuk menghindari memory explosion
        unique_values = X_train[col].nunique()
        
        if unique_values > 10:  # Jika terlalu banyak kategori
            print_progress(f"Reducing cardinality for {col}: {unique_values} -> max 5 categories")
            # Keep only top 4 categories, sisanya jadi 'Other'
            top_categories = X_train[col].value_counts().head(4).index
            X_train[col] = X_train[col].apply(lambda x: x if x in top_categories else 'Other')
            X_test[col] = X_test[col].apply(lambda x: x if x in top_categories else 'Other')
    
    # TAMBAHAN: Drop kolom dengan kardinalitas tinggi yang masih bermasalah
    final_check_cols = X_train.select_dtypes(include=['object', 'category']).columns
    for col in final_check_cols:
        unique_count = X_train[col].nunique()
        if unique_count > 15:  # Threshold lebih ketat
            print_progress(f"WARNING: Dropping high cardinality column {col} with {unique_count} unique values")
            X_train.drop(columns=[col], inplace=True)
            X_test.drop(columns=[col], inplace=True)
    
    return X_train, X_test

def train_models(X_train, X_test, y_train, y_test):
    print_progress("Setting up preprocessing pipeline...")
    
    # Get column names after preprocessing
    num_cols = X_train.select_dtypes(include='number').columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print_progress(f"Final feature count - Numeric: {len(num_cols)}, Categorical: {len(cat_cols)}")
    
    # Check cardinality of categorical columns one more time
    for col in cat_cols:
        cardinality = X_train[col].nunique()
        print_progress(f"Categorical column {col}: {cardinality} unique values")
    
    # Define preprocessing pipelines dengan parameter yang lebih konservatif
    numerical_pipeline = Pipeline([
        ('scaler', StandardScaler()),
    ])
    
    # PERBAIKAN: Gunakan max_categories yang sangat rendah dan sparse=False
    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(
            handle_unknown='ignore', 
            sparse_output=False,  # Pastikan False untuk menghindari sparse matrix
            max_categories=5,     # Sangat rendah untuk menghindari memory issue
            drop='if_binary'      # Drop one category for binary features
        ))
    ])
    
    preprocessor = ColumnTransformer([
        ('numeric', numerical_pipeline, num_cols),
        ('categoric', categorical_pipeline, cat_cols)
    ])
    
    # PERBAIKAN: Kurangi sample size untuk SMOTE jika data terlalu besar
    print_progress("Applying SMOTE for data balancing...")
    
    # Check data size
    total_features_estimated = len(num_cols) + sum([X_train[col].nunique() for col in cat_cols]) * 5
    print_progress(f"Estimated total features after encoding: {total_features_estimated}")
    
    # Jika estimasi terlalu besar, sample data terlebih dahulu
    if len(X_train) > 50000 and total_features_estimated > 1000:
        print_progress("Data too large, sampling 50,000 records for training...")
        sample_indices = np.random.choice(X_train.index, size=50000, replace=False)
        X_train_sample = X_train.loc[sample_indices]
        y_train_sample = y_train.loc[sample_indices]
    else:
        X_train_sample = X_train
        y_train_sample = y_train
    
    # SMOTE dengan parameter yang lebih konservatif
    smote = SMOTE(random_state=42, k_neighbors=3)
    
    # XGBoost Pipeline
    print_progress("Training XGBoost model...")
    pipeline_xgb = ImbPipeline([
        ('prep', preprocessor),
        ('smote', smote),
        ('algo', xgb.XGBClassifier(
            random_state=42,
            n_estimators=50,  # Reduced from 100
            max_depth=4,      # Reduced from 6
            learning_rate=0.1,
            eval_metric='logloss',
            n_jobs=1          # Single thread to reduce memory usage
        ))
    ])
    
    pipeline_xgb.fit(X_train_sample, y_train_sample)
    y_pred_xgb = pipeline_xgb.predict(X_test)
    y_pred_proba_xgb = pipeline_xgb.predict_proba(X_test)[:, 1]
    
    # LightGBM Pipeline
    print_progress("Training LightGBM model...")
    pipeline_lgbm = ImbPipeline([
        ('prep', preprocessor),
        ('smote', smote),
        ('algo', lgb.LGBMClassifier(
            random_state=42,
            n_estimators=50,  # Reduced from 100
            max_depth=4,      # Reduced from 6
            learning_rate=0.1,
            verbose=-1,
            n_jobs=1          # Single thread to reduce memory usage
        ))
    ])
    
    pipeline_lgbm.fit(X_train_sample, y_train_sample)
    y_pred_lgbm = pipeline_lgbm.predict(X_test)
    y_pred_proba_lgbm = pipeline_lgbm.predict_proba(X_test)[:, 1]
    
    # TAMBAHAN: Mendapatkan feature importance
    print_progress("Extracting feature importance...")
    
    # Get feature names after preprocessing
    feature_names = []
    
    # Numeric features
    feature_names.extend(num_cols)
    
    # Categorical features (after one-hot encoding)
    if cat_cols:
        onehot_encoder = pipeline_xgb.named_steps['prep'].named_transformers_['categoric'].named_steps['onehot']
        cat_feature_names = onehot_encoder.get_feature_names_out(cat_cols)
        feature_names.extend(cat_feature_names)
    
    # XGBoost feature importance
    xgb_importance = pipeline_xgb.named_steps['algo'].feature_importances_
    xgb_feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_importance
    }).sort_values('importance', ascending=False)
    
    # LightGBM feature importance
    lgbm_importance = pipeline_lgbm.named_steps['algo'].feature_importances_
    lgbm_feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': lgbm_importance
    }).sort_values('importance', ascending=False)
    
    print_progress("Model training completed!")
    
    return {
        'xgb': {
            'model': pipeline_xgb,
            'predictions': y_pred_xgb,
            'probabilities': y_pred_proba_xgb,
            'feature_importance': xgb_feature_importance
        },
        'lgbm': {
            'model': pipeline_lgbm,
            'predictions': y_pred_lgbm,
            'probabilities': y_pred_proba_lgbm,
            'feature_importance': lgbm_feature_importance
        }
    }

# Load data
df = load_and_preprocess_data()

# Split data
print_progress("Splitting data...")
X = df.drop(columns=['is_default', 'loan_status'])
y = df['is_default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocess features
X_train, X_test = preprocess_features(X_train.copy(), X_test.copy())

# Train models
models = train_models(X_train, X_test, y_train, y_test)

# Calculate metrics for both models
print_progress("Calculating metrics...")
metrics = {}
for model_name, model_data in models.items():
    roc_auc = roc_auc_score(y_test, model_data['probabilities'])
    accuracy = accuracy_score(y_test, model_data['predictions'])
    conf_matrix = confusion_matrix(y_test, model_data['predictions'])
    fpr, tpr, _ = roc_curve(y_test, model_data['probabilities'])
    class_report = classification_report(y_test, model_data['predictions'])
    
    metrics[model_name] = {
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'conf_matrix': conf_matrix,
        'fpr': fpr,
        'tpr': tpr,
        'classification_report': class_report
    }
    
    print(f"{model_name.upper()} - ROC AUC: {roc_auc:.4f}, Accuracy: {accuracy:.4f}")

print_progress("Initializing dashboard...")

# Initialize Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    html.H1("Dashboard Prediksi Risiko Kredit (XGBoost vs LightGBM)", 
            style={'text-align': 'center', 'margin-bottom': '30px'}),
    
    # Model comparison cards
    html.Div([
        html.Div([
            html.H3("XGBoost"),
            html.P(f"ROC AUC: {metrics['xgb']['roc_auc']:.4f}", style={'color': '#1f77b4', 'font-size': '18px'}),
            html.P(f"Accuracy: {metrics['xgb']['accuracy']:.4f}", style={'color': '#1f77b4', 'font-size': '18px'})
        ], className='metric-card', style={'width': '23%', 'display': 'inline-block', 'margin': '1%', 
                                         'padding': '20px', 'border': '2px solid #1f77b4', 'border-radius': '5px'}),
        
        html.Div([
            html.H3("LightGBM"),
            html.P(f"ROC AUC: {metrics['lgbm']['roc_auc']:.4f}", style={'color': '#ff7f0e', 'font-size': '18px'}),
            html.P(f"Accuracy: {metrics['lgbm']['accuracy']:.4f}", style={'color': '#ff7f0e', 'font-size': '18px'})
        ], className='metric-card', style={'width': '23%', 'display': 'inline-block', 'margin': '1%', 
                                         'padding': '20px', 'border': '2px solid #ff7f0e', 'border-radius': '5px'}),
        
        html.Div([
            html.H3("Total Samples"),
            html.H2(f"{len(df):,}", style={'color': '#2ca02c'})
        ], className='metric-card', style={'width': '23%', 'display': 'inline-block', 'margin': '1%', 
                                         'padding': '20px', 'border': '1px solid #ddd', 'border-radius': '5px'}),
        
        html.Div([
            html.H3("Default Rate"),
            html.H2(f"{df['is_default'].mean():.2%}", style={'color': '#d62728'})
        ], className='metric-card', style={'width': '23%', 'display': 'inline-block', 'margin': '1%', 
                                         'padding': '20px', 'border': '1px solid #ddd', 'border-radius': '5px'})
    ], style={'margin-bottom': '30px'}),
    
    # Tabs
    dcc.Tabs(id="tabs", value='overview', children=[
        dcc.Tab(label='Overview & EDA', value='overview'),
        dcc.Tab(label='Model Comparison', value='performance'),
        dcc.Tab(label='Data Exploration', value='exploration'),
        dcc.Tab(label='Feature Importance', value='feature_importance'),  # Tab baru
        dcc.Tab(label='Predictions', value='predictions')
    ]),
    
    html.Div(id='tab-content')
])

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value'),
    prevent_initial_call=False
)
def render_content(tab):
    try:
        if tab == 'overview':
            # Enhanced EDA
            return html.Div([
                html.H2("Enhanced Exploratory Data Analysis"),
                
                html.Div([
                    html.Div([
                        dcc.Graph(
                            figure=px.pie(
                                values=[len(df) - df['is_default'].sum(), df['is_default'].sum()],
                                names=['Non-Default', 'Default'],
                                title="Distribusi Status Default",
                                color_discrete_sequence=['#2ca02c', '#d62728']
                            )
                        )
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    
                    html.Div([
                        dcc.Graph(
                            figure=px.histogram(
                                df, x='loan_amnt', nbins=50,
                                title="Distribusi Jumlah Pinjaman",
                                marginal="box"
                            )
                        )
                    ], style={'width': '50%', 'display': 'inline-block'})
                ]),
                
                html.Div([
                    html.Div([
                        dcc.Graph(
                            figure=px.violin(
                                df, x='is_default', y='int_rate',
                                title="Distribusi Tingkat Bunga vs Default Status",
                                box=True
                            )
                        )
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    
                    html.Div([
                        dcc.Graph(
                            figure=px.sunburst(
                                df.groupby(['grade', 'is_default']).size().reset_index(name='count'),
                                path=['grade', 'is_default'], values='count',
                                title="Grade Distribution by Default Status"
                            )
                        )
                    ], style={'width': '50%', 'display': 'inline-block'})
                ]),
                
                # Correlation heatmap
                html.Div([
                    dcc.Graph(
                        figure=px.imshow(
                            df.select_dtypes(include=[np.number]).corr(),
                            title="Correlation Matrix",
                            aspect="auto"
                        )
                    )
                ])
            ])
        
        elif tab == 'performance':
            return html.Div([
                html.H2("Perbandingan Performa Model"),
                
                # ROC Curves Comparison
                html.Div([
                    dcc.Graph(
                        figure=go.Figure().add_trace(
                            go.Scatter(x=metrics['xgb']['fpr'], y=metrics['xgb']['tpr'],
                                     mode='lines', name=f'XGBoost (AUC = {metrics["xgb"]["roc_auc"]:.4f})',
                                     line=dict(color='#1f77b4', width=3))
                        ).add_trace(
                            go.Scatter(x=metrics['lgbm']['fpr'], y=metrics['lgbm']['tpr'],
                                     mode='lines', name=f'LightGBM (AUC = {metrics["lgbm"]["roc_auc"]:.4f})',
                                     line=dict(color='#ff7f0e', width=3))
                        ).add_trace(
                            go.Scatter(x=[0, 1], y=[0, 1],
                                     mode='lines', name='Random',
                                     line=dict(dash='dash', color='gray'))
                        ).update_layout(
                            title='ROC Curves Comparison',
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate',
                            width=800, height=600
                        )
                    )
                ], style={'text-align': 'center'}),
                
                # Confusion Matrices
                html.Div([
                    html.Div([
                        dcc.Graph(
                            figure=go.Figure(data=go.Heatmap(
                                z=metrics['xgb']['conf_matrix'],
                                x=['Predicted Non-Default', 'Predicted Default'],
                                y=['Actual Non-Default', 'Actual Default'],
                                text=metrics['xgb']['conf_matrix'],
                                texttemplate="%{text}",
                                textfont={"size": 16},
                                colorscale='Blues'
                            )).update_layout(title="XGBoost Confusion Matrix")
                        )
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    
                    html.Div([
                        dcc.Graph(
                            figure=go.Figure(data=go.Heatmap(
                                z=metrics['lgbm']['conf_matrix'],
                                x=['Predicted Non-Default', 'Predicted Default'],
                                y=['Actual Non-Default', 'Actual Default'],
                                text=metrics['lgbm']['conf_matrix'],
                                texttemplate="%{text}",
                                textfont={"size": 16},
                                colorscale='Oranges'
                            )).update_layout(title="LightGBM Confusion Matrix")
                        )
                    ], style={'width': '50%', 'display': 'inline-block'})
                ]),
                
                # Classification Reports
                html.Div([
                    html.Div([
                        html.H3("XGBoost Classification Report"),
                        html.Pre(metrics['xgb']['classification_report'],
                                style={'background-color': '#f0f8ff', 'padding': '15px', 'border-radius': '5px'})
                    ], style={'width': '50%', 'display': 'inline-block', 'padding-right': '10px'}),
                    
                    html.Div([
                        html.H3("LightGBM Classification Report"),
                        html.Pre(metrics['lgbm']['classification_report'],
                                style={'background-color': '#fff8f0', 'padding': '15px', 'border-radius': '5px'})
                    ], style={'width': '50%', 'display': 'inline-block', 'padding-left': '10px'})
                ])
            ])
        
        elif tab == 'exploration':
            return html.Div([
                html.H2("Eksplorasi Data Interaktif"),
                
                html.Div([
                    html.Label("Pilih Variabel untuk Analisis:"),
                    dcc.Dropdown(
                        id='variable-dropdown',
                        options=[
                            {'label': 'Loan Amount', 'value': 'loan_amnt'},
                            {'label': 'Interest Rate', 'value': 'int_rate'},
                            {'label': 'Annual Income', 'value': 'annual_inc'},
                            {'label': 'DTI', 'value': 'dti'},
                            {'label': 'Home Ownership', 'value': 'home_ownership'},
                            {'label': 'Purpose', 'value': 'purpose'},
                            {'label': 'Term', 'value': 'term'},
                            {'label': 'Grade', 'value': 'grade'}
                        ],
                        value='loan_amnt',
                        clearable=False  # Tambahkan ini untuk mencegah nilai None
                    )
                ], style={'margin-bottom': '20px'}),
                
                html.Div(id='exploration-content', children=[
                    html.Div([
                        html.H3("Loading..."),
                        html.P("Sedang memuat visualisasi...")
                    ])
                ])  # Berikan children default untuk mencegah error
            ])
        
        elif tab == 'predictions':
            # Create results dataframe untuk kedua model
            results_df_xgb = pd.DataFrame({
                'Actual': y_test.values,
                'Predicted_XGB': models['xgb']['predictions'],
                'Probability_XGB': models['xgb']['probabilities']
            })
            
            results_df_lgbm = pd.DataFrame({
                'Actual': y_test.values,
                'Predicted_LGBM': models['lgbm']['predictions'],
                'Probability_LGBM': models['lgbm']['probabilities']
            })
            
            return html.Div([
                html.H2("Hasil Prediksi - Perbandingan Model"),
                
                html.Div([
                    html.Div([
                        dcc.Graph(
                            figure=px.histogram(
                                results_df_xgb, x='Probability_XGB',
                                nbins=50,
                                title="Distribusi Probabilitas XGBoost"
                            )
                        )
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    
                    html.Div([
                        dcc.Graph(
                            figure=px.histogram(
                                results_df_lgbm, x='Probability_LGBM',
                                nbins=50,
                                title="Distribusi Probabilitas LightGBM"
                            )
                        )
                    ], style={'width': '50%', 'display': 'inline-block'})
                ]),
                
                html.H3("Sample Predictions Comparison"),
                dash_table.DataTable(
                    data=pd.concat([
                        results_df_xgb[['Actual', 'Predicted_XGB', 'Probability_XGB']].head(10),
                        results_df_lgbm[['Predicted_LGBM', 'Probability_LGBM']].head(10)
                    ], axis=1).to_dict('records'),
                    columns=[
                        {'name': 'Actual', 'id': 'Actual'},
                        {'name': 'XGB Pred', 'id': 'Predicted_XGB'},
                        {'name': 'XGB Prob', 'id': 'Probability_XGB', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                        {'name': 'LGBM Pred', 'id': 'Predicted_LGBM'},
                        {'name': 'LGBM Prob', 'id': 'Probability_LGBM', 'type': 'numeric', 'format': {'specifier': '.4f'}}
                    ],
                    style_cell={'textAlign': 'center'},
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{Actual} = {Predicted_XGB}'},
                            'backgroundColor': '#d4edda',
                            'color': 'black',
                        },
                        {
                            'if': {'filter_query': '{Actual} != {Predicted_XGB}'},
                            'backgroundColor': '#f8d7da',
                            'color': 'black',
                        }
                    ]
                )
            ])
        
        elif tab == 'feature_importance':
            return html.Div([
                html.H2("Feature Importance Comparison"),
                
                html.Div([
                    html.Div([
                        dcc.Graph(
                            figure=px.bar(
                                models['xgb']['feature_importance'].head(15),
                                x='importance', y='feature',
                                orientation='h',
                                title="Top 15 XGBoost Feature Importance",
                                color='importance',
                                color_continuous_scale='Blues'
                            ).update_layout(yaxis={'categoryorder':'total ascending'})
                        )
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    
                    html.Div([
                        dcc.Graph(
                            figure=px.bar(
                                models['lgbm']['feature_importance'].head(15),
                                x='importance', y='feature',
                                orientation='h',
                                title="Top 15 LightGBM Feature Importance",
                                color='importance',
                                color_continuous_scale='Oranges'
                            ).update_layout(yaxis={'categoryorder':'total ascending'})
                        )
                    ], style={'width': '50%', 'display': 'inline-block'})
                ]),
                
                html.H3("Feature Importance Tables"),
                html.Div([
                    html.Div([
                        html.H4("XGBoost Top Features"),
                        dash_table.DataTable(
                            data=models['xgb']['feature_importance'].head(10).to_dict('records'),
                            columns=[
                                {'name': 'Feature', 'id': 'feature'},
                                {'name': 'Importance', 'id': 'importance', 'type': 'numeric', 'format': {'specifier': '.4f'}}
                            ],
                            style_cell={'textAlign': 'left'},
                            style_header={'backgroundColor': '#1f77b4', 'color': 'white'},
                            style_data={'backgroundColor': '#f0f8ff'}
                        )
                    ], style={'width': '48%', 'display': 'inline-block', 'margin-right': '4%'}),
                    
                    html.Div([
                        html.H4("LightGBM Top Features"),
                        dash_table.DataTable(
                            data=models['lgbm']['feature_importance'].head(10).to_dict('records'),
                            columns=[
                                {'name': 'Feature', 'id': 'feature'},
                                {'name': 'Importance', 'id': 'importance', 'type': 'numeric', 'format': {'specifier': '.4f'}}
                            ],
                            style_cell={'textAlign': 'left'},
                            style_header={'backgroundColor': '#ff7f0e', 'color': 'white'},
                            style_data={'backgroundColor': '#fff8f0'}
                        )
                    ], style={'width': '48%', 'display': 'inline-block'})
                ])
            ])
        
        else:
            # Default case jika tab tidak dikenali
            return html.Div([
                html.H2("Tab tidak ditemukan"),
                html.P("Silahkan pilih tab yang tersedia.")
            ])
    
    except Exception as e:
        return html.Div([
            html.H2("Error dalam memuat konten"),
            html.P(f"Terjadi error: {str(e)}"),
            html.P("Silahkan refresh halaman atau pilih tab lain.")
        ])

@app.callback(
    Output('exploration-content', 'children'),
    Input('variable-dropdown', 'value'),
    prevent_initial_call=False  # Tambahkan ini untuk mencegah error saat inisialisasi
)
def update_exploration(selected_variable):
    # Tambahkan pengecekan jika selected_variable None atau tidak valid
    if selected_variable is None or selected_variable not in df.columns:
        return html.Div([
            html.H3("Pilih variabel untuk melihat analisis"),
            html.P("Silahkan pilih variabel dari dropdown di atas untuk melihat visualisasi.")
        ])
    
    try:
        if selected_variable in df.select_dtypes(include='number').columns:
            fig1 = px.histogram(df, x=selected_variable, color='is_default', 
                               title=f"Distribusi {selected_variable} berdasarkan Status Default",
                               barmode='overlay', opacity=0.7, marginal="box")
            
            fig2 = px.violin(df, x='is_default', y=selected_variable,
                            title=f"Violin Plot {selected_variable} berdasarkan Status Default",
                            box=True)
            
            return html.Div([
                html.Div([
                    dcc.Graph(figure=fig1)
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(figure=fig2)
                ], style={'width': '50%', 'display': 'inline-block'})
            ])
        else:
            # For categorical variables - tambahkan error handling
            try:
                crosstab = pd.crosstab(df[selected_variable], df['is_default'], normalize='index') * 100
                
                fig = px.bar(
                    x=crosstab.index,
                    y=[crosstab[0], crosstab[1]],
                    title=f"Persentase Default berdasarkan {selected_variable}",
                    labels={'x': selected_variable, 'y': 'Percentage'},
                    barmode='stack'
                )
                fig.update_traces(name='Non-Default', selector=dict(name='y[0]'))
                fig.update_traces(name='Default', selector=dict(name='y[1]'))
                
                return dcc.Graph(figure=fig)
            except Exception as e:
                return html.Div([
                    html.H3("Error dalam membuat visualisasi"),
                    html.P(f"Terjadi error: {str(e)}")
                ])
    except Exception as e:
        return html.Div([
            html.H3("Error dalam memproses data"),
            html.P(f"Terjadi error: {str(e)}")
        ])

if __name__ == '__main__':
    print_progress("Dashboard ready! Starting server...")
    app.run(debug=True)