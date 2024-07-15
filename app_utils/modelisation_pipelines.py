import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, auc
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from scikeras.wrappers import KerasRegressor
from sklearn.manifold import TSNE



def create_dense_autoencoder(input_dim, encoding_dim=8, optimizer="adam"):
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(32, activation='relu')(input_layer)
    encoder = Dense(16, activation='relu')(encoder)
    encoder = Dense(encoding_dim, activation='relu')(encoder)
    decoder = Dense(16, activation='relu')(encoder)
    decoder = Dense(32, activation='relu')(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
    return autoencoder

def train_and_predict(X, Y, models, kf, tune_hyperparameters=False):
    predictions_dict = {}
    best_models = {}

    for name, (model, params) in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        if tune_hyperparameters:
            grid_search = GridSearchCV(pipeline, param_grid=params, cv=kf, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X, Y)
            best_pipeline = grid_search.best_estimator_
        else:
            best_pipeline = pipeline
            best_pipeline.fit(X, Y)
        
        # y_pred = cross_val_predict(best_pipeline, X, Y, cv=kf, method='predict')
        
        # if hasattr(best_pipeline.named_steps['classifier'], "decision_function"):
        #     y_scores = cross_val_predict(best_pipeline, X, Y, cv=kf, method='decision_function')
        # elif hasattr(best_pipeline.named_steps['classifier'], "predict_proba"):
        #     y_scores = cross_val_predict(best_pipeline, X, Y, cv=kf, method='predict_proba')[:, 1]
        # else:
        #     y_scores = y_pred
        if name in ['Linear SVM']:
            y_scores = cross_val_predict(pipeline, X, Y, cv=kf, method='decision_function')
            y_pred = cross_val_predict(pipeline, X, Y, cv=kf, method='predict')
        elif name in ['Isolation Forest', "One Class SVM"]:
            y_pred = cross_val_predict(pipeline, X, Y, cv=kf)
            # Convertir les prédictions de -1/1 à 0/1
            y_pred = (y_pred == -1).astype(int)
            y_scores = -pipeline.fit(X).decision_function(X)  # Scores inversés pour l'anomalie
        else:
            y_scores = cross_val_predict(pipeline, X, Y, cv=kf, method='predict_proba')[:, 1]
            y_pred = cross_val_predict(pipeline, X, Y, cv=kf, method='predict')
        
        
        predictions_dict[name] = (y_pred, y_scores)
        best_models[name] = best_pipeline

    # Autoencoder
    # scaler_autoencoder = StandardScaler()
    # X_scaled_autoencoder = scaler_autoencoder.fit_transform(X)
    # input_dim = X_scaled_autoencoder.shape[1]

    # autoencoder = KerasRegressor(model=create_dense_autoencoder, input_dim=input_dim, verbose=0)
    # param_dist = {
    #     'model__encoding_dim': [4, 8, 16, 32, 64],
    #     'optimizer': ['adam', 'rmsprop'],
    #     'epochs': [50, 100, 150],
    #     'batch_size': [16, 32, 64]
    # }

    # if tune_hyperparameters:
    #     random_search = RandomizedSearchCV(autoencoder, param_distributions=param_dist, n_iter=10, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
    #     random_search.fit(X_scaled_autoencoder, X_scaled_autoencoder)
    #     best_autoencoder = random_search.best_estimator_
    # else:
    #     best_autoencoder = autoencoder
    #     best_autoencoder.fit(X_scaled_autoencoder, X_scaled_autoencoder)
    
    # reconstructed = best_autoencoder.predict(X_scaled_autoencoder)
    # reconstruction_errors = np.mean((X_scaled_autoencoder - reconstructed) ** 2, axis=1)
    # threshold = np.percentile(reconstruction_errors, 95)
    # y_pred_autoencoder = (reconstruction_errors > threshold).astype(int)
    # predictions_dict['Autoencoder'] = (y_pred_autoencoder, reconstruction_errors)

    return predictions_dict, best_models

def plot_evaluation_metrics(predictions_dict, Y):
    num_models = len(predictions_dict)
    fig, axes = plt.subplots(num_models, 3, figsize=(18, num_models * 5))

    for idx, (name, (y_pred, y_scores)) in enumerate(predictions_dict.items()):
        precision, recall, _ = precision_recall_curve(Y, y_scores)
        fpr, tpr, _ = roc_curve(Y, y_scores)
        conf_mat = confusion_matrix(Y, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall Curve
        axes[idx, 0].plot(recall, precision, marker='.')
        axes[idx, 0].set_title(f'{name} - Precision-Recall Curve')
        axes[idx, 0].set_xlabel('Recall')
        axes[idx, 0].set_ylabel('Precision')
        
        # ROC Curve
        axes[idx, 1].plot(fpr, tpr, marker='.', label=f'AUC = {roc_auc:.2f}')
        axes[idx, 1].set_title(f'{name} - ROC Curve')
        axes[idx, 1].set_xlabel('False Positive Rate')
        axes[idx, 1].set_ylabel('True Positive Rate')
        axes[idx, 1].legend()
        
        # Confusion Matrix
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', ax=axes[idx, 2])
        axes[idx, 2].set_title(f'{name} - Confusion Matrix')
        axes[idx, 2].set_xlabel('Predicted')
        axes[idx, 2].set_ylabel('Actual')
    
    plt.tight_layout()
    return fig

def plot_tsne(X, Y, predictions_dict):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=50, n_iter=300)
    tsne_result = tsne.fit_transform(X_scaled)
    
    df_tsne = pd.DataFrame(tsne_result, columns=['TSNE1', 'TSNE2'])
    df_tsne["Y"] = Y

    for model_name, (y_pred, _) in predictions_dict.items():
        df_tsne[model_name] = y_pred

    num_models = len(predictions_dict)
    n_cols = 3
    n_rows = (num_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for ax, model_name in zip(axes, predictions_dict.keys()):
        sns.scatterplot(
            data=df_tsne,
            x='TSNE1',
            y='TSNE2',
            hue=model_name,
            style='Y',
            markers={0: ',', 1: 'D'},
            ax=ax,
            alpha=0.6,
            legend='brief'
        )
        ax.set_title(f't-SNE Visualization - {model_name}')
        ax.set_xlabel('TSNE1')
        ax.set_ylabel('TSNE2')
        ax.legend(loc='upper right', fontsize='small')

    for i in range(len(predictions_dict), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# Application Streamlit
# def main():
#     st.title("Model Training and Visualization")

#     # Load data
#     PROJECT_SHORT = "MSNA_2023_SYR"
#     df_export_path = f"data/05_model_input/{PROJECT_SHORT}/features_model.csv"
#     df = pd.read_csv(df_export_path)
#     X = df.drop(NON_FEATURES, axis=1)
#     Y = df["anomaly_label"]

#     # Model parameters
#     models = {
#         'KNN': (KNeighborsClassifier(), {'classifier__n_neighbors': [3, 5, 7, 9, 11], 'classifier__weights': ['uniform', 'distance'], 'classifier__metric': ['euclidean', 'manhattan']}),
#         'Linear SVM': (LinearSVC(dual=False), {'classifier__C': [0.01, 0.1, 1, 10, 100], 'classifier__loss': ['hinge', 'squared_hinge']}),
#         'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {'classifier__n_estimators': [50, 100, 200, 300], 'classifier__max_depth': [3, 5, 7], 'classifier__learning_rate': [0, 0.01, 0.1, 0.2]}),
#         'Decision Tree': (DecisionTreeClassifier(), {'classifier__max_depth': [None, 10, 20, 30], 'classifier__min_samples_split': [2, 5, 10]}),
#         'Random Forest': (RandomForestClassifier(), {'classifier__n_estimators': [50, 100, 200, 300], 'classifier__max_depth': [None, 10, 20], 'classifier__min_samples_split': [2, 5, 10]}),
#         'Isolation Forest': (IsolationForest(contamination=0.1, random_state=42), {'classifier__n_estimators': [50, 100, 200, 300], 'classifier__max_samples': ['auto', 0.5, 0.75]}),
#         'One Class SVM': (OneClassSVM(nu=0.1, kernel='rbf'), {'classifier__gamma': ['scale', 'auto'], 'classifier__nu': [0.05, 0.1, 0.2]})
#     }

#     kf = KFold(n_splits=5, shuffle=True, random_state=42)

#     st.sidebar.title("Settings")
#     tune_hyperparameters = st.sidebar.checkbox("Tune Hyperparameters", value=True)

#     if st.sidebar.button("Train Models"):
#         st.write("Training models... This may take a few minutes.")
#         predictions_dict, best_models = train_and_predict(X, Y, models, kf, tune_hyperparameters)

#         st.write("### Model Evaluation Metrics")
#         evaluation_fig = plot_evaluation_metrics(predictions_dict, Y)
#         st.pyplot(evaluation_fig)

#         st.write("### t-SNE Visualization")
#         tsne_fig = plot_tsne(X, Y, predictions_dict)
#         st.pyplot(tsne_fig)

#         st.write("### Best Models")
#         for name, model in best_models.items():
#             st.write(f"**{name}:**")
#             st.write(model)

# if __name__ == "__main__":
#     main()
