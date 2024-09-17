import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, auc
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Attention, Add, Lambda
from scikeras.wrappers import KerasRegressor
from sklearn.manifold import TSNE
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse
import shap
import streamlit.components.v1 as components

##################################################################################################################
############ NEW PART ############################################################################################
##################################################################################################################

# Function to train the model and make predictions
@st.cache_resource
def train_and_predict(model_name, data, survey_id_var=None):

    if model_name == 'IsolationForest':
        model = IsolationForest(contamination=0.1, random_state=42)
    elif model_name == 'OneClassSVM':
        model = OneClassSVM(nu=0.1, kernel='rbf')
    elif model_name == 'LOF':
        model = LocalOutlierFactor(n_neighbors=20, novelty=True)
    else:
        st.error("Unknown model")
        return None, None
    
    # Remove survey ID col if exist
    if survey_id_var is not None:
        survey_id_col_copy = data[survey_id_var]
        data = data.drop(columns = [survey_id_var])
    # Fit and predict
    data = data.copy()
    model.fit(data)
    prediction = model.predict(data)
    score =  model.decision_function(data)
    data['anomaly'] = prediction
    data['anomaly'] = (data['anomaly']  == -1).astype(int)
    data['model_score'] = score
    if survey_id_var is not None:
        data[survey_id_var] = survey_id_col_copy
    return model, data

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Function to plot SHAP visualizations

@st.cache_resource
def train_shap_explainer(model_name, _model, data, survey_id_var=None):
    if survey_id_var is not None:
        X = data.drop(columns = ['anomaly', 'model_score', survey_id_var])
    else:
        X = data.drop(columns = ['anomaly', 'model_score'])

    if model_name in ['LOF', 'OneClassSVM']:
        st.write(model_name + " model slected, SHAP computations values may take a few minutes (KernelExplainer)")
        explainer = shap.KernelExplainer(_model.decision_function, X, link="identity", feature_names=X.columns.tolist())
    else: 
        explainer = shap.TreeExplainer(_model, X)
    shap_values = explainer(X)
    # Train clustering for features correlation
    clustering = shap.utils.hclust(X)
    return explainer, shap_values, X, clustering


@st.experimental_fragment
def id_survey_shap_force_plot(survey_id_var, selected_survey, data, shap_values ) -> None:
   
    index_survey= data[data[survey_id_var] == selected_survey].index[0]
    shap_index_survey = data.index.get_loc(index_survey)
    fp_plot = shap.force_plot(shap_values[shap_index_survey])
    st_shap(fp_plot)

@st.experimental_fragment
def id_survey_shap_bar_plot(survey_id_var, selected_survey, data, shap_values, clustering, clustering_cutoff=0.5) -> None:
   
    nb_features = shap_values.data.shape[1]
    index_survey= data[data[survey_id_var] == selected_survey].index[0]
    shap_index_survey = data.index.get_loc(index_survey)
    local_bar_plot = shap.plots.bar(shap_values[shap_index_survey],  clustering=clustering, clustering_cutoff=clustering_cutoff,  max_display=nb_features)
    st.pyplot(local_bar_plot)

@st.experimental_fragment
def shap_dependence_plot(shap_dependence_feature, shap_dependence_color_feature, shap_values) -> None:
    if shap_dependence_color_feature is None:
        dp_plot = shap.plots.scatter(shap_values[:, shap_dependence_feature], color=shap_values)
    else:
        dp_plot = shap.plots.scatter(shap_values[:, shap_dependence_feature], color=shap_values[:, shap_dependence_color_feature])
    st.pyplot(dp_plot)


##################################################################################################################
############ OLD PART ############################################################################################
##################################################################################################################
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

def gaussian_parametrisation(mu, log_var):
    batch = tf.shape(mu)[0]
    dim = tf.shape(mu)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return mu + tf.exp(0.5 * log_var) * epsilon

def VAE(input_dim, latent_space=2, hidden_dim=64, dropout=0.2, lr=10e-3):
 
    if hidden_dim % 4 != 0:
        raise ValueError("hidden_dim must be a multiple of 4")
   
    # Encoder
    inputs = Input(shape=(input_dim,))
    hidden_layer = Dense(hidden_dim, activation='relu')(inputs)
    hidden_layer = Dropout(dropout)(hidden_layer)
 
    # Attention mechanism (can reduce the dim)
    query = Dense(hidden_dim, activation='relu')(hidden_layer)
    key = Dense(hidden_dim, activation='relu')(hidden_layer)
    value = Dense(hidden_dim, activation='relu')(hidden_layer)
    attention = Attention()([query, key, value])
    h = Add()([h, attention])
 
    mu = Dense(latent_space)(hidden_layer)
    log_var = Dense(latent_space)(hidden_layer)
 
    # Gaussian bit
    z = Lambda(gaussian_parametrisation, output_shape=(latent_space,), name='z')([mu, log_var])
 
    # Encoder
    encoder = Model(inputs, [mu, log_var, z], name='encoder')
    encoder.summary()
 
    # Decoder
    latent_inputs = Input(shape=(latent_space,), name='z_sampling')
    x = Dense(hidden_dim, activation='relu')(latent_inputs)
    x = Dropout(dropout)(x)
 
    # Attention
    query = Dense(hidden_dim, activation='relu')(x)
    key = Dense(hidden_dim, activation='relu')(x)
    value = Dense(hidden_dim, activation='relu')(x)
    attention2 = Attention()([query, key, value])
    x = Add()([x, attention2])
 
    outputs = Dense(input_dim, activation='sigmoid')(x)

    # Decoder
    decoder = Model(latent_inputs, outputs, name='decoder')
    # Instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')
    # loss = reconstruction loss + KL divergence loss
    rec_loss = mse(inputs, outputs)
    rec_loss *= input_dim
    kl_div_loss = tf.sqrt(K.sum(1 + log_var - K.square(mu) - K.exp(log_var)))
    vae_loss = K.mean(kl_div_loss + rec_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=Adam(learning_rate=lr))
    vae.summary()
    return vae


# def train_and_predict(X, Y, models, kf, tune_hyperparameters=False):
#     predictions_dict = {}
#     best_models = {}

#     for name, (model, params) in models.items():


#         pipeline = Pipeline([
#             ('scaler', StandardScaler()),
#             ('classifier', model)
#         ])
        
#         if tune_hyperparameters:
#             grid_search = GridSearchCV(pipeline, param_grid=params, cv=kf, scoring='accuracy', n_jobs=-1)
#             grid_search.fit(X, Y)
#             best_pipeline = grid_search.best_estimator_
#         else:
#             best_pipeline = pipeline
#             best_pipeline.fit(X, Y)
        
#         if name in ['Linear SVM']:
#             y_scores = cross_val_predict(pipeline, X, Y, cv=kf, method='decision_function')
#             y_pred = cross_val_predict(pipeline, X, Y, cv=kf, method='predict')
#         elif name in ['Isolation Forest', "One Class SVM"]:
#             y_pred = cross_val_predict(pipeline, X, Y, cv=kf)
#             # Convertir les prédictions de -1/1 à 0/1
#             y_pred = (y_pred == -1).astype(int)
#             y_scores = -pipeline.fit(X).decision_function(X)  # Scores inversés pour l'anomalie
#         else:
#             y_scores = cross_val_predict(pipeline, X, Y, cv=kf, method='predict_proba')[:, 1]
#             y_pred = cross_val_predict(pipeline, X, Y, cv=kf, method='predict')
        
        
#         predictions_dict[name] = (y_pred, y_scores)
#         best_models[name] = best_pipeline

#     # Autoencoder
#     # scaler_autoencoder = StandardScaler()
#     # X_scaled_autoencoder = scaler_autoencoder.fit_transform(X)
#     # input_dim = X_scaled_autoencoder.shape[1]

#     # autoencoder = KerasRegressor(model=create_dense_autoencoder, input_dim=input_dim, verbose=0)
#     # param_dist = {
#     #     'model__encoding_dim': [4, 8, 16, 32, 64],
#     #     'optimizer': ['adam', 'rmsprop'],
#     #     'epochs': [50, 100, 150],
#     #     'batch_size': [16, 32, 64]
#     # }

#     # if tune_hyperparameters:
#     #     random_search = RandomizedSearchCV(autoencoder, param_distributions=param_dist, n_iter=10, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
#     #     random_search.fit(X_scaled_autoencoder, X_scaled_autoencoder)
#     #     best_autoencoder = random_search.best_estimator_
#     # else:
#     #     best_autoencoder = autoencoder
#     #     best_autoencoder.fit(X_scaled_autoencoder, X_scaled_autoencoder)
    
#     # reconstructed = best_autoencoder.predict(X_scaled_autoencoder)
#     # reconstruction_errors = np.mean((X_scaled_autoencoder - reconstructed) ** 2, axis=1)
#     # threshold = np.percentile(reconstruction_errors, 95)
#     # y_pred_autoencoder = (reconstruction_errors > threshold).astype(int)
#     # predictions_dict['Autoencoder'] = (y_pred_autoencoder, reconstruction_errors)

#     return predictions_dict, best_models

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

# def plot_tsne(X, Y, predictions_dict):
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     tsne = TSNE(n_components=2, perplexity=50, n_iter=300)
#     tsne_result = tsne.fit_transform(X_scaled)
    
#     df_tsne = pd.DataFrame(tsne_result, columns=['TSNE1', 'TSNE2'])
#     df_tsne["Y"] = Y

#     for model_name, (y_pred, _) in predictions_dict.items():
#         df_tsne[model_name] = y_pred

#     num_models = len(predictions_dict)
#     n_cols = 3
#     n_rows = (num_models + n_cols - 1) // n_cols

#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
#     axes = axes.flatten()

#     for ax, model_name in zip(axes, predictions_dict.keys()):
#         sns.scatterplot(
#             data=df_tsne,
#             x='TSNE1',
#             y='TSNE2',
#             hue=model_name,
#             style='Y',
#             markers={0: ',', 1: 'X'},
#             ax=ax,
#             alpha=0.6,
#             legend='brief'
#         )
#         ax.set_title(f't-SNE Visualization - {model_name}')
#         ax.set_xlabel('TSNE1')
#         ax.set_ylabel('TSNE2')
#         ax.legend(loc='upper right', fontsize='small')

#     for i in range(len(predictions_dict), len(axes)):
#         fig.delaxes(axes[i])

#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     return fig