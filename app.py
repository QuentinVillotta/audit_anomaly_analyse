import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import KFold
import pickle
import shap
# Intern module
from app_utils import data_loading as dl
from app_utils import plot_tools as pt
from app_utils import modelisation_interpretation as mi
from app_utils import data_comparaison as dc

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

def set_clicked():
    st.session_state.clicked = True

# Global Vars
ALLOWED_FILE_FORMATS=["csv", "xlsx", "xls", "pq", "parquet"]
ALLOWED_IMAGE_FORMAT = ["png", "jpeg", "svg"]
DEFAULT_ALLOWED_IMAGE_FORMAT = ALLOWED_IMAGE_FORMAT.index("png")
THRESHOLD_QUANTITATIVE_TYPE = 20  

# Modelisation parameters

models = {
    'IsolationForest': (IsolationForest(contamination=0.1, random_state=42), {'classifier__n_estimators': [50, 100, 200, 300], 'classifier__max_samples': ['auto', 0.5, 0.75]}),
    'OneClassSVM': (OneClassSVM(nu=0.1, kernel='rbf'), {'classifier__gamma': ['scale', 'auto'], 'classifier__nu': [0.05, 0.1, 0.2]}),
    'LOF': (LocalOutlierFactor(n_neighbors=20, novelty=True), {'classifier__n_neighbors': [5, 10, 20, 30],'classifier__metric': ['euclidean', 'manhattan']})  
}

# models = {
#     'KNN': (KNeighborsClassifier(), {'classifier__n_neighbors': [3, 5, 7, 9, 11], 'classifier__weights': ['uniform', 'distance'], 'classifier__metric': ['euclidean', 'manhattan']}),
#     'Linear SVM': (LinearSVC(dual=False), {'classifier__C': [0.01, 0.1, 1, 10, 100], 'classifier__loss': ['hinge', 'squared_hinge']}),
#     'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {'classifier__n_estimators': [50, 100, 200, 300], 'classifier__max_depth': [3, 5, 7], 'classifier__learning_rate': [0, 0.01, 0.1, 0.2]}),
#     'Decision Tree': (DecisionTreeClassifier(), {'classifier__max_depth': [None, 10, 20, 30], 'classifier__min_samples_split': [2, 5, 10]}),
#     'Random Forest': (RandomForestClassifier(), {'classifier__n_estimators': [50, 100, 200, 300], 'classifier__max_depth': [None, 10, 20], 'classifier__min_samples_split': [2, 5, 10]}),
#     'Isolation Forest': (IsolationForest(contamination=0.1, random_state=42), {'classifier__n_estimators': [50, 100, 200, 300], 'classifier__max_samples': ['auto', 0.5, 0.75]}),
#     'One Class SVM': (OneClassSVM(nu=0.1, kernel='rbf'), {'classifier__gamma': ['scale', 'auto'], 'classifier__nu': [0.05, 0.1, 0.2]}),
# }

kf = KFold(n_splits=5, shuffle=True, random_state=42)

st.set_page_config(
    page_title="Anomaly Detection Features - Data Explorer",
    layout="wide"
)

# Title of the application
st.title("Anomaly Detection Features - Data Analysis")

# Tabs for univariate and multivariate analysis
tab1, tab2, tab3, tab4, tab5, tab6 , tab7 = st.tabs(["Describe", "Univariate Analysis", "Multivariate Analysis", "Modelisation & Interpretation", "TSNE", "Features Dataset Comparaison", "Dev"])


uploaded_file = st.sidebar.file_uploader("**Choose a file:**", type=ALLOWED_FILE_FORMATS, label_visibility="visible")

if uploaded_file is not None:
    # Load data
    df = dl.data_loader(uploaded_file)
    if df is not None:
        # Sidebar for column selection
        columns_to_remove = st.sidebar.multiselect("**Select columns to remove from analysis:**", df.columns)
        # Remove selected columns
        df_filtered = df.drop(columns=columns_to_remove)
        variable_types = dl.classify_variable_types(df_filtered, THRESHOLD_QUANTITATIVE_TYPE)


        # HUE Var
        HUE_VAR = st.sidebar.selectbox("**Select a variable for hue (color grouping):**", df_filtered.columns, index=None)

        # Format plot
        IMAGE_FORMAT = st.sidebar.selectbox('**Image Format to Export:**', ALLOWED_IMAGE_FORMAT, index=DEFAULT_ALLOWED_IMAGE_FORMAT)
        

        with tab1:
            # Display the first few rows of the filtered DataFrame
            st.header('Descriptive Statistics', divider='rainbow') 
            st.dataframe(df_filtered.describe(include="all"))

        with tab2:

            # Univariate Analysis
            st.header("Univariate Analysis",  divider='rainbow')
            
            tab1_col1, tab2_col2 = st.columns([5, 1])
        
            with tab1_col1:
                # Select a variable for univariate analysis
                variable = st.selectbox("Choose a variable for distribution analysis:", df_filtered.columns, index=None)

                # Plot
                if variable is not None:

                    with st.spinner('Wait for it...'):
                        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
        
                        # assigning a graph to each ax
                        if variable_types[variable] == 'discrete':
                            # Count plot
                            g_bp = sns.boxplot(data = df_filtered, x = variable, hue=HUE_VAR,
                            orient="h", ax=ax_box, legend=False)
                            g_hist = sns.countplot(data=df_filtered, x=variable, hue= HUE_VAR, 
                                                  ax=ax_hist)
                            # Remove x axis name for the boxplot
                            ax_box.set(xlabel='')

                        elif variable_types[variable] == 'non-numeric':  # New case for non-numeric variables
                            # Count plot for non-numeric variables
                            g_hist = sns.countplot(data=df_filtered, x=variable, hue=HUE_VAR, ax=ax_hist)
                            ax_hist.set_xticklabels(ax_hist.get_xticklabels(), rotation=90)
                            # Set title and remove boxplot since it doesn't make sense for non-numeric
                            # ax_hist.set_title(f"Count plot of {variable}")
                            ax_box.axis('off')  # Hide the boxplot
                        else:
                            # Histogram
                            g_bp = sns.boxplot(data = df_filtered, x = variable, hue=HUE_VAR,
                            orient="h", ax=ax_box, legend=False)
                            g_hist = sns.histplot(data=df_filtered, x=variable, hue= HUE_VAR, 
                                                  stat="density",  element="step", common_norm=False, 
                                                  ax=ax_hist)
                            # Remove x axis name for the boxplot
                            ax_box.set(xlabel='')
                    
                        # Save plot memory
                        buf = pt.save_plot_as_png(f, format=IMAGE_FORMAT, dpi=1000)
            
            with tab2_col2:
                if variable is not None:
                    # Placeholder for download button
                    st.download_button(label="Download Plot", data=buf, file_name="univariate_plot.{}".format(IMAGE_FORMAT), mime="image/{}".format(IMAGE_FORMAT))

            if variable is not None:
                # Display a histogram of the variable's distribution
                # st.header(f"Histogram of :  {variable}")
                st.pyplot(f)



        with tab3:
            # Multivariate Analysis
            st.header("Multivariate Analysis",  divider='rainbow')
            
            tab3_col1, tab3_col2 = st.columns([5, 1])

            with tab3_col1:
                # Select variables for bivariate analysis
                X = st.selectbox("Choose the first variable for bivariate analysis:", df_filtered.columns, key="X", index = None)
                Y = st.selectbox("Choose the second variable for bivariate analysis:", df_filtered.columns, key="Y", index = None)

                # Display a displot for bivariate analysis
                if X is not None and Y is not None:
                    st.subheader(f"Kernel Density Bivariate plot")
                    #  Bivariate plot
                    g1 = sns.JointGrid(data=df_filtered, x=X, y=Y,  hue=HUE_VAR)
                    g1.plot_joint(sns.kdeplot, fill=False, alpha=0.4, common_norm=True)
                    g1.plot_joint(sns.rugplot, height=-.02, clip_on=False, alpha=.5 )
                    g1.plot_marginals(sns.boxplot)
                    st.pyplot(g1)
                    
             
                    #  Bivariate plot
                    st.subheader(f"Catplot of variables")
                    if variable_types[X] == 'discrete':
                        # Cat plot
                        g2 = sns.catplot(data=df_filtered, x=X, y =Y, hue=HUE_VAR, kind="bar",
                                         estimator='mean', errorbar=('ci', 95))
                    else:
                        num_bins = pt.freedman_diaconis_rule(df_filtered[X])
                        df_filtered['quantitative_var_binned'] = pd.cut(df_filtered[X], bins=num_bins)
                        g2 = sns.catplot(data=df_filtered, x="quantitative_var_binned", y =Y, hue=HUE_VAR, kind="bar",
                                         estimator='mean', errorbar=('ci', 95))
                        g2.set(xlabel=X)
                        g2.tick_params(axis='x', rotation=90)
                    
                    st.pyplot(g2)
                    # Dis plot by deleted reason (col)
                    st.subheader(f"Distplot of variables")
                    g3 = sns.displot(df_filtered, x=X, y=Y, col=HUE_VAR,
                                     rug=True)
                    st.pyplot(g3)
                
            with tab3_col2:
                if X is not None and Y is not None:
                    # Placeholder for download button
                    buf_g1= pt.save_plot_as_png(g1, format=IMAGE_FORMAT, dpi=1000)
                    st.download_button(label="Download KDE Plot", data=buf_g1, file_name="multivariate_kde_plot.{}".format(IMAGE_FORMAT), mime="image/{}".format(IMAGE_FORMAT))
                    # Placeholder for download button
                    buf_g2= pt.save_plot_as_png(g2, format=IMAGE_FORMAT, dpi=1000)
                    st.download_button(label="Download Catplot", data=buf_g2, file_name="multivariate_catplot.{}".format(IMAGE_FORMAT), mime="image/{}".format(IMAGE_FORMAT))
                    # Placeholder for download button
                    buf_g3= pt.save_plot_as_png(g3, format=IMAGE_FORMAT, dpi=1000)
                    st.download_button(label="Download Distlot", data=buf_g3, file_name="multivariate_distplot.{}".format(IMAGE_FORMAT), mime="image/{}".format(IMAGE_FORMAT))


        with tab4:

            st.title("Model Training and SHAP interpretation")

            # Choisir entre l'entraînement d'un modèle ou le chargement depuis un fichier pickle
            mode = st.radio("Modelisation option:", ["Train a new model", "Load modelisation from file"])
            if mode == "Load modelisation from file":
                # Option pour charger un modèle et un explainer déjà calculé depuis un fichier pickle
                uploaded_modelisation = st.file_uploader("Upload your saved modelisation", type="pkl")

                if uploaded_modelisation is not None:
                    loaded_modelisation = pickle.load(uploaded_modelisation)

                    with st.expander("Model Parameters"):
                        model_name = st.selectbox('Models availables', loaded_modelisation.keys(), index = None)
                        if model_name is not None: 
                            modelisation_selected = loaded_modelisation[model_name]
                            model = modelisation_selected['model']
                            explainer = modelisation_selected['explainer']
                            shap_values = modelisation_selected['shap_values']

                            # Reorder by features
                            shap_values_array = np.array(shap_values.values)
                            feature_names = np.array(shap_values.feature_names)
                            sorted_indices = np.argsort(feature_names)
        
                            # Mettre à jour les valeurs et noms dans shap_values en fonction du tri
                            shap_values.feature_names = feature_names[sorted_indices].tolist()
                            shap_values.values = shap_values_array[:, sorted_indices]

                            features = modelisation_selected['features']
                            survey_id_var = 'audit_id'
                            shap_data = features.drop(survey_id_var, axis=1)
                            clustering = shap.utils.hclust(shap_data)
                            # Get prediction and score
                            y_pred = model.predict(shap_data)
                            features['anomaly'] = (y_pred == -1).astype(int)
                            features['model_score'] = model.decision_function(shap_data)
                                                
                            # Select variables for bivariate analysis
                            st.write('Features used for modeling:')
                            st.write(shap_data.columns.tolist())
            

                if uploaded_modelisation is not None and model_name is not None:
                    # Continuer avec l'interprétation SHAP comme d'habitude
                    st.write("## SHAP Interpretation")
                    sub_tab1, sub_tab2 = st.tabs(["Global Interpretation", "Local Interpretation"])

                    with sub_tab1:
                        sub_sub1_tab1, sub_sub1_tab2, sub_sub1_tab3 = st.tabs(["SHAP Feature Importance", "SHAP Summary Plot", "SHAP Dependence Plot"])
                        with sub_sub1_tab1:
                            nb_features = len(shap_data.columns)

                            fi_plot = shap.plots.bar(shap_values, clustering=clustering, max_display=nb_features,
                                                     order=shap_values.feature_names)
                            st.pyplot(fi_plot)
                        with sub_sub1_tab2:
                            sp_plot = shap.summary_plot(shap_values, shap_data)
                            st.pyplot(sp_plot)
                        with sub_sub1_tab3:
                            col1, col2 = st.columns(2)
                            with col1:
                                shap_dependence_feature = st.selectbox('Choose Feature', shap_data.columns, index=None)
                            with col2:
                                shap_dependence_color_feature = st.selectbox('Choose Interaction Feature (color)', shap_data.columns, index=None)
                            if shap_dependence_feature is not None:
                                mi.shap_dependence_plot(shap_dependence_feature, shap_dependence_color_feature, shap_values)

                    with sub_tab2:
                        # Sélection du survey pour l'interprétation locale
                        survey_id = features[survey_id_var]
                        selected_survey = st.selectbox("Select an survey ID", survey_id)

                        if survey_id_var is not None:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Model prediction")
                                st.write(features.loc[features[survey_id_var] == selected_survey, 'anomaly'])
                            with col2:
                                st.subheader("Model Score")
                                st.write(features.loc[features[survey_id_var] == selected_survey, 'model_score'])

                            sub_sub2_tab1, sub_sub2_tab2 = st.tabs(["SHAP Feature Importance", "SHAP Force Plot"])
                            with sub_sub2_tab1:
                                mi.id_survey_shap_bar_plot(survey_id_var, 
                                                           selected_survey, 
                                                           features, 
                                                           shap_values, 
                                                           clustering, 
                                                           clustering_cutoff=0.5)
                            with sub_sub2_tab2:
                                mi.id_survey_shap_force_plot(survey_id_var=survey_id_var, 
                                                             selected_survey=selected_survey, 
                                                             data=features,
                                                             shap_values=shap_values)

           
            elif mode == "Train a new model": 
                # Model selection
                with st.expander("Model Parameters"):
                    model_name = st.selectbox('Choose the anomaly detection model', ['IsolationForest', 'OneClassSVM', 'LOF'], index = None)
                    # Select variables for bivariate analysis
                    list_X_var = st.multiselect("**Select features:**", df_filtered.columns, key= "X_model")
                    survey_id_var = st.selectbox("**Choose Survey ID variable name  (if not available select 'None'):**", df_filtered.columns, index = None)
                    train_model_btn = st.button("Train Model", key="train_model")
                    
                if train_model_btn or st.session_state.model_trained:
                    st.write("Training models... This may take a few minutes.")
                    if list_X_var is not None and model_name is not None:
                        X = df_filtered[list_X_var]
                        if survey_id_var is not None:
                            X[survey_id_var] = df_filtered[survey_id_var]
                        # Train model and SHAP explainer
                        model, data = mi.train_and_predict(model_name=model_name, data=X, survey_id_var=survey_id_var)
                        explainer, shap_values, shap_data, clustering = mi.train_shap_explainer(model_name=model_name, _model=model , data=data, survey_id_var=survey_id_var)
                       
                        # Reorder by features
                        shap_values_array = np.array(shap_values.values)
                        feature_names = np.array(shap_values.feature_names)
                        sorted_indices = np.argsort(feature_names)
    
                        # Mettre à jour les valeurs et noms dans shap_values en fonction du tri
                        shap_values.feature_names = feature_names[sorted_indices].tolist()
                        shap_values.values = shap_values_array[:, sorted_indices]

                        st.write("## SHAP Interpretation")
                        sub_tab1, sub_tab2 = st.tabs(["Global Interpretation",  "Local Interpretation"])
                        with sub_tab1:
                            sub_sub1_tab1, sub_sub1_tab2, sub_sub1_tab3 = st.tabs(["SHAP Feature Importance",  "SHAP Summary Plot", "SHAP Dependence Plot"])
                            with sub_sub1_tab1:
                                nb_features = len(shap_data.columns)
                      
                                fi_plot = shap.plots.bar(shap_values, clustering=clustering,  max_display=nb_features,
                                                         order=shap_values.feature_names)
                                st.pyplot(fi_plot)
                            with sub_sub1_tab2:
                                sp_plot = shap.summary_plot(shap_values, shap_data)
                                st.pyplot(sp_plot)
                            with sub_sub1_tab3:
                                col1, col2 = st.columns(2)
                                with col1:
                                    shap_dependence_feature = st.selectbox('Choose Feature', shap_data.columns, index = None)
                                with col2:
                                    shap_dependence_color_feature = st.selectbox('Choose Interaction Feature (color)', shap_data.columns, index = None)
                                if shap_dependence_feature is not None:
                                    mi.shap_dependence_plot(shap_dependence_feature, shap_dependence_color_feature, shap_values)

                        with sub_tab2:
                            # Selectbox survey
                            survey_id = data[survey_id_var]
                            selected_survey = st.selectbox("Select an survey ID", survey_id)
                            # Local interpretation
                            if survey_id_var is not None:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.subheader("Model prediction")
                                    data.loc[data["audit_id"] == selected_survey , 'anomaly']
                                with col2:
                                    st.subheader("Model Score")
                                    data.loc[data["audit_id"] == selected_survey , 'model_score']
                                # Sub tab - 
                                sub_sub2_tab1, sub_sub2_tab2 = st.tabs(["SHAP Feature Importance",  "SHAP Force Plot"])
                                with sub_sub2_tab1:
                                    mi.id_survey_shap_bar_plot(survey_id_var,
                                                               selected_survey,
                                                                data,
                                                                shap_values, 
                                                                clustering, 
                                                                clustering_cutoff=0.5)
                                with sub_sub2_tab2:
                                    mi.id_survey_shap_force_plot(survey_id_var=survey_id_var,
                                                                 selected_survey=selected_survey, 
                                                                 data=data,
                                                                 shap_values=shap_values)
                            
                        st.session_state.model_trained = True
        with tab5:
            st.header("Dimension Reduction by TSNE",  divider='rainbow')
            tab5_col1, tab5_col2 = st.columns([5, 1])
            TSNE_TARGET_VAR = None

            with tab5_col1:
                TSNE_FEATURES = st.multiselect("**Select features you want to add in TSNE:**", df_filtered.columns)
                if TSNE_FEATURES:
                    TSNE_TARGET_VAR = st.selectbox("**Select target variable:**", df_filtered.columns, index=None)
             
                if TSNE_TARGET_VAR is not None and TSNE_FEATURES:
                    # Séparer les caractéristiques et la variable cible
                    tsne_features_df = df_filtered[TSNE_FEATURES]
                    tsne_target_df = df[TSNE_TARGET_VAR]

                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(tsne_features_df)
                    # Appliquer t-SNE
                    tsne = TSNE(n_components=2,
                                perplexity=50, 
                                random_state=0)
                    tsne_result = tsne.fit_transform(X_scaled)

                    # Ajouter les résultats t-SNE au DataFrame
                    df_tsne = pd.DataFrame(tsne_result, columns=['TSNE1', 'TSNE2'])
                    df_tsne[TSNE_TARGET_VAR] = tsne_target_df

                    # Visualiser les résultats t-SNE
                    st.subheader('TSNE visualization of features colored by {}'.format(TSNE_TARGET_VAR))
                    g_tsne = sns.scatterplot(x='TSNE1', y='TSNE2', hue=TSNE_TARGET_VAR, palette='viridis', data=df_tsne)
                    st.pyplot(g_tsne.get_figure())

            with tab5_col2:
                if TSNE_TARGET_VAR is not None and TSNE_FEATURES:
                    # Placeholder for download button
                    buf_TSNE= pt.save_plot_as_png(g_tsne.get_figure(), format=IMAGE_FORMAT, dpi=1000)
                    st.download_button(label="Download TSNE Plot", data=buf_TSNE, file_name="TSNE_plot.{}".format(IMAGE_FORMAT), mime="image/{}".format(IMAGE_FORMAT))
else:      
    with tab6:
        st.header("Data Comparison")

        st.markdown("""
        1. Upload multiple datasets (they should have the same indicators).
        2. Descriptive statistics (mean, standard deviation, etc.) will be calculated for each dataset.
        3. You can compare the differences between the datasets.
        """)

         # Initialize session state if not already present
        if 'datasets' not in st.session_state:
            st.session_state.datasets = {}
    
        # Upload CSV files
        uploaded_files = st.file_uploader("Upload your CSV files", accept_multiple_files=True,
                                          type=ALLOWED_FILE_FORMATS, label_visibility="visible")
        
        if uploaded_files:

            # Read files and store DataFrames
            st.session_state.datasets = {file.name: dl.data_loader(file) for file in uploaded_files}
            st.markdown("### Normalize Data")
            standard_scaler_on = st.checkbox("Apply Standard Scaler to Numeric Variables", value=False)

            # Apply standard scaling if button is pressed
            if standard_scaler_on:
                st.session_state.datasets, problematic_columns = dc.apply_standard_scaler(st.session_state.datasets)
                # Display warning messages for columns that were excluded
                if problematic_columns:
                    for dataset_name, cols in problematic_columns.items():
                        st.warning(f"In dataset '{dataset_name}', the following columns were excluded from scaling due to containing infinity or very large values: {', '.join(cols)}")
                
                st.success("Standard Scaling applied to all datasets.")

            # Calculate and display statistics for each dataset
            if st.session_state.datasets:
                stats_list = [(name, dc.calculate_statistics(df)) for name, df in st.session_state.datasets.items()]
            
            # Compare statistics across datasets
            if len(stats_list) > 1:
                st.markdown("## Comparison Distribution Plot")
                # Comparaison distribution plot
                all_numeric_cols = set(col for df in st.session_state.datasets.values() for col in df.select_dtypes(include=['number']).columns)
                selected_indicator = st.selectbox("Select an indicator to compare distributions:", options=all_numeric_cols)
                
                if selected_indicator:
                    dc.plot_distributions(st.session_state.datasets, selected_indicator)

                st.markdown("## Statistics Comparison")
                comparison = pd.concat([stats for _, stats in stats_list], keys=[name for name, _ in stats_list], names=['Dataset', 'Indicator'])
                comparison = comparison.sort_values(by=['Indicator'])
                st.write(comparison)

                # Display differences between the datasets
                st.markdown("## Differences Between Datasets")
                df_agg_comparaison = pd.DataFrame()
                for col in comparison.columns:
                    if col not in ['count']:
                        diff = comparison[col].groupby('Indicator').apply(lambda x: x.max() - x.min())
                        df_agg_comparaison[ col + " difference"] = diff
                st.write(df_agg_comparaison)
            else:
                st.markdown("## Statistics Dataset")
                # Calculate and display statistics for each dataset
                for name, stats in stats_list:
                    st.write(f"**Dataset: {name}:**")
                    st.write(stats)
# TAB DEV 
    with tab7:
        sub_tab1, sub_tab2 = st.tabs(["Sub TAB 1",  "SUB Tab 2"])
        with sub_tab1:
            st.header("SUB TAB 1")
        with sub_tab2:
            st.header("SUB TAB 2")

        # with st.expander("Sous-Onglet 1"):
        #         st.write("Contenu du Sous-Onglet 1")

        # with st.expander("Sous-Onglet 2"):
        #         st.write("Contenu du Sous-Onglet 2")

        # sous_tab = st.radio("Sélectionnez un Sous-Onglet", ["Sous-Onglet 1", "Sous-Onglet 2"])

        # if sous_tab == "Sous-Onglet 1":
        #     st.write("Contenu du Sous-Onglet 1")
        # elif sous_tab == "Sous-Onglet 2":
        #     st.write("Contenu du Sous-Onglet 2")
        
