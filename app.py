import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.manifold import TSNE
# Intern module
from app_utils import data_loading as dl
from app_utils import export_plot as ep

def set_clicked():
    st.session_state.clicked = True

# Global Vars
ALLOWED_FILE_FORMATS=["csv", "xlsx", "xls", "pq", "parquet"]
ALLOWED_IMAGE_FORMAT = ["png", "jpeg", "svg"]
DEFAULT_ALLOWED_IMAGE_FORMAT = ALLOWED_IMAGE_FORMAT.index("png")

st.set_page_config(
    page_title="Anomaly Detection Features - Data Explorer",
    layout="wide"
)

# Title of the application
st.title("Anomaly Detection Features - Data Analysis")

# Tabs for univariate and multivariate analysis
tab1, tab2, tab3, tab4 = st.tabs(["Describe", "Univariate Analysis", "Multivariate Analysis", "TSNE"])


uploaded_file = st.sidebar.file_uploader("**Choose a file:**", type=ALLOWED_FILE_FORMATS, label_visibility="visible")

if uploaded_file is not None:
    # Load data
    df = dl.data_loader(uploaded_file)
    if df is not None:
        # Sidebar for column selection
        columns_to_remove = st.sidebar.multiselect("**Select columns to remove from analysis:**", df.columns)
        # Remove selected columns
        df_filtered = df.drop(columns=columns_to_remove)

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
                    # plot_univar = sns.histplot(data = df_filtered, x = variable, hue = HUE_VAR,
                    #                            multiple="layer", stat="density", common_norm=False)
                    # plot_univar.tick_params(axis='x', rotation=90)

                    # g_univar = sns.JointGrid(data=df_filtered, x=variable, hue = HUE_VAR)
                    # g_univar.plot_joint(sns.histplot, multiple="layer", stat="density", common_norm=False )
                    # g_univar.plot_marginals(sns.boxplot)

                    with st.spinner('Wait for it...'):
                        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
        
                        # assigning a graph to each ax
                        g_bp = sns.boxplot(data = df_filtered, x = variable, hue=HUE_VAR,
                                        orient="h", ax=ax_box, legend=False)
                        g_hist = sns.histplot(data=df_filtered, x=variable, hue= HUE_VAR, 
                                            stat="density",  element="step", common_norm=False, 
                                                ax=ax_hist)
                        # Remove x axis name for the boxplot
                        ax_box.set(xlabel='')

                        # g_univar.tick_params(axis='x', rotation=90)
                        # univar_figure = g_univar.get_figure()
                    
                        # Save plot memory
                        buf = ep.save_plot_as_png(f, format=IMAGE_FORMAT, dpi=1000)
            
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
                    
                    # Cat plot
                    st.subheader(f"Catplot of variables")
                    #  Bivariate plot
                    g2 = sns.catplot(data=df_filtered, x=X, y =Y, hue=HUE_VAR, kind="bar")
                    
                    st.pyplot(g2)
                    # Dis plot by deleted reason (col)
                    st.subheader(f"Disstplot of variables")
                    g3 = sns.displot(df_filtered, x=X, y=Y, col=HUE_VAR,
                                    rug=True)
                    st.pyplot(g3)
                
            with tab3_col2:
                if X is not None and Y is not None:
                    # Placeholder for download button
                    buf_g1= ep.save_plot_as_png(g1, format=IMAGE_FORMAT, dpi=1000)
                    st.download_button(label="Download KDE Plot", data=buf_g1, file_name="multivariate_kde_plot.{}".format(IMAGE_FORMAT), mime="image/{}".format(IMAGE_FORMAT))
                    # Placeholder for download button
                    buf_g2= ep.save_plot_as_png(g2, format=IMAGE_FORMAT, dpi=1000)
                    st.download_button(label="Download Catplot", data=buf_g2, file_name="multivariate_catplot.{}".format(IMAGE_FORMAT), mime="image/{}".format(IMAGE_FORMAT))
                    # Placeholder for download button
                    buf_g3= ep.save_plot_as_png(g2, format=IMAGE_FORMAT, dpi=1000)
                    st.download_button(label="Download Distlot", data=buf_g3, file_name="multivariate_distplot.{}".format(IMAGE_FORMAT), mime="image/{}".format(IMAGE_FORMAT))


            # # Display all possible bivariate combinations
            # st.write("Bivariate combinations:")
            # for combo in combinations(df_filtered.columns, 2):
            #     st.write(f"Displot of variables {combo[0]} and {combo[1]}")
            #     plt.figure(figsize=(10, 6))
            #     sns.displot(df_filtered, x=combo[0], y=combo[1], kind="kde")
            #     st.pyplot(plt)

        with tab4:
            st.header("Dimension Reduction by TSNE",  divider='rainbow')
            tab4_col1, tab4_col2 = st.columns([5, 1])
            TSNE_TARGET_VAR = None

            with tab4_col1:
                TSNE_FEATURES = st.multiselect("**Select features you want to add in TSNE:**", df_filtered.columns)
                if TSNE_FEATURES:
                    TSNE_TARGET_VAR = st.selectbox("**Select target variable:**", df_filtered.columns, index=None)
             
                if TSNE_TARGET_VAR is not None and TSNE_FEATURES:
                    # Séparer les caractéristiques et la variable cible
                    tsne_features_df = df_filtered[TSNE_FEATURES]
                    tsne_target_df = df[TSNE_TARGET_VAR]

                    # Convertir les variables catégorielles en numériques
                    # features = pd.get_dummies(features)

                    # Appliquer t-SNE
                    tsne = TSNE(n_components=2,
                                perplexity=50, 
                                random_state=0)
                    tsne_result = tsne.fit_transform(tsne_features_df)

                    # Ajouter les résultats t-SNE au DataFrame
                    df_tsne = pd.DataFrame(tsne_result, columns=['TSNE1', 'TSNE2'])
                    df_tsne[TSNE_TARGET_VAR] = tsne_target_df

                    # Visualiser les résultats t-SNE
                    st.subheader('TSNE visualization of features colored by {}'.format(TSNE_TARGET_VAR))
                    g_tsne = sns.scatterplot(x='TSNE1', y='TSNE2', hue=TSNE_TARGET_VAR, palette='viridis', data=df_tsne)
                    st.pyplot(g_tsne.get_figure())

            with tab4_col2:
                if TSNE_TARGET_VAR is not None and TSNE_FEATURES:
                    # Placeholder for download button
                    buf_TSNE= ep.save_plot_as_png(g_tsne.get_figure(), format=IMAGE_FORMAT, dpi=1000)
                    st.download_button(label="Download TSNE Plot", data=buf_TSNE, file_name="TSNE_plot.{}".format(IMAGE_FORMAT), mime="image/{}".format(IMAGE_FORMAT))