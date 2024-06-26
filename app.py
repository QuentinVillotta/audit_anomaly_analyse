import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
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
tab1, tab2, tab3 = st.tabs(["Describe", "Univariate Analysis", "Multivariate Analysis"])


uploaded_file = st.sidebar.file_uploader("**Choose a file:**", type=ALLOWED_FILE_FORMATS, label_visibility="visible")

if uploaded_file is not None:
    # Load data
    df = dl.data_loader(uploaded_file)

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
        st.header("Univariate Analysis")
        
        col1, col2 = st.columns([5, 1])
    
        with col1:
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
        
        with col2:
            if variable is not None:
                # Placeholder for download button
                st.download_button(label="Download Plot", data=buf, file_name="univariate_plot.{}".format(IMAGE_FORMAT), mime="image/{}".format(IMAGE_FORMAT))

        if variable is not None:
            # Display a histogram of the variable's distribution
            # st.header(f"Histogram of :  {variable}")
            st.pyplot(f)



    with tab3:
        # Multivariate Analysis
        st.header("Multivariate Analysis")
        
        # Select variables for bivariate analysis
        X = st.selectbox("Choose the first variable for bivariate analysis:", df_filtered.columns, key="X", index = None)
        Y = st.selectbox("Choose the second variable for bivariate analysis:", df_filtered.columns, key="Y", index = None)

        # Display a displot for bivariate analysis
        if X is not None and Y is not None:
            st.subheader(f"Kernel Density Bivariate plot of variables {X} and {Y}")
            #  Bivariate plot
            g1 = sns.JointGrid(data=df_filtered, x=X, y=Y,  hue=HUE_VAR)
            g1.plot_joint(sns.kdeplot, fill=False, alpha=0.4, common_norm=True)
            g1.plot_joint(sns.rugplot, height=-.02, clip_on=False, alpha=.5 )
            g1.plot_marginals(sns.boxplot)
            st.pyplot(g1)
            
            # Cat plot
            st.subheader(f"Catplot of variables {X} and {Y}")
            #  Bivariate plot
            g2 = sns.catplot(data=df_filtered, x=X, y =Y, hue=HUE_VAR, kind="bar")
            st.pyplot(g2)
            # Dis plot by deleted reason (col)
            st.subheader(f"Disstplot of variables {X} and {Y}")
            g3 = sns.displot(df_filtered, x=X, y=Y, col=HUE_VAR,
                             rug=True)
            st.pyplot(g3)
            
             

        # # Display all possible bivariate combinations
        # st.write("Bivariate combinations:")
        # for combo in combinations(df_filtered.columns, 2):
        #     st.write(f"Displot of variables {combo[0]} and {combo[1]}")
        #     plt.figure(figsize=(10, 6))
        #     sns.displot(df_filtered, x=combo[0], y=combo[1], kind="kde")
        #     st.pyplot(plt)

