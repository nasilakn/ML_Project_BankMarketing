import streamlit as st
import base64
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# st.set_page_config(
#     page_title="Bank Marketing",
#     page_icon=":bank:",)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string});
            background-size: cover;}}
        </style>""",unsafe_allow_html=True)

if __name__ == '__main__':
    add_bg_from_local('image1.png')


def load_data(path):
    data = pd.read_csv(path,sep=';')
    return data

df = load_data('bank-full.csv')

def load_data(path):
    data = pd.read_csv(path)
    return data

df1 = load_data('df (3).csv')


def bank():





    # Sidebar for navigation

        page = st.sidebar.radio("Go To",["Home Page", "About The Dataset", "About the ML Model", "Model Prediction","Conclusion"])

        # Display selected page
        if page == "Home Page":
            st.markdown("<h1 style='font-size:60px; color:darkorange'>Welcome to the Homepage</h1>", unsafe_allow_html=True)
            #st.markdown("<h2 style='font-size:24px; color:green;'>Subtitle with Custom Style</h2>" unsafe_allow_html=True)


            st.markdown("<h1 style='font-size:30px; color:teal'>Intorduction</h1>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:18px;color:black'>This is a Streamlit web app created for the purpose of my Machine Learning (ML) project. The project involves creating and training an ML model on data from a bank marketing campaign.</p>", unsafe_allow_html=True)
            st.markdown("<h1 style='font-size:30px; color:teal'>Purpose of the Project.</p>",unsafe_allow_html=True)
            st.markdown("""<ul style='font-size:18px;'>
                            <li>To demonstrate the application of Machine Learning techniques on real-world data.</li>
                            <li>To help banking institutions analyze the effectiveness of their campaigns and better target their future campaigns.</li>
                            <li>To predict the likelihood of a customer subscribing to a term deposit based on the marketing campaign data.</li>
                            <li>To understand which factor of their campaign influenced their clients more
                            <li>Data-Driven Decision Making
                            <li>Thereby increasing the efficiency of their campaign
                        </ul>""", unsafe_allow_html=True)

            st.markdown("<h1 style='font-size:30px; color:teal'>Features of the App.</p>", unsafe_allow_html=True)
            st.markdown("""
                        <ul style="font-size:30px;">
                        <li> User-friendly interface.
                        <li> Interactive data visualizations to explore the dataset.
                        <li> Model predictions based on user inputs.
                        <ul>
                        """, unsafe_allow_html=True)
            st.markdown("<h1 style='font-size:30px; color:teal'>Getting Started.</p>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:18px;'>Use the sidebar to navigate through the different sections of the app. Each section provides specific functionalities and visualizations related to the project's objectives.</p>", unsafe_allow_html=True)
            st.markdown("<h1 style='font-size:30px;color:teal'>Enjoy exploring the app!</p>", unsafe_allow_html=True)



        elif page == "About The Dataset":
            st.markdown("<h1 style='font-size:50px; color:darkorange'>Bank Marketing Campaign Dataset</h1>", unsafe_allow_html=True)

            # Tabs in Data Upload Page
            tab1, tab2, tab3 = st.tabs(["Introduction", "Dataset Information","Dataset Visualizations"])
            with tab1:
                st.markdown("<h1 style='font-size:30px;color:teal'>The data set is related with direct marketing campaigns of a Portuguese banking institution.</p>", unsafe_allow_html=True)
                st.markdown( "<p style='font-size:18px;'>The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (Bank term deposit) would be ('Yes') or not ('No') subscribed.</p>",
                    unsafe_allow_html=True)

            with tab2:
                st.markdown("<h1 style='font-size:30px;color:teal'>Here is the dataset used for training the ML model</p>", unsafe_allow_html=True)
                st.write(
                    "- [Link to Bank Marketing Campaign Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)")
                st.dataframe(df)
                st.markdown("<h1 style='font-size:30px;color:teal'>Dataset Information</p>",
                    unsafe_allow_html=True)
                p = st.selectbox('Select the information you want to know',['Shape', 'Size', 'Describe'])
                if p=='Shape':
                    dataset_shape=df.shape
                    st.write(f"**Shape**:{dataset_shape}")

                elif p=='Size':
                    dataset_size = df.size
                    st.write(f"**Size**{dataset_size}")

                elif p=='Describe':
                    st.dataframe(df.describe())



            with tab3:
                st.markdown("<h1 style='font-size:30px;color:teal'>Visualizations of Dataset</p>",unsafe_allow_html=True)
                # Select columns to visualize
                numeric_columns = df.select_dtypes(['float64', 'int64']).columns
                categorical_columns = df.select_dtypes(['object']).columns

                # Joint plot
                st.write("### Joint Plot")
                on = st.toggle("Show Joint Plot")
                if on:
                    selected_joint_x = st.selectbox("Select X-axis variable for joint plot", numeric_columns,
                                                    key="joint_x")
                    selected_joint_y = st.selectbox("Select Y-axis variable for joint plot", numeric_columns,
                                                    key="joint_y")
                    selected_joint_hue = st.selectbox("Select hue for joint plot", categorical_columns, key="joint_hue")
                    fig = sns.jointplot(x=selected_joint_x, y=selected_joint_y, data=df, hue=selected_joint_hue,
                                        kind="scatter")
                    st.pyplot(fig)
                    # Count plot
                st.write("### Count Plot")
                on1 = st.toggle("Show Count Plot")
                if on1:
                        selected_count_column = st.selectbox("Select column for count plot", categorical_columns,
                                                             key="count")
                        selected_count_hue = st.selectbox("Select hue for count plot", categorical_columns,
                                                          key="count_hue")
                        fig, ax = plt.subplots()
                        sns.countplot(x=selected_count_column, hue=selected_count_hue, data=df, ax=ax)
                        plt.xticks(rotation=90)  # Rotate x-axis labels if necessary
                        st.pyplot(fig)

                    #Correlation heatmap
                st.write("### Correlation Heatmap")
                on2 = st.toggle("Show Correlation Heatmap")
                if on2:
                        df1 = load_data('df (3).csv')
                        corr_matrix = df1.corr()
                        fig, ax = plt.subplots()
                        sns.heatmap(corr_matrix, annot=True,fmt=".2f", annot_kws={"size": 5}, linewidths=.5,cmap='coolwarm', ax=ax, )
                        plt.title('Correlation Heatmap')
                        st.pyplot(fig)


        elif page == "About the ML Model":
            st.markdown("<h1 style='font-size:60=px; color:darkorange'>About the ML Model</h1>", unsafe_allow_html=True)


            st.markdown("<h1 style='font-size:30px;color:teal'>Model Overview</p>",unsafe_allow_html=True)
            st.markdown("<p style='font-size:18px;'>This machine learning model is Random Forest Classifier. It is trained on data from a bank marketing campaign, with the aim of predicting the success of marketing efforts.odel Overview</p>",unsafe_allow_html=True)
            st.markdown("<h1 style='font-size:30px;color:teal'>Data Preprocessing</p>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:18px;'>The initial step in training the model involved preprocessing the data. This included:</p>", unsafe_allow_html=True)
            st.markdown("""<ul style="font-size:20px;">
                                  <li> Handling missing values.
                                  <li> Encoding categorical variables.
                                  <li> Scaling numerical features.
                                  <ul>""", unsafe_allow_html=True)
            st.markdown("<p style='font-size:18px;'>These steps ensured that the data was clean and ready for model training.</p>",
                        unsafe_allow_html=True)
            st.markdown("<h1 style='font-size:30px;color:teal'>Handling Imbalanced Data</p>",unsafe_allow_html=True)
            st.markdown("<p style='font-size:18px;'>The dataset exhibited class imbalance, which could negatively impact the model's performance on the minority class. To address this, the data was oversampled using techniques SMOTE (Synthetic Minority Over-sampling Technique). This helped in creating a balanced dataset for training.</p>", unsafe_allow_html=True)
            st.markdown("<h1 style='font-size:30px;color:teal'>Model Selection</p>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:18px;'>The following models were tested:</p>", unsafe_allow_html=True)
            st.markdown("""<ul style="font-size:20px;">
                                            <li> K-Nearest Neighbors (KNN)
                                            <li> Support Vector Classifier (SVC)
                                            <li> Naive Bayes (NB)
                                            <li>Decision Tree (DT)
                                            <li>Random Forest (RF)
                                            <li>AdaBoost
                                            <li>Gradient Boosting
                                            <li>XGBoost
                                            <ul>""", unsafe_allow_html=True)
            st.markdown("<h1 style='font-size:30px;color:teal'>Model Performance Comparison</p>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:18px;'>A Random Forest Classifier was chosen as the model for this project as it gave high accuracy,precession and recall compared to other models. Random Forest is an ensemble learning method that combines multiple decision trees to improve the accuracy and robustness of the prediction.</p>", unsafe_allow_html=True)
            st.markdown("<h1 style='font-size:30px;color:teal'>Hyperparameter Tuning</p>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:18px;'>Hyperparameter tuning was performed to optimize the model's performance. Various combinations of parameters were tested using grid search and cross-validation. The key hyperparameters tuned included:</p>", unsafe_allow_html=True)
            st.markdown("""<ul style="font-size:20px;">
                                                    <li> Criterion (Gini or Entropy)
                                                    <li> Maximum depth of the trees
                                                    <li> Random state
                                                    <ul>""", unsafe_allow_html=True)
            st.markdown("<p style='font-size:18px;'>The best combination of hyperparameters was selected based on the performance on the validation set.</p>", unsafe_allow_html=True)
            st.markdown("<h1 style='font-size:30px;color:teal'>Training the Model</p>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:18px;'>The model was trained on both the original and oversampled datasets. The training process involved fitting the model to the training data and evaluating its performance using metrics such as accuracy, precision, and recall.</p>", unsafe_allow_html=True)
            st.markdown("<h1 style='font-size:30px;color:teal'>Evaluation and Validation</p>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:18px;'>After training the model, it was evaluated on the test dataset to assess its performance. The evaluation metrics provided insights into how well the model generalizes to unseen data. The confusion matrix, accuracy, precision, and recall were key metrics used in this evaluation.</p>", unsafe_allow_html=True)

        elif page == "Model Prediction":
            st.markdown("<h1 style='font-size:60px; color:darkorange;'>Model Prediction</h1>", unsafe_allow_html=True)
            st.write(
                "- [Link to my ColabNotebook](https://colab.research.google.com/drive/15lWSW_5Y4oW5-w4vDLp49rb1JFgx4QjI?usp=sharing)")

            model=pickle.load(open('rf_model.sav','rb'))

            scaler=pickle.load(open('scaler.sav','rb'))

            age = st.text_input('Enter the age of client')
            job = st.selectbox('Enter the job of the client',
                               ['Blue-collar', 'Management', 'Technician', 'Admin', 'Services', 'Retired',
                                'Self-employed', 'Entrepreneur', 'Unemployed', 'Housemaid', 'Student'])
            if job == 'Blue-collar':
                j = 1
            elif job == 'Management':
                j = 4
            elif job == 'Technician':
                j = 9
            elif job == 'Admin':
                j = 0
            elif job == 'Services':
                j = 7
            elif job == 'Retired':
                j = 5
            elif job == 'Self-employed':
                j = 6
            elif job == 'Entrepreneur':
                j = 2
            elif job == 'Unemployed':
                j = 10
            elif job == 'Housemaid':
                j = 3
            else:
                j = 8

            marital_status = st.radio('Enter the marital status of client', ['Single', 'Married', 'Divorced'])
            if marital_status == 'Single':
                ms = 2
            elif marital_status == 'Married':
                ms = 1
            else:
                ms = 0

            educational_status = st.radio('Enter the educational status of client',
                                          ['Primary', 'Secondary', 'Tertiary'])
            if educational_status == 'Primary':
                es = 0
            elif educational_status == 'Secondary':
                es = 1
            else:
                es = 2

            loan_default_status = st.radio('Loan default history of client', ['No', 'Yes'])
            if loan_default_status == 'No':
                lds = 0
            else:
                lds = 1

            account_balance = st.text_input('Enter the account balance of client')

            housing_status = st.radio('Does the client owns a house', ['No', 'Yes'])
            if housing_status == 'No':
                hs = 0
            else:
                hs = 1

            loan_status = st.radio('Does the client has taken any loan', ['No', 'Yes'])
            if loan_status == 'No':
                ls = 0
            else:
                ls = 1
            contacted_day_of_month = st.slider('Enter the day of the month when the client was coontacted', min_value=1,
                                               max_value=31)

            month = st.selectbox('Enter the month in which client was contacted',
                                 ['January', 'Febuary', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
                                  'October', 'November', 'December'])
            if month == 'January':
                m = 4
            elif month == 'Febuary':
                m = 3
            elif month == 'March':
                m = 7
            elif month == 'April':
                m = 0
            elif month == 'May':
                m = 8
            elif month == 'June':
                m = 6
            elif month == 'July':
                m = 5
            elif month == 'August':
                m = 1
            elif month == 'September':
                m = 11
            elif month == 'October':
                m = 10
            elif month == 'November':
                m = 9
            else:
                m = 2

            duration_of_call = st.text_input('Enter the duration of call with the client in second')

            campaign_call = st.text_input('Enter the number of campaign calls made to the client')

            no_days_passed = st.text_input(
                'Enter the number of days passed after the last campaign calls made to the client')

            previous_campaign_call = st.text_input(
                'Enter the number of contacts made to the client before this campaign')

            pred = st.button(':red[PREDICT]')
            if pred:
                try:
                    prediction = model.predict(scaler.transform([[age, j, ms, es, lds, account_balance, hs, ls,
                                                                  contacted_day_of_month, m, duration_of_call,
                                                                  campaign_call, no_days_passed,
                                                                  previous_campaign_call]]))
                    if prediction == 0:
                        st.write('#### Client will not subscribe to Term deposit :x:')
                    else:
                        st.write('#### Client will subscribe to Term deposit :white_check_mark:')
                except:
                    st.write('#### Make sure you have given all the necessary inputs :exclamation:')



        elif page == "Conclusion":
           st.markdown("<h1 style='font-size:60px; color:darkorange;'>Conclusion</h1>", unsafe_allow_html=True)


           st.markdown("<h1 style='font-size:30px; color:teal'>Summary Of The Project</h1>", unsafe_allow_html=True)
           st.write("""This project focused on developing and training a machine learning model using data from a bank marketing campaign. The main objectives were to understand customer behavior, identify key features influencing customer decisions, and predict the success of marketing campaigns.""")
           st.markdown("<h1 style='font-size:30px; color:teal'>Relevance Of This Project</h1>", unsafe_allow_html=True)
           st.write("""This project will enable banking institutions to conduct successful marketing campaigns. Effective campaigns will encourage households to develop stronger saving habits, which in turn will enhance the banks' ability to extend credit.
                   According to the latest data from the ministry of statistics, net household savings declined sharply by ₹9 trillion in the last 3 years
                   The decline in household savings underscores the necessity of the project. By enabling banks to run effective marketing campaigns, the project aims to counteract this decline, encourage saving habits, and ultimately strengthen the banks' capacity to provide credit.""")

           st.markdown("<h1 style='font-size:30px; color:teal;'>Key Findings</h1>", unsafe_allow_html=True)
           st.write("""
             1. **Contact Duration**: The duration of the last contact with the customer has a substantial impact on the likelihood of a positive outcome.
             2. **Marital Status**: Married clients opted more for term deposit
             3. **Job**: Clients belonging to Management job opted more for term deposit
             4. **Age and Account balance**: Irrespective of the age and account balance of the clients most of them chose not to opt for term deposit """)

           st.markdown("<h1 style='font-size:30px; color:teal'>Model Performance</h1>", unsafe_allow_html=True)
           st.write("""The machine learning model was evaluated using various performance metrics. The model showed promising results, with a high accuracy and precision in predicting customer responses. However, there is always room for improvement, and further tuning and feature engineering could enhance the model's performance.""")



bank()