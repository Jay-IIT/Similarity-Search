import streamlit as st
import requests
import pandas as pd
API_URL = 'http://localhost:5000/recommendations'
st.set_page_config(page_title='Peloton App')
st.title("Peloton Recommendation System")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
question = st.text_input("Enter your preference")
if st.button("Get Recommendations"):
    if question:
        payload = {'question': question}
        try:
            response = requests.post(API_URL, json=payload)
            recommendations = response.json()
            st.subheader("Recommendations:")
            df = pd.DataFrame(recommendations)
            df = df.drop_duplicates()
           # CSS to inject contained in a string
            hide_dataframe_row_index = """
                        <style>
                        .row_heading.level0 {display:none}
                        .blank {display:none}
                        </style>
                        """

            # Inject CSS with Markdown
            st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
            # Iterate over rows
            # Iterate over rows
            for index, row in df.iterrows():
               # st.write(f"Row index: {index}")
                st.write(f'<span style="color:red; font-weight:bold;">Recommendation {index+1}</span>', unsafe_allow_html=True)
                for column, value in row.items():
                    st.write(f'<span style="color:red; font-weight:bold;">{column.upper()}</span>: {value}', unsafe_allow_html=True)
                    #st.write(f"{column}: {value}")
                st.write("---------------------")
            # Display an interactive table
           # st.dataframe(df)

        except requests.exceptions.RequestException as e:
            st.error("Error occurred during API request.")
    else:
        st.warning("Please enter a question.")

