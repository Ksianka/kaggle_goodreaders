import requests
import streamlit as st

text_input = st.text_input('Please input book review to get predicted rating from 0 to 5')

if len(text_input) > 0:
    st.write('### Text:\n\n')
    st.write('### '+ text_input)

    r = requests.get(f'http://127.0.0.1:8000/{text_input}')
    st.write('### ' + str(r.text))