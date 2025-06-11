import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit.components.v1 import html
import base64


class GetReport:
    @staticmethod
    def generate_html_report(df):
        profile = ProfileReport(df, title="DataWizard EDA Report", explorative=True)
        profile.to_file("eda_report.html")
        return "eda_report.html"

    @staticmethod
    def download_button(file_path, label="Download Report", file_type="html"):
        with open(file_path, "rb") as f:
            content = f.read()
            b64 = base64.b64encode(content).decode()
            mime = "application/pdf" if file_type == "pdf" else "application/html"
            href = f'<a href="data:{mime};base64,{b64}" download="{file_path}">{label}</a>'
            st.markdown(href, unsafe_allow_html=True)
