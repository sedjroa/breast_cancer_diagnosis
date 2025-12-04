import streamlit as slt
import pickle as pk
import pandas as pd

def main():
    slt.set_page_config(page_title='Breast Cancer Diagnosis', page_icon=":female-doctor:")
    slt.write('Hello world')
    
if __name__ == "__main__":
    main()