import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from streamlit_extras.add_vertical_space import add_vertical_space
from pandasai import SmartDataframe
from pandasai.llm import Starcoder as LLM
from pandasai.callbacks import StdoutCallback

# At the moment, PandasAI supports the following HuggingFace models:

# Starcoder: bigcode/starcoder
# Falcon: tiiuae/falcon-7b-instruct


load_dotenv()


# Sidebar contents
with st.sidebar:
    st.set_page_config(
        page_title="Pandas DataFrame Analyzer - Prompt-driven analysis with PandasAI"
    )
    st.title("🤗💬 Pandas DataFrame Analyzer")
    st.markdown(
        """
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    """
    )
    add_vertical_space(5)
    st.write("Made with ❤️ by [AlexPepito](https://youtube.com/@engineerprompt)")


llm = LLM(api_token=os.environ["HUGGINGFACE_API_KEY"])


def main():
    st.header("Ask Pandas about your file")

    csv_files = st.file_uploader("Upload your CSV file", type="csv")

    if csv_files is not None:
        df = pd.read_csv(csv_files)
        st.write("Mean of Fare price: %s" % df["Fare"].mean())
        st.write(df.head(5))
        # conversational=False is supposed to display lower usage and cost
        # Callbacks are functions that are called at specific points during the execution,
        # i.e: to print the code as soon as it is generated
        smart_df = SmartDataframe(
            df,
            config={"llm": llm, "conversational": False, "callback": StdoutCallback()},
        )

        prompt = st.text_area("Enter your question:")
        if st.button("Submit"):
            if prompt:
                with st.spinner("Generating response..."):
                    res = smart_df.chat(prompt)
                    st.write(res)
        else:
            st.warning("Please provide a prompt")


if __name__ == "__main__":
    main()
