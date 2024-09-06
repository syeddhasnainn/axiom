import os
import sys
import json
import requests
import time
import random
import string
import hashlib
import base64
from parsel import Selector
import scraper_helper
import asyncio
import aiohttp
import os
from together import Together
from dotenv import load_dotenv
from pypdf import PdfReader
from pydantic import BaseModel, Field
import streamlit as st
load_dotenv()

st.set_page_config(page_title="Arxiv Scraper", page_icon=":page_facing_up:")

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
PROMPT = """
You are a helpful assistant that extracts specific sections that are given to you from pdfs, all of the pdfs will be some kind of research paper, you have to find abstract, introduction,future direction, research gaps and conclusion. I don't need the results, just text data. You have to extract the data in a structured format, and you have to give me the data in a json format. do not hallucinate. Output the json only
"""

class ModelOutput(BaseModel):
    abstract: str = Field(
        description="The abstract of the paper usually in the beginning of the paper"
    )
    introduction: str
    future_direction: str
    research_gap: str
    conclusion: str


def pdf_to_text(pdf_path):
    reader = PdfReader(pdf_path)
    meta = reader.metadata
    creation_date = meta.creation_date
    page = reader.pages
    full_text = ""
    for p in page:
        text = p.extract_text()
        full_text += text
    return full_text, creation_date


def extract_data(full_text, download_path):
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": full_text},
        ],
        response_format={
            "type": "json_object",
            "schema": ModelOutput.model_json_schema(),
        },
    )
    with open(f"{download_path}/summary.json", "w") as f:
        f.write(response.choices[0].message.content)
    return response.choices[0].message.content


def arxiv_scraper(query, max_results):
    print('we are in arxiv')
    url = requests.get(
        f"https://arxiv.org/search/?query={query}&searchtype=all&abstracts=show&order=&size=50"
    )
    resp = Selector(text=url.text)
    all_pdfs = []
    for li in resp.xpath('//ol[@class="breathe-horizontal"]/li'):
        title = scraper_helper.cleanup(
            "".join(li.xpath('./p[contains(@class,"title")]//text()').getall())
        )
        title = title.replace("/", "_")
        pdf = li.xpath('.//a[contains(text(), "pdf")]/@href').get()
        all_pdfs.append({"title": title, "pdf": pdf})
    print(all_pdfs)
    return all_pdfs[:max_results]


async def download_pdf(session, pdf_url, title, query):
    async with session.get(pdf_url) as response:
        if response.status == 200:
            os.makedirs("db", exist_ok=True)
            os.makedirs(f"db/{query}", exist_ok=True)
            os.makedirs(f"db/{query}/{title}", exist_ok=True)
            download_path = f"db/{query}/{title}"
            with open(f"{download_path}/paper.pdf", "wb") as f:
                f.write(await response.read())
            full_text, creation_date = pdf_to_text(f"{download_path}/paper.pdf")
            return extract_data(full_text, download_path)
        else:
            print(f"Failed to download {title}")


async def main(query, max_results):
    all_pdfs = arxiv_scraper(query, max_results)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for pdf in all_pdfs:
            tasks.append(download_pdf(session, pdf["pdf"], pdf["title"], query))
        
        progress_bar = st.progress(0)
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            await task
            progress_bar.progress(i / len(tasks))


def display_db_contents():
    st.title("Database Contents")

    if not os.path.exists("db"):
        st.warning("The database folder does not exist yet. Run a search to populate it.")
        return

    queries = os.listdir("db")
    selected_query = st.selectbox("Select a query", [""] + queries)

    if selected_query:
        query_path = os.path.join("db", selected_query)
        papers = os.listdir(query_path)
        
        for paper in papers:
            paper_path = os.path.join(query_path, paper)
            summary_path = os.path.join(paper_path, "summary.json")
            
            if os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    data = json.load(f)
                
                with st.expander(f"Paper: {paper}"):
                    tabs = st.tabs(list(data.keys()))
                    for tab, (key, value) in zip(tabs, data.items()):
                        with tab:
                            st.write(value)
                    
                    pdf_path = os.path.join(paper_path, "paper.pdf")
                    if os.path.exists(pdf_path):
                        with open(pdf_path, "rb") as pdf_file:
                            st.download_button(
                                label="Download PDF",
                                data=pdf_file,
                                file_name=f"{paper}.pdf",
                                mime="application/pdf"
                            )
        st.divider()

def main_page():
    st.title('Research Tool')

    max_results = st.number_input("Max results", min_value=2, value=5)
    query = st.text_input("Keyword to search for")

    if st.button("Search and Extract"):
        if query:
            with st.spinner("Searching and extracting data..."):
                asyncio.run(main(query, max_results))
            st.success("Extraction complete!")
            st.info("Please go to the 'Database Contents' page in the sidebar to view the extracted PDFs.")
        else:
            st.warning("Please enter a keyword to search for.")

if __name__ == "__main__":
    if 'page' not in st.session_state:
        st.session_state.page = "Search"

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Search", "Database Contents"], index=0 if st.session_state.page == "Search" else 1)

    if page == "Search":
        main_page()
    elif page == "Database Contents":
        display_db_contents()