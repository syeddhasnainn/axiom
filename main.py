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

load_dotenv()
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


reader = PdfReader(
    "nlp/A Comparison of LLM Finetuning Methods & Evaluation Metrics with Travel Chatbot Use Case.pdf"
)
page = reader.pages
full_text = ""
for p in page:
    text = p.extract_text()
    full_text += text


client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    messages=[
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": full_text},
    ],
    response_format={"type": "json_object", "schema": ModelOutput.model_json_schema()},
)
print(response.choices[0].message.content)
# query = "nlp"
# def arxiv_scraper(query):
#     url = requests.get(f"https://arxiv.org/search/?query={query}&searchtype=all&abstracts=show&order=&size=50")
#     resp = Selector(text=url.text)
#     all_pdfs = []
#     for li in resp.xpath('//ol[@class="breathe-horizontal"]/li'):
#         title = scraper_helper.cleanup(''.join(li.xpath('./p[contains(@class,"title")]//text()').getall()))
#         title = title.replace("/", "_")
#         pdf = li.xpath('.//a[contains(text(), "pdf")]/@href').get()
#         all_pdfs.append({"title": title, "pdf": pdf})
#     return all_pdfs

# async def fetch_pdf(session, pdf_url, title):
#     async with session.get(pdf_url) as response:
#         if response.status == 200:
#             os.makedirs(query, exist_ok=True)
#             with open(f"{query}/{title}.pdf", "wb") as f:
#                 f.write(await response.read())
#         else:
#             print(f"Failed to download {title}")

# async def main():
#     all_pdfs = arxiv_scraper(query)
#     async with aiohttp.ClientSession() as session:
#         tasks = []
#         for pdf in all_pdfs:
#             tasks.append(fetch_pdf(session, pdf["pdf"], pdf["title"]))
#         await asyncio.gather(*tasks)

# if __name__ == "__main__":
#     asyncio.run(main())
