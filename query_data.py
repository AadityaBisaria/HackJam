
import argparse
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from rich import print
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question only in the context of the pelopponesian war in around a hundred words based on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
from diffusers import DiffusionPipeline
import torch
from PIL import Image

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors = True, variant = "fp16")
pipe.to("cuda")

prompt = "Ancient Athenian Soldier"

images = pipe(prompt=prompt).images[0]
images.show()
images.save("ancient athenian soldier.png")  # Save the image to a file
print("Image saved.png'")
# Load environment variables
load_dotenv()

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

    prompt = formatted_response

    images = pipe(prompt=prompt).images[0]
    images.show()
    images.save("A.png")  # Save the image to a file

if __name__ == "__main__":
    main()

