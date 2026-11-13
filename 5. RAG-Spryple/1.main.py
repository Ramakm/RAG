from dotenv import load_dotenv
load_dotenv()

from langchain_core import __version__ as core_version
from importlib.metadata import version as get_version

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

print(f"LangChain Core Version: {core_version}")
print(f"LangChain OpenAI Version: {get_version('langchain_openai')}")
print(f"LangChain Anthropic Version: {get_version('langchain_anthropic')}")
print(f"LangChain Groq Version: {get_version('langchain_groq')}")
print(f"Langgraph Version: {get_version('langgraph')}")

def main():

    # Let's chat with GROQ

    llm = ChatGroq(
        model_name = "openai/gpt-oss-20b",
        temperature= 0 
    )
    response = llm.invoke("We are just testing an LLM in my RAG project!!!")
    print(response)

    # Let's chat with OpenAI

if __name__ == "__main__":
    main()