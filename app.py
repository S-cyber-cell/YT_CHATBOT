import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import os

st.title("🎥 YouTube Q&A with AI")
# inp=st.text_input("enter something")
# if st.button("click"):
#code runs 

# Input OpenAI API key (no secrets file dependency)
OPENAI_API_KEY = st.text_input("Enter your OpenAI API key", type="password")
if not OPENAI_API_KEY:
    st.warning("Please enter your OpenAI API key above.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

video_id = st.text_input("Enter YouTube Video ID (e.g. dQw4w9WgXcQ):")
user_question = st.text_input("What would you like to ask about this video?")

if st.button("Get Answer"):
    if not video_id or not user_question:
        st.error("Please provide both YouTube ID and your question.")
    else:
        try:
            st.info("Fetching transcript...")
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id,languages=['hi'])
            transcript = " ".join([t['text'] for t in transcript_list])

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_text(transcript)
            st.info("Embedding transcript and building vector store...")
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(texts, embedding=embeddings)

            retriever = vectorstore.as_retriever()

            template = """Answer the following question based on the video transcript:\n\n{question}"""
            prompt = PromptTemplate.from_template(template)
            llm = ChatOpenAI(temperature=0)

            docs = retriever.get_relevant_documents(user_question)
            context = " ".join([doc.page_content for doc in docs])

            st.info("Generating answer...")
            answer = llm.invoke(prompt.format(question=f"{user_question}\nContext: {context}"))

            st.success("Answer:")
            st.write(answer.content)

        except TranscriptsDisabled:
            st.error("This video does not have transcripts enabled.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
#xkPoXrmOXMg     
#0-FUhQKe-eU     14170287
#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