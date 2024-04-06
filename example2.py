## Integrate our code OpenAI API
import os
from constants import openai_key
from langchain_openai import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

from langchain.memory import ConversationBufferMemory

from langchain.chains import SequentialChain

import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

# streamlit framework

st.title('Celebrity Search Results')
input_text=st.text_input("Search the topic u want")

# Prompt Templates

messages = [
    "That's your quality???",
    "Enough is enough....",
    "Bangladesh  me delivary available?"
    "*آن لائن ورکرز کی ضرورت ہے۔ کام فیس بک اور واٹس ایپ پہ ہوگا۔ ڈیلی انکم*
*1ہزار سے 2 ہزار🤝💯*
*سیریس لوگ ہی رابطہ کریں*

*Reply "Yes" For Details*inbox me",
    "নাম্বার টা  দিন প্লিজ",
    "Height looks good in this dress",
    "",
]

# We kindly request you to contact the TCS Courier company to get an update on it directly. 

# Best Regards, 
# Team Asim Jofa.

first_input_prompt=PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)

# Memory

person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')

## OPENAI LLMS
llm=OpenAI(temperature=0.8)
chain=LLMChain(
    llm=llm,prompt=first_input_prompt,verbose=True,output_key='person',memory=person_memory)

# Prompt Templates

second_input_prompt=PromptTemplate(
    input_variables=['person'],
    template="when was {person} born"
)

chain2=LLMChain(
    llm=llm,prompt=second_input_prompt,verbose=True,output_key='dob',memory=dob_memory)
# Prompt Templates

third_input_prompt=PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events happened around {dob} in the world"
)
chain3=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='description',memory=descr_memory)
parent_chain=SequentialChain(
    chains=[chain,chain2,chain3],input_variables=['name'],output_variables=['person','dob','description'],verbose=True)



if input_text:
    st.write(parent_chain({'name':input_text}))

    with st.expander('Person Name'): 
        st.info(person_memory.buffer)

    with st.expander('Major Events'): 
        st.info(descr_memory.buffer)