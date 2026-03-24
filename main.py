import time

from similarity import get_scores
from chunker import chunk_data
from chunker import encode
import streamlit as st
import torch
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")


question = None
spinner_placeholder = None
answer_placeholder = None
new_info_placeholder = None
assistant_text = None
audio_input = None

if "chunk_cache" not in st.session_state:
    st.session_state.chunk_cache = []
    st.session_state.current_string = ""
    st.session_state.topics = {}
    st.session_state.current_topic = None
    st.session_state.chats = {}
    st.session_state.audio_key = 0
    st.session_state.personality = "Professor"

if "my_input" not in st.session_state:
    st.session_state.my_input = 0

client = None


@st.cache_resource
def get_client():
    client = Groq(api_key=api_key)
    return client



def get_past_conversation_texts(question):
    start = max(1, len(st.session_state.current_string) - 6000)
    prompt = f"""
    Search the past context for any relevance to the question or statement. Return only context that relates to them, eliminate all other information.
    If there is no past context, return nothing.
    Past Context: {st.session_state.current_string[start:]}
    """

    response = get_response(question, prompt)

    return response.choices[0].message.content

def get_keywords(question):
    prompt = f"""You are a Wikipedia search assistant, you must follow these rules.
    Extract the main search topics from the question below.
    If the question references something from the past conversation (like "he", "his", "they"), 
    resolve the reference and include the actual topic.
    If multiple distinct topics are mentioned, return each on a separate line.
    Return only Wikipedia search terms, nothing else.
    You may expand the topic up to ONLY SEVEN similar topics and include their keywords as well.
    If a topic is already found, do not repeat the topic again.
    Do not include commas.

    
    Past conversation: {st.session_state.current_string[:min(1500, len(st.session_state.current_string))]}
    Question: {question}"""
    
    response  = get_response(question, prompt)

    return response.choices[0].message.content.split("\n")
def get_response(userInput, prompt, stream=False):
    response = client.chat.completions.create(model = "llama-3.1-8b-instant", messages = [
        {
            "role" : "system",
            "content" : prompt,
        },
        {
            "role" : "user",
            "content" : userInput,
        },
    ], stream=stream)
    return response

def generate_answers(question, context, topics):
    prompt = f"""Your name is Cayde, and you are a Wikipedia Research Assistant. You must follow these rules:
    You MUST explain ALL of the following topics: {", ".join(topics)}
    If you want, you can add "\n" to separate lines and/or paragraphs. If so, add \n to the output.
    Be nice and kind, however do not be afraid to be honest with the user!
    Don't be afraid to call the user sir.
    Use the reference material below for information.
    Write a separate paragraph for each topic.
    Use your own words completely.
    Only respond with your response, do not include sources, hashtags, or anything extra.
    Your maximum word count is 200 words. Do not go over, and do not cut off at the end. End with a clear conclusion.
    Do not cite sources.
    At the end, ask the user another question to keep them thinking, or if they need anything else.
    If any, you may use the past conversations to help with your response. (i.e: Dont introduce yourself with each new message if a conversation has started!) {st.session_state.current_string[:min(1000, len(st.session_state.current_string))]}
    If the last conversation ended with a question, check to see if the user input answers that question. If so, respond accordingly.
    Your personality is: {st.session_state.personality}. Talk like your personality.

    Reference material: {context}
    Response:"""

    response = get_response(question, prompt)

    return response.choices[0].message.content
def get_searches(list_of_keywords):
    chunk_list = []
    for keyword in list_of_keywords:
        if len(keyword) == 0:
            continue
        chunk_dict, alreadyFound = chunk_data(keyword, st.session_state)
        if not chunk_dict:
            continue
        
        chunk_list.append(chunk_dict)
    return chunk_list

def add_chats():
    print(st.session_state.chats)
    for index in range(len(st.session_state.chats)):
        message = st.session_state.chats[str(index)]
        with st.chat_message(message["role"]):
            st.write(message["text"])

def add_to_chats(role, message):
    print(role, message[:50])
    length = len(st.session_state.chats)
    st.session_state.chats[str(length)] = {"role" : role, "text" : message}



def get_answer(chunk_searches, key_words):
    top_sentence_scores = ""
    for chunk_dict in chunk_searches:
        userinput = question
        embeds = [chunk_dict[index]["embedding"] for index in chunk_dict]

        embedding_matrix = torch.stack(embeds)
        print("-")
        print("-------")
        embedded_input = encode(userinput)

        if not embedded_input in locals() or not embedding_matrix in locals():
            continue

        k_scores = get_scores(embedded_input, embedding_matrix, 3)

                        
        stringUsing = ""


        for i in k_scores.indices:
            sentence = chunk_dict[i.item()]["chunk"]
            stringUsing = f"{stringUsing} {sentence}"
                        
        top_sentence_scores = f"{top_sentence_scores} {stringUsing}"
                

    answer = generate_answers(userinput, top_sentence_scores, key_words)
    paragraph = answer
    
    answer = answer.split(" ")

    spinner_placeholder.empty()
    answer_placeholder.success("Heres what I found:")

    return answer, paragraph

def update_database(paragraph):
    st.session_state.chunk_cache.append(f"{paragraph}\n\n")
    st.session_state.current_string = f"{st.session_state.current_string}\n\n {paragraph}"

def main(): 
    global question
    if question:
        if audio_input:
            print("has audio input!")
            question = audio_input
            transcription = client.audio.transcriptions.create(
                model = "whisper-large-v3",
                file = audio_input
            )
            question = transcription.text
            clear_audio()

        update_database(question)
        with st.chat_message("user"):
            st.write(question)
            add_to_chats("user", question)
        with st.chat_message("assistant"):
            paragraph = None
            answer = None
            with st.spinner("Thinking..."):
                key_words = get_keywords(question)
                max_keywords = 8
                key_words = key_words[:min(max_keywords, len(key_words))]
                
                print(f"KEY WORDS: {key_words}")
                chunk_searches = get_searches(key_words)

                answer, paragraph = get_answer(chunk_searches, key_words)

            update_database(paragraph)
            answer_placeholder.empty()
            add_to_chats("assistant", f"{paragraph}\n\n")

            new_total_text = ""
            message = st.empty()
            for word in answer:
                new_total_text = f"{new_total_text} {word}"
                message.write(new_total_text)
                time.sleep(0.1)
        
                


def clear_text():
    print("CLEARING INPUT")
    st.session_state["my_input"] += 1
    value = st.session_state.get("my_input", "")

def clear_audio():
    print("CLEARING AUDIO")
    st.session_state.audio_key += 1

def load_app():
    global question
    global spinner_placeholder
    global answer_placeholder
    global new_info_placeholder
    global assistant_text
    global audio_input
    st.title("Cayde AI")
    st.text("Your Personal Wikipedia Assistant")
    st.text("Made by Nikobro82")

    spinner_placeholder = st.empty()
    new_info_placeholder = st.empty()
    answer_placeholder = st.empty()
    assistant_text = st.empty()


    question = st.chat_input("Ask a question!", key=f"audio_{st.session_state.my_input}")
        
    with st.sidebar:
        audio_input = st.audio_input("Use your voice!", key=f"input_{st.session_state.audio_key}")
        st.session_state.personality = st.selectbox(
            "Personality",
            ["Professor", "Cayde-6", "Abraham Lincoln", "Ghost from Destiny 2", "Napoleon Bonaparte", "Winston Churchill", "Jotaro Kujo", "Dio Brando", "Commander Zavala", "Johnny Joestar", "Gyro Zeppeli", "Joseph Joestar (Part 2)"]
        )



client = get_client()
load_app()
add_chats()
main()