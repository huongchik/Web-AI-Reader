import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
import streamlit_authenticator as stauth
import sqlite3
from langchain.memory import ChatMessageHistory
from langchain import PromptTemplate
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


models = [
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-32k",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-16k",
]
MODEL = models[5]


connection_users = sqlite3.connect("users.db")
cursor_users = connection_users.cursor()


connection_dialogues = sqlite3.connect("dialogues.db")
cursor_dialogues = connection_dialogues.cursor()


def create_db_dialog(name):
    cursor_dialogues.execute(
        f"""CREATE TABLE IF NOT EXISTS {name}
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_message TEXT,
                ai_response TEXT)"""
    )


def add_user_message(message, name):
    cursor_dialogues.execute(
        f"INSERT INTO {name} (user_message) VALUES (?)", (message,)
    )
    connection_dialogues.commit()


def add_ai_response(response, name):
    cursor_dialogues.execute(
        f"UPDATE {name} SET ai_response = ? WHERE id = (SELECT MAX(id) FROM {name})",
        (response,),
    )
    connection_dialogues.commit()


def get_all_saved_messages(name):
    if name != "":
        cursor_dialogues.execute(f"SELECT * FROM {name}")
        return cursor_dialogues.fetchall()


def create_usertable():
    cursor_users.execute(
        "CREATE TABLE IF NOT EXISTS userstable(name TEXT, username TEXT,password TEXT)"
    )


def add_userdata(name, username, password):
    cursor_users.execute(
        "INSERT INTO userstable(name, username,password) VALUES (?,?,?)",
        (name, username, password),
    )
    connection_users.commit()


def view_all_users():
    cursor_users.execute("SELECT * FROM userstable")
    data = cursor_users.fetchall()
    return data


def get_all_dialogues(username):
    cursor_dialogues.execute(
        f"SELECT name FROM sqlite_master WHERE type='table' and name is not 'sqlite_sequence' and name GLOB '{username}_*'"
    )
    tables = cursor_dialogues.fetchall()
    return [(table[0], i + 1) for i, table in enumerate(tables)]


def save_dialog(dialogue_id):
    create_db_dialog(dialogue_id)
    for msg in st.session_state.chat_history:
        if type(msg) == HumanMessage:
            add_user_message(msg.content, dialogue_id)
        if type(msg) == AIMessage:
            add_ai_response(msg.content, dialogue_id)


def load_context_from_sql(name):
    st.session_state.chat_history.clear()
    results = get_all_saved_messages(name)
    chat_history = []
    for result in results:
        question = str(result[1])
        answer = str(result[2])
        user_message = HumanMessage(content=question)
        ai_message = AIMessage(content=answer)
        chat_history.append(user_message)
        chat_history.append(ai_message)
    return chat_history


def create_context_from_history(db):
    llm = ChatOpenAI(
        model=MODEL,
        temperature=0.7,
    )
    memory = ConversationBufferMemory(
        chat_memory=ChatMessageHistory(messages=st.session_state.chat_history),
        return_messages=True,
        memory_key="chat_history",
    )
    if not st.session_state.chat:
        retriever = db.as_retriever()
        context = ConversationalRetrievalChain.from_llm(
            llm=llm,
            memory=memory,
            retriever=retriever,
        )
    else:
        return None
    return context


def initialize_session_states():
    if "db" not in st.session_state:
        st.session_state.db = None
    if "context" not in st.session_state:
        st.session_state.context = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


# Display a list of buttons for saved dialogues
def all_dialogues_buttons(name):
    with st.sidebar:
        st.sidebar.subheader("Saved dialogues")
        dialogues = get_all_dialogues(name)
        for dialogue in dialogues:
            if st.button(
                dialogue[0][len(name) + 1 :],
                disabled=not st.session_state.context and not st.session_state.chat,
            ):
                st.session_state.chat_history = load_context_from_sql(dialogue[0])
                st.session_state.context = create_context_from_history(
                    st.session_state.db
                )


def main_page(name):
    load_dotenv()
    initialize_session_states()
    st.header("Chat with PDFs 💬")

    with st.sidebar:
        st.session_state.chat = st.checkbox("using only chatGPT")
        st.session_state.temperature = st.sidebar.slider(
            "temperature", 0.0, 1.0, 0.7, step=0.1
        )
        if not st.session_state.chat:
            st.subheader("Your documents")
            pdfs = st.file_uploader(
                "Upload your PDF",
                accept_multiple_files=True,
                type="pdf",
            )

            st.session_state.promt = st.checkbox("responding outside of the text")
            confirm = st.button("Сonfirm")
            if confirm and pdfs:
                with st.spinner("Processing"):
                    if pdfs is not None:
                        text = get_text_from_docs(pdfs)
                        chunks = get_chunks_from_text(text)
                        st.session_state.db = create_db_vectors(chunks)
                        st.session_state.context = get_context(
                            st.session_state.db, st.session_state.temperature
                        )

    all_dialogues_buttons(name)
    display_dialog(st.session_state.chat_history)
    if query := st.chat_input("Enter a query"):
        with st.chat_message("user"):
            st.markdown(query)
        if not st.session_state.chat:
            dialog_questions(query)
        else:
            only_chat(query)
    if st.session_state.chat_history:
        with st.sidebar:
            save_chat(name)


def generate_ai_response(messages):
    import openai

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        stream=True,
        temperature=st.session_state.temperature,
    )
    return response


def only_chat(query):
    st.session_state.chat_history.append(HumanMessage(content=query))
    messages = [
        {"role": "user", "content": message.content}
        for message in st.session_state.chat_history
    ]
    with st.chat_message("assistant"):
        ai_response_placeholder = st.empty()
        response = generate_ai_response(messages)
        answer = ""
        for chunk in response:
            content = chunk.choices[0].delta.get("content", "")
            answer += content
            ai_response_placeholder.markdown(answer + "▌")
        ai_response_placeholder.markdown(answer)
        st.session_state.chat_history.append(AIMessage(content=answer))


def only_chat_v2(query):
    st.session_state.chat_history.append(HumanMessage(content=query))
    messages = [
        {"role": "user", "content": message.content}
        for message in st.session_state.chat_history
    ]
    completion = openai.ChatCompletion.create(model=MODEL, messages=messages)
    answer = completion.choices[0].message
    st.markdown(answer)
    st.session_state.chat_history.append(AIMessage(content=answer))


def save_chat(name):
    with st.form("id_input", clear_on_submit=True):
        dialogue_id = st.text_input("Save your dialogue")
        save_button = st.form_submit_button("Save")
    if save_button and dialogue_id:
        print(f"{name}_{dialogue_id}")
        if is_valid_name(f"{name}_{dialogue_id}"):
            save_dialog(f"{name}_{dialogue_id}")
            st.success("Successfully saved")
        else:
            st.error("Invalid dialogue name")


def is_valid_name(table_name):
    if table_name[0].isdigit():
        return False
    if not all(c.isalnum() or c == "_" for c in table_name):
        return False
    return True


def create_new_account():
    passwords = []
    st.subheader("Create New Account")
    new_name = st.text_input("Name")
    new_user = st.text_input("Username")
    new_password = st.text_input("Password", type="password")
    passwords.append(new_password)
    hashed_passwords = stauth.Hasher(passwords).generate()
    if st.button("Signup"):
        create_usertable()
        add_userdata(new_name, new_user, hashed_passwords.pop())
        st.success("You have successfully created a valid Account")
        st.info("Go to Home Menu to login")


def make_authenticator():
    create_usertable()
    users = view_all_users()
    usernames = [user[1] for user in users]
    passwords = [user[2] for user in users]
    names = [user[0] for user in users]

    credentials = {"usernames": {}}

    for uname, name, pwd in zip(usernames, names, passwords):
        user_dict = {"name": name, "password": pwd}
        credentials["usernames"].update({uname: user_dict})

    return stauth.Authenticate(
        credentials, "cokkie_name", "random_key", cookie_expiry_days=1
    )


def auth_page():
    st.set_page_config(
        menu_items={
            "About": """
        This app help you analysing your pdf files
        - [IIG AI](https://t.me/+6dN1Eey3-BVkZGNk)
        - Made by huongchi
        """
        },
        page_icon="icon.png",
    )
    st.sidebar.image("icon.png", width=100)
    if "show" not in st.session_state:
        st.session_state.show = True
    if st.session_state.show:
        st.sidebar.title("OpenAI PDF Reader")
        menu = ["Home", "Sign Up"]
        st.session_state.choice = st.sidebar.selectbox("Menu", menu)
    if st.session_state.choice == "Home":
        authenticator = make_authenticator()
        name, authentication_status, username = authenticator.login("Login", "main")
        if authentication_status == False:
            st.error("Username/password is incorrect")

        if authentication_status == None:
            st.warning("Please enter your username and password")
        if authentication_status:
            st.session_state.show = False
            st.sidebar.title(f"Welcome home, {name.title()}")
            authenticator.logout("Logout", "sidebar")
            main_page(username)

    elif st.session_state.choice == "Sign Up":
        create_new_account()


def get_text_from_docs(docs):
    raw_text = ""
    for doc in docs:
        pdf_reader = PdfReader(doc)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()

    return raw_text


def get_chunks_from_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    return chunks


def get_context(db, temperature):
    llm = ChatOpenAI(
        model=MODEL,
        temperature=temperature,
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    template_inside = """
    CONTEXT: {context}
    You are a helpful assistant, above is some context, 
    Please answer the question, and make sure you follow ALL of the rules below:
    1. Answer the questions only based on context provided, do not make things up
    2. Answer questions in a helpful manner that straight to the point, with clear structure & all relevant information that might help users answer the question
    3. Anwser should be formatted in Markdown
    4. If there are relevant images, video, links, they are very important reference data, please include them as part of the answer

    QUESTION: {question}
    ANSWER (formatted in Markdown):
    """
    template_outside = """ 
    CONTEXT: {context} 
    You are a helpful assistant, above is some context, 
    Please answer the question, and make sure you follow ALL of the rules below: 
    1. Answer the questions using both the context provided and your own knowledge base, but prioritize the information from the context 
    2. Answer questions in a helpful manner that is straight to the point, with clear structure & all relevant information that might help users answer the question 
    3. Anwser should be formatted in Markdown 
    4. If there are relevant images, video, links, they are very important reference data, please include them as part of the answer 
    
    QUESTION: {question} 
    ANSWER (formatted in Markdown): """
    if st.session_state.promt:
        combine_docs_custom_prompt = PromptTemplate.from_template(template_outside)
    else:
        combine_docs_custom_prompt = PromptTemplate.from_template(template_inside)

    context = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(),
        memory=memory,
        # condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs=dict(prompt=combine_docs_custom_prompt),
    )

    return context


def create_db_vectors(chunks):
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(chunks, embedding=embeddings)
    return db


def dialog_questions(query):
    response = st.session_state.context({"question": query})
    ai_response = response["chat_history"][-1].content
    st.session_state.chat_history.extend([HumanMessage(content=query)])

    with st.chat_message("assistant"):
        ai_response_placeholder = st.empty()
        ai_response_placeholder.markdown(ai_response)

    st.session_state.chat_history.extend([AIMessage(content=ai_response)])


def display_dialog(history):
    for message in history:
        if type(message) == HumanMessage:
            role = "user"
        else:
            role = "assistant"
        with st.chat_message(role):
            st.markdown(message.content)


def main():
    auth_page()


if __name__ == "__main__":
    main()
