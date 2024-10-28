import os
import yaml
from time import time
import uuid

#from langchain.callbacks import get_openai_callback
from langchain_community.callbacks.manager import get_openai_callback
from commons.utils import read_text
from rags.rag import build_retriever, build_chain, build_rag, parse_evaluation
from database import db

#In this demo we will explore using RetirvalQA chain to retrieve relevant documents and send these as a context in a query.
# We will use Chroma vectorstore.



# get the current working directory
current_working_directory = os.getcwd()

# print output to the console
print(f"current_working_directory={current_working_directory}")
print(f"OPENAI_API_KEY={os.environ['OPENAI_API_KEY']}")

def read_conf(env_deploy):
    config_file = 'config.yaml'
    if env_deploy:
        config_file = f'config-{env_deploy}.yaml'
    print("="*50)
    print(f"= config_file: {config_file}")
    print("="*50)
    with open(f'config/streamlit/{config_file}', 'r') as file:
        config = yaml.safe_load(file)
    return config

config = read_conf(os.getenv('ENV_DEPLOYMENT'))

print(config)


# Prompts

entry_template = read_text(config["prompts"]["entry_template"])
prompt_template_rewrite = read_text(config["prompts"]["prompt_template_rewrite"])
prompt_template_rag = read_text(config["prompts"]["prompt_template_rag"])
prompt_template_judge = read_text(config["prompts"]["prompt_template_judge"])
#system_prompt_template = read_text(config["prompts"]["system_template_path"])
#human_prompt_template = read_text(config["prompts"]["human_template_path"])


ai_avatar="🤖"
user_avatar="🦖"

#Step 1 - this will set up chain , to be called later
# Función para construir el contexto de los resultados de búsqueda


#Step 1 - this will set up chain , to be called later
retriever = build_retriever(config["elasticsearch"], entry_template)

#Step 2 - create chain, here we create a ConversationalRetrievalChain.
generator_params = config["rag"]["generator"]["params"]
model_used = generator_params["model"]
print(f"model_used: {model_used}")
rag_chain = build_rag(
    template=prompt_template_rag,
    gen_params=generator_params,
    retriever=retriever
)

judge_params = config["rag"]["judge"]["params"]
judge_chain = build_chain(
    generator_params, 
    prompt_template_judge
)

def handle_question(question):
    # Run RAG
    t0 = time()
    with get_openai_callback() as cb: 
        response = rag_chain.invoke(user_input)
    token_stats = {
        "prompt_tokens": cb.prompt_tokens,
        "completion_tokens": cb.completion_tokens,
        "total_tokens": cb.total_tokens,
        "total_cost": cb.total_cost
    }

    # Run judge
    with get_openai_callback() as cb:
        evaluation = judge_chain.invoke({"question": question, "answer_llm": response})
    eval_token_stats = {
        "eval_prompt_tokens": cb.prompt_tokens,
        "eval_completion_tokens": cb.completion_tokens,
        "eval_total_tokens": cb.total_tokens,
        "eval_total_cost": cb.total_cost
    }
    relevance_json = parse_evaluation(evaluation)
    t1 = time()
    t_response = t1 - t0
    

    # Merge
    openai_cost = token_stats["total_cost"] + eval_token_stats["eval_total_cost"]
    token_stats.pop("total_cost")
    eval_token_stats.pop("eval_total_cost")
    answer_data = token_stats | eval_token_stats
    answer_data["answer"] = response
    answer_data["model_used"] = model_used
    answer_data["response_time"] = t_response
    answer_data["relevance"] = relevance_json["relevance"]
    answer_data["relevance_explanation"] = relevance_json["explanation"]
    answer_data["openai_cost"] = openai_cost
    
    # Save database
    conversation_id = str(uuid.uuid4())
    db.save_conversation(
        conversation_id=conversation_id,
        question=user_input,
        answer_data=answer_data,
    )

    return response, conversation_id


def handle_feedback(
    conversation_id,
    feedback
):
    db.save_feedback(
        conversation_id=conversation_id,
        feedback=feedback,
    )



#Step 4 - here we setup Streamlit text input and pass input text to chat function.
# chat function returns the response and we print it.

if __name__ == "__main__":
    import streamlit as st
    from streamlit_feedback import streamlit_feedback

    st.set_page_config(page_title="🤗💬 Cooking Recipe Chatbot")
    # Title
    st.subheader("Zoomcamp LLM: Cooking Chatbot powered by Emmuzoo")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "feedback" not in st.session_state:
        st.session_state.feedback = None
    if "current_response" not in st.session_state:
        st.session_state.current_response = None
    if "feedback_radio" in st.session_state:
        print("="*40)
        print("= Session")
        feedback = st.session_state.feedback_radio
        conversation_id = st.session_state.conversation_id
        if feedback and conversation_id:
            feedback_value = 1 if feedback == "👍 (Positive)" else -1
            handle_feedback(conversation_id, feedback_value)
            print(f"{feedback} => {feedback_value}")
        print("="*40)
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        avatar = ai_avatar if message["role"] == "assistant" else user_avatar
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Display hello message
    with st.chat_message("assistant", avatar=ai_avatar):
        st.write("I am a cooking recipe assistant, how can I help you?")

    # Accept user input
    if user_input := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message in chat message container
        with st.chat_message("user", avatar=user_avatar):
            st.markdown(user_input)

        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar=ai_avatar):
            #conversation_id = str(uuid.uuid4())
            #response = f'Esto es un prueba: {conversation_id}'
            response, conversation_id = handle_question(user_input)
            # Write reponses+
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.conversation_id = conversation_id

        # Crear un contenedor para el feedback
        feedback_container = st.empty()  # Creamos un contenedor vacío

        with feedback_container.container():
            # Mostrar el radio solo si hay una respuesta actual
            feedback = st.radio(
                "How would you rate this response?",
                options=["👍 (Positive)", "👎 (Negative)", "Pass (Skip feedback)"],
                index=None,
                horizontal=True,
                key="feedback_radio"
            )