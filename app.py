import torch

from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory.buffer_window import ConversationBufferWindowMemory
from langchain_community.llms import HuggingFaceHub
from langchain.chains import (
    LLMChain,
    StuffDocumentsChain,
    MapReduceDocumentsChain,
    ReduceDocumentsChain,
)

from gradio_client import Client
import gradio as gr
import yt_dlp
import json
import gc
import datetime
import os
import numpy as np


"""Prepare data"""

whisper_jax_api = "https://sanchit-gandhi-whisper-jax.hf.space/"
whisper_jax = Client(whisper_jax_api)


def transcribe_audio(audio_path, task="transcribe", return_timestamps=True) -> str:
    text, runtime = whisper_jax.predict(
        audio_path,
        task,
        return_timestamps,
        api_name="/predict_1",
    )
    return text

def format_whisper_jax_output(
    whisper_jax_output: str, max_duration: int = 60
) -> list[dict]:
    """Whisper JAX outputs are in the format
    '[00:00.000 -> 00:00.000] text\n[00:00.000 -> 00:00.000] text'.

    Returns a list of dict with keys 'start', 'end', 'text'
    The segments from whisper jax output are merged to form paragraphs.

    `max_duration` controls how many seconds of the audio's transcripts are merged

    For example, if `max_duration`=60, in the final output, each segment is roughly
    60 seconds.
    """

    final_output = []
    max_duration = datetime.timedelta(seconds=max_duration)
    segments = whisper_jax_output.split("\n")
    current_start = datetime.datetime.strptime("00:00", "%M:%S")
    current_text = ""

    for i, seg in enumerate(segments):
        text = seg.split("]")[-1].strip()
        current_text += " " + text

        # Sometimes whisper jax returns None for timestamp
        try:
            end = datetime.datetime.strptime(seg[14:19], "%M:%S")
        except ValueError:
            end = current_start + max_duration

        if i == len(segments) - 1:
            final_output.append(
                {
                    "start": current_start.strftime("%H:%M:%S"),
                    "end": end.strftime("%H:%M:%S"),
                    "text": current_text.strip(),
                }
            )

        else:
            if end - current_start >= max_duration and current_text[-1] == ".":
                # If we have exceeded max duration, check whether we have
                # reached the end of a sentence. If not, keep merging.
                final_output.append(
                    {
                        "start": current_start.strftime("%H:%M:%S"),
                        "end": end.strftime("%H:%M:%S"),
                        "text": current_text.strip(),
                    }
                )

                # Update current start and text
                current_start = end
                current_text = ""

    return final_output

def yt_audio_to_text(url: str, max_duration: int = 60):
    """Given a YouTube url, download audio and transcribe it to text. Reformat
    the output from Whisper JAX and save the final result in a json file.
    """

    progress = gr.Progress()
    progress(0.1)

    with yt_dlp.YoutubeDL(
        {"extract_audio": True, "format": "bestaudio", "outtmpl": "audio.mp3"}
    ) as video:
        info_dict = video.extract_info(url, download=False)
        global video_title
        video_title = info_dict["title"]
        video.download(url)

    progress(0.4)
    audio_file = "audio.mp3"

    result = transcribe_audio(audio_file, return_timestamps=True)
    progress(0.7)

    result = format_whisper_jax_output(result, max_duration=max_duration)
    progress(0.9)

    with open("audio.json", "w") as f:
        json.dump(result, f)

    os.remove(audio_file)




"""Load data"""

def metadata_func(record: dict, metadata: dict) -> dict:
    """This function is used to tell the Langchain loader the keys that
    contain metadata and extract them.
    """
    metadata["start"] = record.get("start")
    metadata["end"] = record.get("end")
    metadata["source"] = metadata["start"] + " -> " + metadata["end"]

    return metadata


def load_data():
    loader = JSONLoader(
        file_path="audio.json",
        jq_schema=".[]",
        content_key="text",
        metadata_func=metadata_func,
    )

    data = loader.load()
    os.remove("audio.json")

    return data




"""Create embeddings and vector store"""

embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model_kwargs = {"device": device}

embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name, model_kwargs=embedding_model_kwargs
)


def create_vectordb(data, n_retrieved_docs: int, collection_name="YouTube"):
    """Returns a retriever which is used to fetch relevant documents from
    the vector database.

    `n_retrieved_docs` is the number of retrieved documents.
    """

    vectordb = Chroma.from_documents(
        documents=data, embedding=embeddings, collection_name=collection_name
    )
    n_docs = len(vectordb.get()["ids"])
    retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={"k": n_retrieved_docs, "fetch_k": n_docs}
    )

    return retriever




"""Load LLM"""

repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"max_new_tokens": 1000})




"""Summarisation"""

# Map
map_template = """Summarise the following text:
{docs}

Answer:"""
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt)

# Reduce
reduce_template = """The following is a set of summaries:
{docs}

Take these and distill it into a final, consolidated summary of the main themes \
in 150 words or less.

Answer:"""

reduce_prompt = PromptTemplate.from_template(reduce_template)
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# Takes a list of documents, combines them into a single string, and passes this to llm
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="docs"
)


# Combines and iteravely reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called.
    combine_documents_chain=combine_documents_chain,
    # If documents exceed context for `StuffDocumentsChain`
    collapse_documents_chain=combine_documents_chain,
    # The maximum number of tokens to group documents into.
    token_max=4000,
)


# Combining documents by mapping a chain over them, then combining results
map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="docs",
    # Return the results of the map steps in the output
    return_intermediate_steps=False,
)


def get_summary(documents) -> str:
    summary = map_reduce_chain.invoke(documents, return_only_outputs=True)
    return summary["output_text"].strip()



"""Contextualising the question"""

contextualise_q_prompt = PromptTemplate.from_template(
    """Given a chat history and the latest user question \
    which might reference the chat history, formulate a \
    standalone question that can be understood without \
    the chat history. Do NOT answer the question, just \
    reformulate it if needed and otherwise return it as is.

    Chat history: {chat_history}

    Question: {question}

    Answer:
    """
)

contextualise_q_chain = contextualise_q_prompt | llm



"""Standalone question chain"""

standalone_prompt = PromptTemplate.from_template(
    """Given a chat history and the latest user question, \
    identify whether the question is a standalone question \
    or the question references the chat history. Answer 'yes' \
    if the question is a standalone question, and 'no' if the \
    question references the chat history. Do not answer \
    anything other than 'yes' or 'no'.

    Chat history:
    {chat_history}

    Question:
    {question}

    Answer:
    """
)


def format_output(answer: str) -> str:
    """All lower case and remove all whitespace to ensure
    that the answer given by the LLM is either 'yes' or 'no'.
    """
    return "".join(answer.lower().split())


standalone_chain = standalone_prompt | llm | format_output



"""Q&A chain"""

qa_prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. \
    ONLY use the following context to answer the question. \
    Do NOT answer with information that is not contained in \
    the context. If you don't know the answer, just say:\
    "Sorry, I cannot find the answer to that question in the video."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)

class YouTubeChatbot:
    instance_count = 0

    def __init__(
        self,
        n_sources: int = 3,
        n_retrieved_docs: int = 5,
        timestamp_interval: datetime.timedelta = datetime.timedelta(minutes=2),
        memory: int = 5,
    ):
        YouTubeChatbot.instance_count += 1
        self.chatbot_id = YouTubeChatbot.instance_count
        self.n_sources = n_sources
        self.n_retrieved_docs = n_retrieved_docs
        self.timestamp_interval = timestamp_interval
        self.chat_history = ConversationBufferWindowMemory(k=memory)
        self.retriever = None
        self.qa_chain = None


    def format_docs(self, docs: list) -> str:
        """Combine documents into a single string which will be included
        in the prompt given to the LLM.
        """
        self.sources = [doc.metadata["start"] for doc in docs]

        return "\n\n".join(doc.page_content for doc in docs)


    def standalone_question(self, input_: dict) -> str:
        """If the question is a not a standalone question,
        run contextualise_q_chain.
        """
        if input_["standalone"] == "yes":
            return contextualise_q_chain
        else:
            return input_["question"]


    def format_answer(self, answer: str) -> str:
        """Add timestamps to answers.
        """
        if "cannot find the answer" in answer:
            return answer.strip()
        else:
            timestamps = self.filter_timestamps()
            answer_with_sources = (
                answer.strip() + " You can find more information "
                "at these timestamps: {}.".format(", ".join(timestamps))
            )
            return answer_with_sources


    def filter_timestamps(self) -> list[str]:
        """Returns a list of timestamps with length less or
        equal to `n_sources`. The timestamps are at least an
        `timestamp_interval` apart. This prevents returning
        a list of timestamps that are too close together.
        """

        filtered_timestamps = np.array(
            [datetime.datetime.strptime(self.sources[0], "%H:%M:%S")]
        )

        i = 1

        while len(filtered_timestamps) < self.n_sources:
            try:
                new_timestamp = datetime.datetime.strptime(self.sources[i], "%H:%M:%S")
            except IndexError:
                break

            absolute_time_difference = abs(new_timestamp - filtered_timestamps)

            if all(absolute_time_difference >= self.timestamp_interval):
                filtered_timestamps = np.append(filtered_timestamps, new_timestamp)

            i += 1

        filtered_timestamps = [
            timestamp.strftime("%H:%M:%S") for timestamp in filtered_timestamps
        ]
        filtered_timestamps.sort()

        return filtered_timestamps


    def process_video(self, url: str, data=None, retriever=None):
        """Given a YouTube URL, transcribe YouTube audio to text.
        Then set up the vector database.
        """
        yt_audio_to_text(url)
        data = load_data()

        if retriever is not None:
            # If we already have documents in the vector store, delete them.
            ids = retriever.vectorstore.get()["ids"]
            retriever.vectorstore.delete(ids)

        retriever = create_vectordb(
            data, self.n_retrieved_docs,
            collection_name=f"Chatbot{self.chatbot_id}"
        )

        return url, data, retriever


    def setup_qa_chain(self, retriever, qa_chain=None):
        qa_chain = (
            RunnablePassthrough.assign(standalone=standalone_chain)
            | {
                "question": self.standalone_question,
                "context": self.standalone_question | retriever | self.format_docs,
            }
            | qa_prompt
            | llm
        )

        return retriever, qa_chain


    def setup_chatbot(self, url: str):
        _, _, self.retriever = self.process_video(url=url, retriever=self.retriever)
        _, self.qa_chain = self.setup_qa_chain(retriever=self.retriever)


    def get_answer(self, question: str) -> str:
        try:
            ai_msg = self.qa_chain.invoke(
                {"question": question, "chat_history": self.chat_history}
            )
        except AttributeError:
            raise AttributeError(
                "You haven't setup the chatbot yet. "
                "Setup the chatbot by calling the "
                "instance method `setup_chatbot`."
            )

        self.chat_history.save_context({"question": question}, {"answer": ai_msg})

        answer = self.format_answer(ai_msg)

        return answer



"""Web app"""

class YouTubeChatbotApp(YouTubeChatbot):
    def __init__(
        self,
        n_sources: int,
        n_retrieved_docs: int,
        timestamp_interval: datetime.timedelta,
        memory: int,
        default_youtube_url: str,
    ):
        super().__init__(n_sources, n_retrieved_docs, timestamp_interval, memory)

        self.default_youtube_url = default_youtube_url
        self.memory = memory
        self.chat_history = None
        self.data = None
        self.retriever = None
        self.qa_chain = None

        # Gradio components
        self.url_input = None
        self.url_button = None
        self.app_chat_history = None
        self.chatbot = None
        self.user_input = None
        self.clear_button = None

    def greet(self, data, app_chat_history) -> dict:
        """Summarise the video and greet the user.
        """
        summary_message = f'Here is a summary of the video "{video_title}":'
        app_chat_history.append((None, summary_message))

        summary = get_summary(data)
        self.data = gr.State(None)
        app_chat_history.append((None, summary))

        greeting_message = (
            "You can ask me anything about the video. " "I will do my best to answer!"
        )
        app_chat_history.append((None, greeting_message))

        return {self.app_chat_history: app_chat_history, self.chatbot: app_chat_history}

    def question(self, user_question: str, app_chat_history) -> dict:
        """Display the question asked by the user in the chat window,
        and delete from the input textbox.
        """
        app_chat_history.append((user_question, None))
        return {
            self.user_input: "",
            self.app_chat_history: app_chat_history,
            self.chatbot: app_chat_history,
        }

    def respond(self, qa_chain, chat_history, app_chat_history) -> dict:
        """Respond to user's latest question"""
        question = app_chat_history[-1][0]

        try:
            ai_msg = qa_chain.invoke(
                {"question": question, "chat_history": chat_history}
            )
        except AttributeError:
            raise gr.Error(
                "You need to process the video " "first by pressing the `Go` button."
            )

        chat_history.save_context({"question": question}, {"answer": ai_msg})

        answer = self.format_answer(ai_msg)

        app_chat_history.append((None, answer))

        return {
            self.qa_chain: qa_chain,
            self.chat_history: chat_history,
            self.app_chat_history: app_chat_history,
            self.chatbot: app_chat_history,
        }

    def clear_chat_history(self, chat_history, app_chat_history):
        chat_history.clear()
        app_chat_history = []
        return {
            self.chat_history: chat_history,
            self.app_chat_history: app_chat_history,
            self.chatbot: app_chat_history,
        }

    def launch(self, **kwargs):
        with gr.Blocks() as demo:
            self.chat_history = gr.State(ConversationBufferWindowMemory(k=self.memory))
            self.app_chat_history = gr.State([])
            self.data = gr.State()
            self.retriever = gr.State()
            self.qa_chain = gr.State()

            # App structure
            with gr.Row():
                self.url_input = gr.Textbox(
                    value=self.default_youtube_url, label="YouTube URL", scale=5
                )
                self.url_button = gr.Button(value="Go", scale=1)

            self.chatbot = gr.Chatbot()
            self.user_input = gr.Textbox(label="Ask a question:")
            self.clear_button = gr.Button(value="Clear")


            # App actions

            # When a new url is given, clear past chat history and process
            # the new video. Set up the Q&A chain with the new video's data.
            # Provide a summary of the new video.
            self.url_button.click(
                self.clear_chat_history,
                inputs=[self.chat_history, self.app_chat_history],
                outputs=[self.chat_history, self.app_chat_history, self.chatbot],
                trigger_mode="once",
            ).then(
                self.process_video,
                inputs=[self.url_input, self.data, self.retriever],
                outputs=[self.url_input, self.data, self.retriever],
            ).then(
                self.setup_qa_chain,
                inputs=[self.retriever, self.qa_chain],
                outputs=[self.retriever, self.qa_chain],
            ).then(
                self.greet,
                inputs=[self.data, self.app_chat_history],
                outputs=[self.app_chat_history, self.chatbot],
            )

            # When a user asks a question, display the question in the chat
            # window and remove it from the text input area. Then respond
            # with the Q&A chain.
            self.user_input.submit(
                self.question,
                inputs=[self.user_input, self.app_chat_history],
                outputs=[self.user_input, self.app_chat_history, self.chatbot],
                queue=False,
            ).then(
                self.respond,
                inputs=[self.qa_chain, self.chat_history, self.app_chat_history],
                outputs=[
                    self.qa_chain,
                    self.chat_history,
                    self.app_chat_history,
                    self.chatbot,
                ],
            )

            # When the `Clear` button is clicked, clear the chat history from
            # the chat window.
            self.clear_button.click(
                self.clear_chat_history,
                inputs=[self.chat_history, self.app_chat_history],
                outputs=[self.chat_history, self.app_chat_history, self.chatbot],
                queue=False,
            )

        demo.launch(**kwargs)


if __name__ == "__main__":
    app = YouTubeChatbotApp(
        n_sources=3,
        n_retrieved_docs=5,
        timestamp_interval=datetime.timedelta(minutes=2),
        memory=5,
        default_youtube_url="https://www.youtube.com/watch?v=SZorAJ4I-sA",
    )

    app.launch()

