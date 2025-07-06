from bs4 import BeautifulSoup
from faiss import IndexFlatL2
from langchain.document_transformers import LongContextReorder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain.schema.runnable.passthrough import RunnableAssign
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
import logging
from operator import itemgetter
import os
from dotenv import load_dotenv
import requests


EMBEDDING_MODEL_NAME = 'nvidia/nv-embed-v1'
CHAT_MODEL_NAME = 'meta/llama-3.3-70b-instruct'
SECTION_DIVIDER = '-' * 50 + '\n\n'
TARGET_HEADERS = [
    'Characteristics', 'Appearance', 'Personality', 'Character Relationships',
    'Story', 'Gameplay']
SF_CHAR_WIKI_PAGES = {
    'Ryu': 'https://streetfighter.fandom.com/wiki/Ryu',
    'Dhalsim': 'https://streetfighter.fandom.com/wiki/Dhalsim',
    'Chun-Li': 'https://streetfighter.fandom.com/wiki/Chun-Li',
    'M. Bison': 'https://streetfighter.fandom.com/wiki/M._Bison',
    'Ken': 'https://streetfighter.fandom.com/wiki/Ken_Masters',
    'Guile': 'https://streetfighter.fandom.com/wiki/Guile',
    'Sakura': 'https://streetfighter.fandom.com/wiki/Sakura',
    'Blanka': 'https://streetfighter.fandom.com/wiki/Blanka',
    'E. Honda': 'https://streetfighter.fandom.com/wiki/E._Honda',
    'Akuma': 'https://streetfighter.fandom.com/wiki/Akuma',
    'Cammy': 'https://streetfighter.fandom.com/wiki/Cammy',
    'Zangief': 'https://streetfighter.fandom.com/wiki/Zangief',
    'Dan': 'https://streetfighter.fandom.com/wiki/Dan',
    'Akira': 'https://streetfighter.fandom.com/wiki/Akira',
    'Balrog': 'https://streetfighter.fandom.com/wiki/Balrog',
    'Sagat': 'https://streetfighter.fandom.com/wiki/Sagat',
    'Vega': 'https://streetfighter.fandom.com/wiki/Vega',
    'Fei Long': 'https://streetfighter.fandom.com/wiki/Fei_Long',
    'Gouken': 'https://streetfighter.fandom.com/wiki/Gouken',
    'Luke': 'https://streetfighter.fandom.com/wiki/Luke'
}
CONV_HISTORY_LIMIT = 20  # must be even number to account for user input and
# StreetChatter output
NUM_DOCS_TO_RETRIEVE = 10


class StreetChatter:
    def __init__(self, character_name=None):
        # check character name
        valid_char_names = set(SF_CHAR_WIKI_PAGES.keys())
        if character_name not in valid_char_names:
            raise Exception('Character name must be one of the following: '
                            + f'{valid_char_names}')
        self.character_name = character_name

        # get logger
        self.logger = logging.getLogger('StreetChatter_Logger')
        self.logger.setLevel(logging.INFO)

        # load environment variables
        if os.path.isfile('./.env'):
            load_dotenv()
        self.logger.info('Loaded environment variables from .env file.')

        # specify embedding model and chat model
        embedding_model = NVIDIAEmbeddings(model=EMBEDDING_MODEL_NAME)
        chat_model = ChatNVIDIA(model=CHAT_MODEL_NAME, temperature=0)
        self.logger.info(f'Using {EMBEDDING_MODEL_NAME} embedding model.')
        self.logger.info(f'Using {CHAT_MODEL_NAME} chat model.')

        # utility function to get Street Fighter character context from Street
        # Fighter Wiki via webscraping
        def get_sf_char_context():
            # get Street Fighter Wiki page from character name
            sf_char_wiki_page = SF_CHAR_WIKI_PAGES[self.character_name]

            # get HTML of webpage as text, get header and paragraph elements
            response = requests.get(sf_char_wiki_page)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            header_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
            elements = soup.find_all(header_tags + ['p'])

            # utility function for adding breadcrumbs path
            def _add_breadcrumbs_to_context(context, breadcrumbs):
                # add breadcrumbs path (filter out empty breadcrumbs, which
                # account for skipped header levels)
                path = ' > '.join(
                    list(filter(lambda b: len(b) > 0, breadcrumbs)))
                context += f'* {path}\n\n'
                return context

            # utility function for updating breadcrumbs with nested header
            def _update_breadcrumbs_with_nested_header(
                    breadcrumbs, header_text, header_level,
                    current_base_level):
                # add header text to breadcrumbs (add empty breadcrumbs for
                # skipped header levels)
                relative_level = header_level - current_base_level
                while len(breadcrumbs) < relative_level + 1:
                    breadcrumbs.append('')
                breadcrumbs[relative_level] = header_text
                breadcrumbs = breadcrumbs[:relative_level + 1]
                return breadcrumbs

            # loop through header and paragraph elements, add contents to
            # context with breadcrumbs path and dividers between sections
            context = ''
            breadcrumbs = []  # for tracking section hierarchy
            collecting = False  # True when adding content in/under a target
            # header
            current_base_level = None  # level number of top-level header
            first_section = True  # for skipping divider before first section
            i = 0
            while i < len(elements):
                el = elements[i]

                # if header, add section divider and breadcrumbs title
                if el.name in header_tags:
                    header_text = el.get_text(strip=True)
                    header_level = int(el.name[1])
                    if header_text in TARGET_HEADERS:
                        if not collecting or header_level <= \
                                current_base_level:
                            # new top-level target header
                            collecting = True
                            current_base_level = header_level
                            breadcrumbs = [header_text]  # reset breadcrumbs
                            # to current header

                            # add divider between sections (not before the
                            # first one)
                            if not first_section:
                                context += SECTION_DIVIDER
                            first_section = False
                        else:
                            # update breadcrumbs with nested target header
                            breadcrumbs = \
                                _update_breadcrumbs_with_nested_header(
                                    breadcrumbs, header_text, header_level,
                                    current_base_level)

                            # add section divider
                            context += SECTION_DIVIDER

                        # add breadcrumbs path
                        context = _add_breadcrumbs_to_context(
                            context, breadcrumbs)
                    elif collecting:
                        if header_level <= current_base_level:
                            # reached header outside current target section
                            collecting = False
                            current_base_level = None
                            i -= 1  # re-process this header in next loop
                        else:
                            # update breadcrumbs with nested header
                            breadcrumbs = \
                                _update_breadcrumbs_with_nested_header(
                                    breadcrumbs, header_text, header_level,
                                    current_base_level)

                            # add section divider
                            context += SECTION_DIVIDER

                            # add breadcrumbs path
                            context = _add_breadcrumbs_to_context(
                                context, breadcrumbs)

                # if paragraph, add paragraph text
                elif el.name == 'p' and collecting:
                    paragraph_text = el.get_text(strip=True)
                    if paragraph_text:
                        context += paragraph_text + '\n\n'

                i += 1

            # return context
            return context.strip()

        # set character name, get character context
        sf_char_context = get_sf_char_context()
        self.logger.info(
            f'Obtained context for {self.character_name} from Street Fighter Wiki.')

        # get chunker
        chunker = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=100,
            separators=['\n\n', '\n', '.', '!', '?', ''])

        # split context into documents with metadata
        context_docs = []
        for context_split in sf_char_context.split(SECTION_DIVIDER):
            # extract topic (section name) and text
            topic_and_text = context_split.split('\n\n', maxsplit=1)
            topic = topic_and_text[0].replace('* ', '')
            text = topic_and_text[1]

            # skip context that includes only the section name (topic) and no
            # other text
            if len(text.strip()) == 0:
                continue

            # split text into chunks and and add topic to text and metadata of
            # each chunk
            text_chunks = chunker.split_text(text)
            for text_chunk in text_chunks:
                context_doc = Document(
                    page_content=f'[Topic: {topic}] {text_chunk}',
                    metadata={'topic': topic})
                context_docs.append(context_doc)

        # create in-memory document store, that stores the embeddings (of
        # the context documents/chunks) generated by the embedding model
        # similarity is calculated via Euclidean distance (L2 norm)
        embed_dims = len(
            embedding_model.embed_query('lorem ipsum dolor'))
        doc_store = FAISS(
            embedding_function=embedding_model,
            index=IndexFlatL2(embed_dims),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            normalize_L2=False)
        self.logger.info('Created FAISS document store.')

        # load context into document store
        doc_store.merge_from(
            FAISS.from_documents(context_docs, embedding_model))
        self.logger.info('Loaded context into FAISS document store.')

        # specify prompt template
        self.conv_history = []
        role_text = f'''
            You are a chatbot that assumes the role of {self.character_name}, a
            character from the Street Figher videogame series. You will answer
            the user's questions about {self.character_name} and their lore,
            from the perspective of {self.character_name}.
        '''
        sys_prompt = role_text + '''
            The user asked: {input}.

            Relevant context retrieved from Street Fighter Wiki:
            ----------------------------------------------------
            {context}
            ----------------------------------------------------

            Using only the retrieved context and the conversation history,
            answer the user's question conversationally, from the point of
            view of the Street Fighter character whose persona you have
            assumed.
        '''
        prompt_template = ChatPromptTemplate.from_messages(
            self.conv_history + [('system', sys_prompt), ('user', '{input}')])

        # utility function for logging
        def _log_message(input, message):
            self.logger.info(message)
            return input

        # utility function for getting context string
        def _get_context_str(docs):
            return '\n\n'.join(list(map(lambda d: d.page_content, docs)))

        # create retrieval augmented generation (RAG) chain
        long_reorder = RunnableLambda(LongContextReorder().transform_documents)
        retrieval_chain = (
            {'input': (lambda x: x)}
            | RunnableAssign({
                'context': itemgetter('input')
                | doc_store.as_retriever(
                    search_kwargs={'k': NUM_DOCS_TO_RETRIEVE})
                | long_reorder
                | RunnableLambda(_get_context_str)
                | RunnableLambda(lambda x: _log_message(
                    x, 'Retrieved relevant context from document store.'))}))
        generation_chain = (
            RunnableLambda(lambda x: x)
            | RunnableAssign({'output': prompt_template | chat_model})
            | RunnableLambda(lambda x: _log_message(
                    x, 'Generated response to user input with retrieved '
                    + 'context.'))
            | RunnableLambda(lambda x: x['output'])
            | StrOutputParser())
        self.rag_chain = retrieval_chain | generation_chain

    def invoke(self, prompt):
        # get StreetChatter response
        response = self.rag_chain.invoke(prompt)

        # update conversation history
        self.conv_history += [('user', prompt), ('assistant', response)]
        self.logger.info('Updated conversation history with user prompt and '
                         + 'StreetChatter response')

        # return StreetChatter response
        return response
