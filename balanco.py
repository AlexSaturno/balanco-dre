# PASTAS NECESSÁRIAS:
# avaliacao
# uploaded_files
# vectordb
################################################################################################################################
# Bibliotecas
################################################################################################################################
from langchain_openai import AzureChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.chains import TransformChain
from langchain_core.messages import HumanMessage
from langchain_core.runnables import chain

from datetime import datetime, timedelta
import os
import time
import json
import tiktoken
import streamlit as st
import numpy as np
import pandas as pd
from unidecode import unidecode
from time import sleep

import io
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph

import base64

from utils import *

################################################################################################################################
# Ambiente
################################################################################################################################

# Parametros das APIS
# arquivo de secrets

llm = AzureChatOpenAI(
    azure_deployment=st.secrets["AZURE_OPENAI_DEPLOYMENT"],
    model=st.secrets["AZURE_OPENAI_MODEL"],
    azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
    api_version=st.secrets["AZURE_OPENAI_API_VERSION"],
    api_key=st.secrets["AZURE_OPENAI_API_KEY"],
    openai_api_type="azure",
)


###############################################################################
# Conversão de imagens para base64 para enviar para o modelo
def load_images(inputs: dict) -> dict:
    """Load multiple images from files and encode them as base64."""
    image_paths = inputs["image_paths"]

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    images_base64 = [encode_image(path) for path in image_paths]
    return {"images": images_base64}


load_images_chain = TransformChain(
    input_variables=["image_paths"], output_variables=["images"], transform=load_images
)


@chain
def image_model(
    inputs: dict,
) -> str | list[str] | dict:
    """Invoke model with images and prompt."""
    image_urls = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
        for img in inputs["images"]
    ]

    content = [{"type": "text", "text": inputs["prompt"]}] + image_urls

    msg = llm.invoke([HumanMessage(content=content)])
    # print("msg: ", msg)
    # return {"response": str(msg.content), "images_base64": inputs["images"]}
    return str(msg.content)


chain = load_images_chain | image_model
###############################################################################


# Funcoes auxiliares
def normalize_filename(filename):
    # Mapeamento de caracteres acentuados para não acentuados
    substitutions = {
        "á": "a",
        "à": "a",
        "ã": "a",
        "â": "a",
        "ä": "a",
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "í": "i",
        "ì": "i",
        "î": "i",
        "ï": "i",
        "ó": "o",
        "ò": "o",
        "õ": "o",
        "ô": "o",
        "ö": "o",
        "ú": "u",
        "ù": "u",
        "û": "u",
        "ü": "u",
        "ç": "c",
        "Á": "A",
        "À": "A",
        "Ã": "A",
        "Â": "A",
        "Ä": "A",
        "É": "E",
        "È": "E",
        "Ê": "E",
        "Ë": "E",
        "Í": "I",
        "Ì": "I",
        "Î": "I",
        "Ï": "I",
        "Ó": "O",
        "Ò": "O",
        "Õ": "O",
        "Ô": "O",
        "Ö": "O",
        "Ú": "U",
        "Ù": "U",
        "Û": "U",
        "Ü": "U",
        "Ç": "C",
    }

    # Substitui caracteres especiais conforme o dicionário
    normalized_filename = "".join(substitutions.get(c, c) for c in filename)

    # Remove caracteres não-ASCII
    ascii_filename = normalized_filename.encode("ASCII", "ignore").decode("ASCII")

    # Substitui espaços por underscores
    safe_filename = ascii_filename.replace(" ", "_")

    return safe_filename


def clear_respostas():
    st.session_state["clear_respostas"] = True
    st.session_state["Q&A_done"] = False
    st.session_state["Q&A"] = {}
    st.session_state["Q&A_downloadable"] = {}
    st.session_state["data_processamento"] = None
    st.session_state["hora_processamento"] = None
    st.session_state["tempo_ia"] = None
    st.session_state["answer_downloads"] = False


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def zera_vetorizacao():
    st.session_state["vectordb_object"] = None
    st.session_state["status_vetorizacao"] = False
    st.session_state["clear_respostas"] = True
    st.session_state["Q&A_done"] = False
    st.session_state["Q&A"] = {}
    st.session_state["Q&A_downloadable"] = {}
    st.session_state["data_processamento"] = None
    st.session_state["hora_processamento"] = None
    st.session_state["tempo_ia"] = None
    st.session_state["answer_downloads"] = False


def write_stream(stream):
    result = ""
    container = st.empty()
    for chunk in stream:
        result += chunk
        container.markdown(
            f'<p class="font-stream">{result}</p>', unsafe_allow_html=True
        )


def get_stream(texto):
    for word in texto.split(" "):
        yield word + " "
        time.sleep(0.01)


# Function to initialize session state
def initialize_session_state():
    if "my_dict" not in st.session_state:
        st.session_state.my_dict = []  # Initialize as an empty list


################################################################################################################################
# UX
################################################################################################################################

# Inicio da aplicação
initialize_session_state()

st.set_page_config(
    page_title="Balanço",
    page_icon=":black_medium_square:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Leitura do arquivo css de estilização
with open("./styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


################################################################################################################################
# UI
################################################################################################################################


# Inicio da aplicação
def main():
    if "selecionadas" not in st.session_state:
        st.session_state["selecionadas"] = None

    if "respostas_download_txt" not in st.session_state:
        st.session_state["respostas_download_txt"] = None

    if "respostas_download_pdf" not in st.session_state:
        st.session_state["respostas_download_pdf"] = None

    if "file_name" not in st.session_state:
        st.session_state["file_name"] = None

    if "vectordb_object" not in st.session_state:
        st.session_state["vectordb_object"] = None

    if "status_vetorizacao" not in st.session_state:
        st.session_state["status_vetorizacao"] = False

    if "tipo_documento" not in st.session_state:
        st.session_state["tipo_documento"] = None

    if "Q&A" not in st.session_state:
        st.session_state["Q&A"] = {}

    if "Q&A_done" not in st.session_state:
        st.session_state["Q&A_done"] = False

    if "clear_respostas" not in st.session_state:
        st.session_state["clear_respostas"] = False

    if "data_processamento" not in st.session_state:
        st.session_state["data_processamento"] = None

    if "hora_processamento" not in st.session_state:
        st.session_state["hora_processamento"] = None

    if "pdf_IMG" not in st.session_state:
        st.session_state["pdf_IMG"] = None

    if "versao_prompt" not in st.session_state:
        st.session_state["versao_prompt"] = "v1"

    if "resposta_ia" not in st.session_state:
        st.session_state["resposta_ia"] = ""

    if "tempo_ia" not in st.session_state:
        st.session_state["tempo_ia"] = 0

    if "tempo_vetorizacao" not in st.session_state:
        st.session_state["tempo_vetorizacao"] = 0

    if "tempo_Q&A" not in st.session_state:
        st.session_state["tempo_Q&A"] = 0

    if "tempo_manual" not in st.session_state:
        st.session_state["tempo_manual"] = 0

    if "tokens_doc_embedding" not in st.session_state:
        st.session_state["tokens_doc_embedding"] = 0

    if "disable_downloads" not in st.session_state:
        st.session_state["disable_downloads"] = True

    if "pdf_store" not in st.session_state:
        st.session_state["pdf_store"] = True

    if "id_unico" not in st.session_state:
        st.session_state["id_unico"] = True

    username = "max.saito"
    id = np.random.rand()
    if "." not in username:
        username = "User_NA" + id
    session_name = username

    st.subheader("Análise de Balanço")
    st.write("")

    tab1, tab2 = st.tabs(["Extração padrão", "Perguntas adicionais"])
    # ----------------------------------------------------------------------------------------------
    with tab1:
        with st.container(border=True):
            pdf_file = st.file_uploader(
                "Carregamento de arquivo",
                type=["pdf"],
                key="pdf_file",
                on_change=zera_vetorizacao,
            )

            if pdf_file is not None and not st.session_state["status_vetorizacao"]:
                # Se tiver PDFs na pasta quando inicializar a aplicação, apagá-los
                for arquivo in PASTA_ARQUIVOS.glob("*.pdf"):
                    arquivo.unlink()
                savefile_name = normalize_filename(pdf_file.name)
                with open(PASTA_ARQUIVOS / f"{savefile_name}", "wb") as f:
                    f.write(pdf_file.read())

                st.session_state["pdf_store"] = pdf_file.getbuffer()
                st.session_state["file_name"] = pdf_file.name[:-4]

                data_processamento = datetime.now().strftime("%Y-%m-%d")
                hora_processamento = (datetime.now() - timedelta(hours=3)).strftime(
                    "%H:%M"
                )
                st.session_state["data_processamento"] = data_processamento
                st.session_state["hora_processamento"] = hora_processamento
                tipo = unidecode(
                    str(st.session_state["tipo_documento"]).replace(" ", "_")
                )
                file_name = st.session_state["file_name"]

                id_unico = (
                    str(st.session_state["data_processamento"])
                    + "_"
                    + str(st.session_state["hora_processamento"]).replace(":", "-")
                    + "_"
                    + unidecode(str(st.session_state["file_name"]).lower())
                )
                st.session_state["id_unico"] = id_unico

                pdf_store_full_path = f"{str(PASTA_ARQUIVOS)}/{id_unico}" + ".pdf"
                pdf_store_full_path = str(PASTA_ARQUIVOS) + "/" + id_unico + ".pdf"

                with open(pdf_store_full_path, "wb") as file:
                    file.write(st.session_state["pdf_store"])

                vectordb_store_folder = f"{str(PASTA_VECTORDB)}/{id_unico}"
                if not os.path.exists(vectordb_store_folder):
                    os.makedirs(vectordb_store_folder)

                if not st.session_state["status_vetorizacao"]:
                    st.session_state["tempo_ia"] = 0
                    start_time = time.time()

                    with st.spinner("Processando documento..."):
                        # Converter PDF para imagens
                        convert_pdf_to_images(pdf_store_full_path)
                        st.session_state["status_vetorizacao"] = True

                        end_time = time.time()
                        tempo_vetorizacao = end_time - start_time
                        st.session_state["tempo_vetorizacao"] = tempo_vetorizacao
                        st.session_state["tempo_ia"] = 0

        st.write("")
        if st.session_state["status_vetorizacao"]:
            llm_call = st.button(f"Processar balanço da empresa")
            st.write("")
            ph = st.empty()
            with ph.container():
                if llm_call:
                    if not st.session_state["clear_respostas"]:
                        with st.spinner("Processando balanço..."):
                            start_time = time.time()

                            query = """
                                    Você é um analista de balanços de empresas.
                                    Extraia do documento anexado as informações de ativo e passivo/patrimônio líquido.
                                    
                                    Seja conciso. Sua resposta será somente um dicionário Python obrigatoriamente no formato abaixo.
                                    Sempre incluir TODOS os anos disponibilizados no documento, substituindo as chaves "ano_referencia" pelo ano do balanço.
                                    Inclua somente as chaves que tiverem valores.

                                    Formato obrigatório do dicionário Python de saída:
                                    ####
                                    {
                                        "nome_da_empresa": nome_extraido,
                                        "ano_referencia": {
                                            "ativo_circulante": {
                                            "disponibilidades": valor_extraido,
                                            "aplicacoes_financeiras": valor_extraido,
                                            "duplicatas_a_receber": valor_extraido,
                                            "estoques": valor_extraido,
                                            "impostos_a_recuperar": valor_extraido,
                                            "controladas_e_coligadas": valor_extraido,
                                            "outros_operacionais": valor_extraido,
                                            "outros_nao_operacionais": valor_extraido
                                            },
                                            "realizavel_longo_prazo": {
                                            "aplicacoes_financeiras": valor_extraido,
                                            "controladas_e_coligadas": valor_extraido,
                                            "depositos_judiciais": valor_extraido,
                                            "impostos_diferidos": valor_extraido,
                                            "outros_operacionais": valor_extraido,
                                            "outros_nao_operacionais": valor_extraido
                                            },
                                            "permanente": {
                                            "investimentos": valor_extraido,
                                            "imobilizado_tecnico": valor_extraido,
                                            "intangivel": valor_extraido
                                            },
                                            "total_do_ativo": valor_extraido,
                                            "passivo_circulante": {
                                            "fornecedores": valor_extraido,
                                            "instituicoes_financeiras": valor_extraido,
                                            "salarios_tributos_contribuicoes": valor_extraido,
                                            "dividendos_participacoes": valor_extraido,
                                            "adiantamento_de_clientes": valor_extraido,
                                            "controladas_e_coligadas": valor_extraido,
                                            "outros_operacionais": valor_extraido,
                                            "outros_nao_operacionais": valor_extraido
                                            },
                                            "exigivel_longo_prazo": {
                                            "instituicoes_financeiras": valor_extraido,
                                            "controladas_e_coligadas": valor_extraido,
                                            "provisao_por_contingencias": valor_extraido,
                                            "impostos_diferidos": valor_extraido,
                                            "outros_operacionais": valor_extraido,
                                            "outros_nao_operacionais": valor_extraido
                                            },
                                            "resultados_exercicios_futuros": valor_extraido,
                                            "patrimonio_liquido": {
                                            "capital_social": valor_extraido,
                                            "reservas": valor_extraido,
                                            "participacoes_minoritarias": valor_extraido
                                            },
                                            "total_do_passivo": valor_extraido,
                                            "indicadores": {
                                            "receita_operacional_liquida": valor_extraido,
                                            "ebitda_anual": valor_extraido,
                                            "ebitda_no_periodo": valor_extraido,
                                            "divida_financeira_liquida": valor_extraido,
                                            "divida_liquida_ebitda": valor_extraido,
                                            "margem_ebitda": valor_extraido
                                            }
                                        }
                                    }
                                    ####
                                    """

                            tokens_query_embedding = num_tokens_from_string(
                                query, "cl100k_base"
                            )

                            with get_openai_callback() as cb:
                                id_unico = st.session_state["id_unico"]
                                path_atual = f"{PASTA_IMAGENS}/{id_unico}_images"
                                quantidade_paginas = len(os.listdir(path_atual))
                                response = chain.invoke(
                                    {
                                        "image_paths": [
                                            f"{path_atual}/page{n}.jpg"
                                            for n in range(0, quantidade_paginas)
                                        ],
                                        "prompt": query,
                                    }
                                )
                                print("Total tokens: ", cb.total_tokens)
                                print("Response JSON: ", response)

                                # Remover o prefixo e sufixo indesejados
                                response_cleaned = (
                                    response.lstrip("```python").rstrip("```").strip()
                                )

                                try:
                                    response_dict = json.loads(response_cleaned)
                                    print(
                                        "Nome da empresa:",
                                        response_dict["nome_da_empresa"],
                                    )
                                    st.session_state["resposta_ia"] = response_dict
                                except json.JSONDecodeError as e:
                                    print("Erro ao decodificar JSON:", e)

                            with st.container(border=True):
                                st.session_state["Q&A_done"] = True
                                end_time = time.time()
                                tempo_qa = end_time - start_time
                                st.session_state["tempo_Q&A"] = tempo_qa

                                st.session_state["Q&A"].update(
                                    {
                                        "resposta_ia": response,
                                        "tokens_completion": cb.completion_tokens,
                                        "tokens_prompt": cb.prompt_tokens,
                                        "tokens_query_embedding": tokens_query_embedding,
                                    }
                                )

            if st.session_state["clear_respostas"]:
                ph.empty()
                sleep(0.01)
                st.session_state.clear_respostas = False
            else:
                if st.session_state["Q&A_done"]:
                    id_unico = st.session_state["id_unico"]

                    df_avaliacao = pd.DataFrame(
                        columns=[
                            "id_unico",
                            "data_processamento",
                            "hora_processamento",
                            "versao_prompt",
                            "tokens_prompt",
                            "tokens_completion",
                            "tokens_doc_embedding",
                            "tokens_query_embedding",
                            "custo_prompt",
                            "custo_completion",
                            "custo_doc_embedding",
                            "custo_query_embedding",
                        ]
                    )

                    token_cost = {
                        "tokens_prompt": 15 / 1e6,
                        "tokens_completion": 30 / 1e6,
                        "tokens_doc_embedding": 0.001 / 1e3,
                        "tokens_query_embedding": 0.001 / 1e3,
                    }

                    with ph.container():
                        with st.container(border=True):
                            st.json(
                                json.dumps(
                                    st.session_state["resposta_ia"],
                                    indent=4,
                                    ensure_ascii=False,
                                )
                            )

                            tokens_prompt = st.session_state["Q&A"]["tokens_prompt"]
                            tokens_completion = st.session_state["Q&A"][
                                "tokens_completion"
                            ]
                            tokens_doc_embedding = st.session_state[
                                "tokens_doc_embedding"
                            ]
                            tokens_query_embedding = st.session_state["Q&A"][
                                "tokens_query_embedding"
                            ]

                            custo_prompt = token_cost["tokens_prompt"] * tokens_prompt
                            custo_completion = (
                                token_cost["tokens_completion"] * tokens_completion
                            )
                            custo_doc_embedding = round(
                                token_cost["tokens_doc_embedding"]
                                * tokens_doc_embedding,
                                6,
                            )
                            custo_query_embedding = round(
                                token_cost["tokens_query_embedding"]
                                * tokens_query_embedding,
                                6,
                            )

                            st.session_state["tempo_ia"] = (
                                st.session_state["tempo_vetorizacao"]
                                + st.session_state["tempo_Q&A"]
                            )

                            df_avaliacao.loc[len(df_avaliacao)] = [
                                id_unico,
                                st.session_state["data_processamento"],
                                st.session_state["hora_processamento"],
                                st.session_state["versao_prompt"],
                                tokens_prompt,
                                tokens_completion,
                                tokens_doc_embedding,
                                tokens_query_embedding,
                                custo_prompt,
                                custo_completion,
                                custo_doc_embedding,
                                custo_query_embedding,
                            ]

                    # Formatando as respostas para saída IMG
                    formatted_output_PDF = ""
                    formatted_output_PDF = "<b>Exportação de Balanço</b><br/><br/><br/>"
                    formatted_output_PDF += f"<b>Nome do Arquivo:</b> {st.session_state['file_name']}.pdf<br/>"
                    formatted_output_PDF += f"<b>Data de Processamento:</b> {st.session_state['data_processamento']} <br/>"
                    formatted_output_PDF += f"<b>Hora de Processamento:</b> {st.session_state['hora_processamento']}<br/><br/><br/>"
                    formatted_output_PDF += f"<b>PERGUNTAS E RESPOSTAS</b><br/><br/>"
                    formatted_output_PDF += json.dumps(
                        st.session_state["resposta_ia"], indent=4, ensure_ascii=False
                    )

                    formatted_output_TXT = json.dumps(
                        st.session_state["resposta_ia"], indent=4, ensure_ascii=False
                    )

                    st.session_state["respostas_download_txt"] = formatted_output_TXT
                    st.session_state["respostas_download_pdf"] = formatted_output_PDF

                    buf = io.StringIO()
                    buf.write(formatted_output_TXT)
                    buf.seek(0)

                    def export_result():
                        buf.seek(0)

                    full_path = os.path.join(PASTA_RESPOSTAS, id_unico)

                    col1, col2, col3 = st.columns([4, 1, 1])
                    with col2:
                        st.write("")
                        nome_empresa = unidecode(
                            st.session_state["resposta_ia"]["nome_da_empresa"]
                        )
                        nome_empresa = nome_empresa.replace(" ", "_")

                        txt_file_download_name = id_unico + nome_empresa + ".txt"
                        if st.download_button(
                            "Exportar TXT",
                            buf.getvalue().encode("utf-8"),
                            txt_file_download_name,
                            "text/plain",
                            on_click=export_result,
                            # disabled=st.session_state["disable_downloads"],
                        ):

                            with open(full_path + ".json", "w", encoding="utf-8") as f:
                                json.dump(st.session_state["Q&A"], f, indent=4)
                            with open(
                                full_path + ".txt", "w", encoding="utf-8"
                            ) as file:
                                file.write(formatted_output_TXT)

                    with col3:
                        st.write("")
                        # Output PDF file
                        pdf_file_store_name = full_path + ".pdf"
                        st.session_state["pdf_IMG"] = pdf_file_store_name

                        # Create a PDF document
                        pdf_document = SimpleDocTemplate(pdf_file_store_name)
                        pdf_elements = []

                        # Create a stylesheet for styling
                        styles = getSampleStyleSheet()

                        # Parse the HTML-like text into a Paragraph
                        paragraph = Paragraph(formatted_output_PDF, styles["Normal"])

                        # Add the Paragraph to the PDF elements
                        pdf_elements.append(paragraph)

                        # Build the PDF document
                        pdf_document.build(pdf_elements)

                        pdf_file_download_name = id_unico + nome_empresa + ".pdf"
                        with open(pdf_file_store_name, "rb") as f:
                            st.download_button(
                                "Exportar PDF",
                                f,
                                pdf_file_download_name,
                                # disabled=st.session_state["disable_downloads"],
                            )

                    st.write("")
                    st.write("")
                    st.write("")

                    # ----- df_resumo
                    doc_tokens_prompt_count = df_avaliacao["tokens_prompt"].sum()
                    doc_tokens_completion_count = df_avaliacao[
                        "tokens_completion"
                    ].sum()
                    doc_tokens_doc_embedding_count = df_avaliacao[
                        "tokens_doc_embedding"
                    ].values[0]
                    doc_tokens_query_embedding_count = df_avaliacao[
                        "tokens_query_embedding"
                    ].sum()
                    doc_tokens_prompt_cost = df_avaliacao["custo_prompt"].sum()
                    doc_tokens_completion_cost = df_avaliacao["custo_completion"].sum()
                    doc_tokens_doc_embedding_cost = df_avaliacao[
                        "custo_doc_embedding"
                    ].values[0]
                    doc_tokens_query_embedding_cost = df_avaliacao[
                        "custo_query_embedding"
                    ].sum()

                    custo_ia = round(
                        doc_tokens_prompt_cost
                        + doc_tokens_completion_cost
                        + doc_tokens_doc_embedding_cost
                        + doc_tokens_query_embedding_cost,
                        3,
                    )

                    if st.session_state["tempo_ia"] is not None:
                        tempo_ia = st.session_state["tempo_ia"]
                        tempo_ia = round(tempo_ia / 60, 3)
                    else:
                        tempo_ia = ""

                    df_resumo = pd.DataFrame(
                        {
                            "Usuario": username,
                            "Id_Unico": id_unico,
                            "Data_Processamento": [
                                st.session_state["data_processamento"]
                            ],
                            "Hora_Processamento": [
                                st.session_state["hora_processamento"]
                            ],
                            "Tipo_Documento": [st.session_state["tipo_documento"]],
                            "Versao_Prompt": [st.session_state["versao_prompt"]],
                            "Nome_Arquivo": [st.session_state["file_name"]],
                            "Tempo_IA (min)": [tempo_ia],
                            "Custo_IA": [custo_ia],
                            "Total tokens": [doc_tokens_prompt_count],
                        }
                    )
                    # ----- df_resumo - FIM

                    try:
                        df_avaliacao_current = pd.read_csv(
                            str(PASTA_RAIZ) + "/avaliacao/df_avaliacao.csv"
                        )
                        df_avaliacao = pd.concat(
                            [df_avaliacao_current, df_avaliacao]
                        ).reset_index(drop=True)
                    except:
                        pass
                    df_avaliacao.to_csv(
                        str(PASTA_RAIZ) + "/avaliacao/df_avaliacao.csv",
                        index=False,
                    )

                    try:
                        df_resumo_current = pd.read_csv(
                            str(PASTA_RAIZ) + "/avaliacao/df_resumo.csv"
                        )
                        df_resumo = pd.concat(
                            [df_resumo_current, df_resumo]
                        ).reset_index(drop=True)
                    except:
                        pass
                    df_resumo.to_csv(
                        str(PASTA_RAIZ) + "/avaliacao/df_resumo.csv",
                        index=False,
                    )

            # ----------------------------------------------------------------------------------------------
        with tab2:

            def clear_text():
                st.session_state.query_add = st.session_state.widget
                st.session_state.widget = ""

            st.write("")
            if st.session_state["status_vetorizacao"]:
                st.text_input("**Digite aqui a sua pergunta**", key="widget")
                query_add = st.session_state.get("query_add", "")
                with st.form(key="myform1"):
                    submit_button = st.form_submit_button(
                        label="Enviar", on_click=clear_text
                    )

                    if submit_button:
                        with st.spinner("Processando pergunta adicional"):
                            with get_openai_callback() as cb:
                                id_unico = st.session_state["id_unico"]
                                path_atual = f"{PASTA_IMAGENS}/{id_unico}_images"
                                quantidade_paginas = len(os.listdir(path_atual))
                                with get_openai_callback() as cb:
                                    response = chain.invoke(
                                        {
                                            "image_paths": [
                                                f"{path_atual}/page{n}.jpg"
                                                for n in range(0, quantidade_paginas)
                                            ],
                                            "prompt": query_add,
                                        }
                                    )
                            with st.empty():
                                st.markdown(f"**{query_add}**" + "  \n " + response)

            else:
                st.write("Documento não vetorizado!")


if __name__ == "__main__":
    main()
