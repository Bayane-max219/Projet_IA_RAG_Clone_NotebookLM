# -*- coding: utf-8 -*-
"""
=====================================================================
  CLONE NOTEBOOKLM — Système RAG 100% Local
=====================================================================
Application Streamlit qui permet de charger ses propres documents
(PDF, Markdown, TXT) et d'interagir avec eux de façon intelligente,
SANS aucun appel à des API externes (confidentialité totale).

Deux modes de fonctionnement (bouton "Toggle" dans l'interface) :
  1. Recherche Sémantique pure  -> Toggle OFF : on retourne uniquement
     les passages les plus proches, AUCUN LLM n'est appelé.
  2. Assistant RAG complet       -> Toggle ON  : un LLM local (Ollama)
     rédige une réponse, mais STRICTEMENT à partir des documents.

Pile technique : Streamlit + LangChain + ChromaDB + sentence-transformers + Ollama
=====================================================================
"""

import os
import tempfile

# --- Correctif SSL ---
# Sur certains réseaux (entreprise / école) avec un proxy qui inspecte le HTTPS,
# Python (certifi) ne reconnaît pas le certificat alors que Windows, lui, le
# reconnaît. truststore fait utiliser le magasin de certificats du système
# d'exploitation -> le téléchargement du modèle d'embeddings (HuggingFace)
# fonctionne. Sans effet néfaste si le réseau est "normal".
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import streamlit as st

# --- Chargement des documents (DocumentLoaders) ---
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader

# --- Découpage en chunks ---
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Embeddings (vectorisation locale via sentence-transformers) ---
from langchain_huggingface import HuggingFaceEmbeddings

# --- Base vectorielle ---
from langchain_chroma import Chroma

# --- LLM local + outils de prompt (utilisés seulement en mode RAG) ---
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =====================================================================
#  PARAMÈTRES GLOBAUX
# =====================================================================

# Modèle d'embeddings : version MULTILINGUE de MiniLM.
#   - Comprend le français (documents souvent en FR) aussi bien que l'anglais.
#   - Léger (~120 Mo) et rapide -> idéal pour tourner sur une machine locale.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- Stratégie de Chunking (JUSTIFICATION demandée par le TP) ---
#   CHUNK_SIZE = 1000 caractères :
#     compromis entre un contexte assez riche pour avoir du sens
#     et des segments assez petits pour une recherche précise.
#     Trop grand -> du bruit dans le contexte ; trop petit -> on perd le sens.
#   CHUNK_OVERLAP = 200 caractères (20%) :
#     le chevauchement évite de "couper" une idée en plein milieu entre
#     deux chunks ; une phrase importante à la frontière reste retrouvable.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Nombre de passages récupérés à chaque recherche (valeur par défaut du
# curseur "Nombre d'extraits" dans la sidebar ; l'utilisateur peut la
# changer de 1 à 10 au moment de la recherche).
TOP_K = 5

# Nombre de candidats examinés avant la sélection MMR (diversité).
FETCH_K = 20

# Dossier de persistance de la base vectorielle Chroma.
PERSIST_DIR = "chroma_db"


# =====================================================================
#  FONCTIONS UTILITAIRES (mises en cache quand c'est pertinent)
# =====================================================================

@st.cache_resource(show_spinner="Chargement du modèle d'embeddings...")
def get_embeddings():
    """Charge le modèle d'embeddings une seule fois (coûteux à instancier)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_document(uploaded_file):
    """
    ÉTAPE 2.1 — EXTRACTION
    Lit un fichier uploadé et en extrait le texte via le DocumentLoader adapté.
    Retourne une liste de `Document` LangChain.
    """
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    # Streamlit donne un fichier en mémoire : on l'écrit dans un fichier
    # temporaire car les loaders LangChain attendent un chemin sur disque.
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            loader = PyMuPDFLoader(tmp_path)
        elif suffix in (".txt", ".md"):
            # encoding="utf-8" pour gérer les accents français.
            loader = TextLoader(tmp_path, encoding="utf-8")
        else:
            st.warning(f"Format non supporté : {uploaded_file.name}")
            return []

        docs = loader.load()

        # ÉTAPE 2.3 (métadonnées) — on force le nom de fichier d'origine
        # dans chaque Document pour pouvoir l'afficher comme SOURCE plus tard.
        for d in docs:
            d.metadata["source"] = uploaded_file.name
        return docs
    finally:
        os.remove(tmp_path)  # nettoyage du fichier temporaire


def build_vectorstore(uploaded_files):
    """
    PIPELINE D'INGESTION COMPLET (Étape 2).
    Extraction -> Chunking -> Vectorisation -> Stockage dans ChromaDB.
    """
    # 1) Extraction de tous les fichiers
    all_docs = []
    for f in uploaded_files:
        all_docs.extend(load_document(f))

    if not all_docs:
        return None, 0

    # 2) Chunking (découpage). RecursiveCharacterTextSplitter essaie de
    #    couper en priorité aux sauts de paragraphe/phrase -> chunks "propres".
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)

    # 3) Vectorisation + 4) Stockage dans Chroma (les métadonnées,
    #    dont "source", sont conservées automatiquement).
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    return vectorstore, len(chunks)


def get_llm(model_name):
    """Instancie le LLM local servi par Ollama."""
    # temperature=0 -> réponses factuelles et déterministes (on ne veut PAS
    # que le modèle invente : il doit coller au contexte fourni).
    return OllamaLLM(model=model_name, temperature=0)


# Prompt SYSTÈME strict (Étape 4.2) : le LLM doit répondre EXCLUSIVEMENT
# à partir du contexte. {context} et {question} sont injectés dynamiquement.
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un assistant documentaire. Réponds à la question de l'utilisateur "
     "en te basant EXCLUSIVEMENT sur le CONTEXTE fourni ci-dessous. "
     "N'utilise aucune connaissance extérieure. "
     "Si la réponse ne se trouve pas dans le contexte, réponds exactement : "
     "\"Je ne trouve pas la réponse dans les documents fournis.\" "
     "Réponds en français, de façon claire et concise.\n\n"
     "CONTEXTE :\n{context}"),
    ("human", "Question : {question}"),
])


def format_context(documents):
    """Concatène les chunks récupérés en un seul bloc de contexte."""
    return "\n\n---\n\n".join(
        f"[Source : {d.metadata.get('source', 'inconnue')}]\n{d.page_content}"
        for d in documents
    )


# =====================================================================
#  INTERFACE — ÉTAPE 1 : Squelette
# =====================================================================

st.set_page_config(page_title="Clone NotebookLM (RAG Local)", page_icon="📓", layout="wide")
st.title("📓 Clone NotebookLM — RAG 100% Local")

# Initialisation de l'état de session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Si aucune base n'est chargée en mémoire MAIS qu'une base Chroma a déjà été
# persistée sur le disque (dossier chroma_db/), on la recharge automatiquement.
# Ainsi un simple rechargement de page ne fait PAS perdre les documents indexés.
if st.session_state.get("vectorstore") is None:
    if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        st.session_state.vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=get_embeddings(),
        )
    else:
        st.session_state.vectorstore = None

# ----------------------- BARRE LATÉRALE -----------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # Zone de téléchargement de fichiers
    uploaded_files = st.file_uploader(
        "Chargez vos documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    # Bouton d'indexation -> lance le pipeline d'ingestion
    if st.button("📥 Indexer les documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Veuillez d'abord charger au moins un fichier.")
        else:
            # Réinitialisation : chaque indexation reconstruit la base
            # UNIQUEMENT à partir des fichiers actuellement chargés, pour ne
            # pas la mélanger avec d'anciens documents déjà indexés.
            if st.session_state.get("vectorstore") is not None:
                try:
                    st.session_state.vectorstore.delete_collection()
                except Exception:
                    pass
                st.session_state.vectorstore = None
            with st.spinner("Indexation en cours (extraction, découpage, vectorisation)..."):
                vs, n_chunks = build_vectorstore(uploaded_files)
            if vs is not None:
                st.session_state.vectorstore = vs
                st.success(f"✅ {len(uploaded_files)} fichier(s) indexé(s) — {n_chunks} chunks.")
            else:
                st.error("Aucun texte n'a pu être extrait.")


    st.divider()

    # Bouton TOGGLE d'activation/désactivation du LLM (cœur du TP)
    use_llm = st.toggle("🤖 Activer l'Assistant RAG (LLM)", value=False)

    # Choix du modèle Ollama (actif uniquement si le LLM est activé)
    model_name = st.selectbox(
        "Modèle Ollama",
        # On ne liste que les modèles réellement installés via `ollama pull`
        # (sinon la sélection d'un modèle absent provoque une erreur).
        # qwen2.5:3b : bon compromis qualité/RAM (~2 Go) pour le français.
        # llama3.2:1b : ultra léger (~1.3 Go) mais moins fiable.
        options=["qwen2.5:3b", "llama3.2:1b"],
        index=0,
        disabled=not use_llm,
    )

    if use_llm:
        st.caption("Mode **Assistant RAG complet** : le LLM rédige une réponse à partir des documents.")
    else:
        st.caption("Mode **Recherche Sémantique pure** : extraits bruts, aucun LLM.")

    st.divider()

    # --- Nombre d'extraits (k) réglable par l'utilisateur ---
    # Curseur de 1 à 10 : la recherche renverra EXACTEMENT ce nombre de
    # passages (chunks). Cette valeur remplace la constante TOP_K figée.
    top_k = st.slider(
        "🔍 Nombre d'extraits à récupérer (k)",
        min_value=1,
        max_value=10,
        value=TOP_K,
        help="Nombre de passages les plus pertinents renvoyés par la recherche.",
    )

    st.divider()

    # Bouton pour effacer l'historique de la conversation (repartir à zéro).
    if st.button("🗑️ Effacer l'historique", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------------- ZONE PRINCIPALE -----------------------

# Affichage de l'historique de conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Si des sources avaient été attachées, on les redéploie
        if msg.get("sources"):
            with st.expander("🔎 Voir les extraits utilisés comme contexte"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Extrait {i} — `{src['source']}`**")
                    st.text(src["content"])


# Saisie utilisateur
question = st.chat_input("Posez une question sur vos documents...")


if question:
    # Affiche et mémorise la question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Garde-fou : aucune base indexée
    if st.session_state.vectorstore is None:
        with st.chat_message("assistant"):
            warn = "⚠️ Aucun document indexé. Chargez des fichiers puis cliquez sur **Indexer**."
            st.markdown(warn)
        st.session_state.messages.append({"role": "assistant", "content": warn})
    else:
        # Recherche des passages les plus proches (commune aux deux modes).
        # search_type="mmr" (Maximal Marginal Relevance) : sélectionne des
        # passages à la fois PERTINENTS et DIVERS -> évite de remplir le
        # contexte avec des quasi-doublons (utile quand un document répète
        # plusieurs fois le même passage).
        # fetch_k doit rester >= k (MMR examine fetch_k candidats avant
        # d'en choisir k) ; on le recalcule ici à partir du curseur.
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "fetch_k": max(FETCH_K, top_k * 4), "lambda_mult": 0.5},
        )
        retrieved_docs = retriever.invoke(question)

        sources = [
            {"source": d.metadata.get("source", "inconnue"), "content": d.page_content}
            for d in retrieved_docs
        ]

        with st.chat_message("assistant"):
            # =====================================================
            #  MODE 1 — RECHERCHE SÉMANTIQUE PURE (Toggle OFF)
            #  Étape 3 : on affiche les extraits BRUTS + le fichier source.
            # =====================================================
            if not use_llm:
                st.markdown("**🔍 Extraits les plus pertinents (recherche sémantique) :**")
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**Extrait {i} — source : `{src['source']}`**")
                    st.info(src["content"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "🔍 Extraits les plus pertinents (recherche sémantique) :",
                    "sources": sources,
                })

            # =====================================================
            #  MODE 2 — ASSISTANT RAG COMPLET (Toggle ON)
            #  Étape 4 : LLM contraint par le contexte + transparence.
            # =====================================================
            else:
                try:
                    context = format_context(retrieved_docs)
                    llm = get_llm(model_name)
                    # Chaîne : prompt -> LLM -> texte
                    chain = RAG_PROMPT | llm | StrOutputParser()
                    with st.spinner(f"🤖 {model_name} rédige la réponse..."):
                        answer = chain.invoke({"context": context, "question": question})

                    st.markdown(answer)

                    # Étape 4.4 — TRANSPARENCE : extraits dépliables
                    with st.expander("🔎 Voir les extraits utilisés comme contexte"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**Extrait {i} — `{src['source']}`**")
                            st.text(src["content"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                except Exception as e:
                    err = (f"❌ Erreur lors de l'appel au LLM. "
                           f"Vérifiez qu'Ollama tourne et que le modèle `{model_name}` "
                           f"est téléchargé (`ollama run {model_name}`).\n\n`{e}`")
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
