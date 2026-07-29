# 📖 Explication ligne par ligne de `app.py`

> Document de soutenance — explique **chaque ligne** du code, la **syntaxe Python**
> utilisée, et le **rôle** de chaque élément dans le système RAG.
> Tout le code du projet tient dans le seul fichier **`app.py`**.

---

## RAPPEL : c'est quoi un RAG ?

**RAG = Retrieval-Augmented Generation** (Génération Augmentée par la Recherche).

Principe en une phrase :
> On découpe les documents en petits morceaux, on les transforme en vecteurs
> (nombres). Quand l'utilisateur pose une question, on retrouve les morceaux les
> plus proches du sens de la question, et on les donne à un LLM qui rédige une
> réponse **uniquement** à partir de ces morceaux.

Les 6 étapes du pipeline :
```
1. UPLOAD     : charger un fichier (PDF/TXT/MD)
2. CHUNKING   : le découper en petits morceaux (chunks)
3. EMBEDDINGS : transformer chaque chunk en vecteur de nombres
                → stocker dans la base ChromaDB
4. QUESTION   : l'utilisateur pose une question (elle aussi transformée en vecteur)
5. RECHERCHE  : trouver les 5 chunks les plus proches → "Extrait 1 à 5"
6. LLM        : donner ces 5 chunks au modèle → il rédige la réponse
```

---

# 1. EN-TÊTE ET IMPORTS

```python
# -*- coding: utf-8 -*-
```
**Ligne 1.** Déclare que le fichier est encodé en UTF-8. Permet d'écrire des
accents (é, à, ç) dans le code sans erreur. C'est une convention Python.

```python
"""
  CLONE NOTEBOOKLM — Système RAG 100% Local
  ...
"""
```
**Lignes 2-18.** Un **docstring** : un gros commentaire entre triple guillemets
`"""`. Il décrit ce que fait le fichier. Python l'ignore à l'exécution ; c'est
de la documentation pour le lecteur.

```python
import os
import tempfile
```
**Lignes 20-21.** `import` charge un **module** (une boîte à outils de Python).
- `os` : pour parler au système (chemins de fichiers, supprimer un fichier…).
- `tempfile` : pour créer des fichiers temporaires.

```python
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass
```
**Lignes 29-33.** Correctif réseau.
- `try: ... except: ...` = « **essaie** ce code ; **s'il échoue**, ne plante pas ».
- `truststore.inject_into_ssl()` fait utiliser à Python les certificats de
  Windows (sinon le téléchargement du modèle d'embeddings échoue sur un réseau
  d'école avec proxy).
- `except Exception: pass` : si `truststore` n'est pas installé, on **ignore**
  (`pass` = « ne rien faire ») et le programme continue.

```python
import streamlit as st
```
**Ligne 35.** Importe **Streamlit** (la librairie qui crée l'interface web) et la
renomme `st` (un alias court). On écrira donc `st.title(...)` au lieu de
`streamlit.title(...)`.

```python
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
```
**Ligne 38.** `from X import Y` = importe seulement les outils `Y` du module `X`.
Ici : `PyMuPDFLoader` (lit les PDF) et `TextLoader` (lit les .txt/.md).

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```
**Ligne 41.** Importe l'outil qui **découpe** un texte en chunks.

```python
from langchain_huggingface import HuggingFaceEmbeddings
```
**Ligne 44.** Importe l'outil qui transforme du texte en **vecteurs**
(embeddings).

```python
from langchain_chroma import Chroma
```
**Ligne 47.** Importe **ChromaDB**, la base de données vectorielle (elle stocke
les vecteurs et sait retrouver les plus proches).

```python
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
```
**Lignes 50-52.** Outils pour le mode LLM :
- `OllamaLLM` : pour parler au modèle local (Ollama).
- `ChatPromptTemplate` : pour construire le prompt (la consigne donnée au LLM).
- `StrOutputParser` : pour récupérer la réponse du LLM sous forme de texte simple.

---

# 2. PARAMÈTRES GLOBAUX (les réglages)

```python
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```
**Ligne 62.** Une **variable** = une étiquette qui range une valeur. Ici on range
le nom du modèle d'embeddings. Multilingue (comprend le français), léger (~120 Mo).

```python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```
**Lignes 72-73.** Réglages du découpage (**à justifier en soutenance**) :
- `CHUNK_SIZE = 1000` : chaque morceau fait ~1000 caractères. Assez grand pour
  garder du sens, assez petit pour une recherche précise.
- `CHUNK_OVERLAP = 200` : deux morceaux voisins se **chevauchent** de 200
  caractères, pour ne pas couper une idée en deux.

```python
TOP_K = 5
FETCH_K = 20
```
**Lignes 76-79.**
- `TOP_K = 5` : on récupère les **5** chunks les plus pertinents (= Extrait 1 à 5).
  C'est un **maximum** : s'il y a moins de chunks, on en récupère moins.
- `FETCH_K = 20` : la recherche MMR examine 20 candidats avant de choisir les 5
  plus pertinents ET diversifiés.

```python
PERSIST_DIR = "chroma_db"
```
**Ligne 82.** Le nom du dossier où la base vectorielle est sauvegardée sur le
disque.

> 💡 Pourquoi des MAJUSCULES (`CHUNK_SIZE`) ? Convention Python : une variable en
> majuscules est une **constante** (une valeur de réglage qui ne change pas).

---

# 3. LES FONCTIONS

> Une **fonction** = un bloc de code réutilisable, défini avec `def nom(...):`.
> On l'« appelle » plus tard pour exécuter ce bloc.

## 3.1 `get_embeddings()` — charger le modèle d'embeddings

```python
@st.cache_resource(show_spinner="Chargement du modèle d'embeddings...")
def get_embeddings():
    """Charge le modèle d'embeddings une seule fois (coûteux à instancier)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
```
- `@st.cache_resource(...)` : un **décorateur** Streamlit. Il met le résultat
  **en cache** → le modèle n'est chargé qu'**une seule fois** (gain de temps).
- `def get_embeddings():` : définit la fonction.
- `return ...` : la fonction **renvoie** l'objet embeddings au code qui l'appelle.

## 3.2 `load_document()` — EXTRACTION (Étape 2.1)

```python
def load_document(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
```
- `uploaded_file` est le **paramètre** (le fichier reçu).
- `os.path.splitext("part 2.txt")` renvoie `("part 2", ".txt")`. Le `[1]`
  prend le 2ᵉ élément (`.txt`). `.lower()` le met en minuscule (`.TXT` → `.txt`).
- Résultat : `suffix` contient l'extension du fichier.

```python
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
```
- `with ... as tmp:` : ouvre une ressource et la referme proprement à la fin.
- On crée un fichier temporaire et on y **écrit** le contenu du fichier uploadé.
- Pourquoi ? Streamlit donne le fichier « en mémoire », mais les loaders
  LangChain veulent un **chemin sur le disque**. On fait donc ce détour.

```python
    try:
        if suffix == ".pdf":
            loader = PyMuPDFLoader(tmp_path)
        elif suffix in (".txt", ".md"):
            loader = TextLoader(tmp_path, encoding="utf-8")
        else:
            st.warning(f"Format non supporté : {uploaded_file.name}")
            return []
```
- `if / elif / else` : choix selon le type de fichier.
  - PDF → `PyMuPDFLoader` (l'outil conseillé par le sujet).
  - .txt ou .md → `TextLoader` (avec `encoding="utf-8"` pour les accents).
  - sinon → on affiche un avertissement et on `return []` (liste vide = on arrête).
- `f"...{uploaded_file.name}"` : une **f-string**. Le `{...}` insère la valeur
  d'une variable dans le texte.

```python
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = uploaded_file.name
        return docs
```
- `loader.load()` : lit le fichier et renvoie une liste de `Document`.
- `for d in docs:` : **boucle** sur chaque document.
- `d.metadata["source"] = uploaded_file.name` : on **enregistre le nom du
  fichier** dans les métadonnées (Étape 2.3). C'est ce qui s'affichera comme
  « source » à côté de chaque extrait.
- `return docs` : renvoie les documents extraits.

```python
    finally:
        os.remove(tmp_path)
```
- `finally:` : ce bloc s'exécute **toujours** (succès ou erreur). On y **supprime
  le fichier temporaire** pour ne pas encombrer le disque.

## 3.3 `build_vectorstore()` — LE PIPELINE COMPLET (Étape 2)

```python
def build_vectorstore(uploaded_files):
    all_docs = []
    for f in uploaded_files:
        all_docs.extend(load_document(f))
    if not all_docs:
        return None, 0
```
- `all_docs = []` : crée une **liste vide**.
- `for f in uploaded_files:` : boucle sur chaque fichier chargé.
- `all_docs.extend(...)` : ajoute les documents extraits à la liste.
- `if not all_docs:` : « si la liste est vide » → `return None, 0` (rien à indexer).
  *(Une fonction peut renvoyer plusieurs valeurs séparées par une virgule.)*

```python
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
```
- On crée le **découpeur** avec nos réglages (1000 / 200).
- `separators=[...]` : il essaie de couper d'abord aux **paragraphes** (`\n\n`),
  puis aux lignes (`\n`), puis aux phrases (`. `)… pour des coupures « propres ».
- `splitter.split_documents(all_docs)` : découpe tout → liste de **chunks**
  (par ex. 80 chunks pour votre fichier).

```python
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    return vectorstore, len(chunks)
```
- `Chroma.from_documents(...)` : pour chaque chunk, calcule son **vecteur** et le
  **stocke** dans ChromaDB. `persist_directory` = on sauvegarde sur le disque.
- `return vectorstore, len(chunks)` : renvoie la base + le **nombre** de chunks
  (`len(...)` = longueur d'une liste).

## 3.4 `get_llm()` — créer le LLM

```python
def get_llm(model_name):
    return OllamaLLM(model=model_name, temperature=0)
```
- Crée l'objet qui parle au modèle Ollama choisi.
- `temperature=0` : le modèle est **déterministe** et factuel (il n'invente pas).
  Une température élevée rendrait les réponses plus « créatives » (= risque
  d'invention), ce qu'on ne veut pas en RAG.

## 3.5 `RAG_PROMPT` — la consigne stricte (Étape 4.2)

```python
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un assistant documentaire. Réponds ... EXCLUSIVEMENT sur le CONTEXTE ..."
     "Si la réponse ne se trouve pas dans le contexte, réponds exactement : "
     "\"Je ne trouve pas la réponse dans les documents fournis.\" "
     "... CONTEXTE :\n{context}"),
    ("human", "Question : {question}"),
])
```
- Un **template de prompt** = un texte à trous. Deux messages :
  - `"system"` : la **consigne** au modèle → « réponds UNIQUEMENT à partir du
    CONTEXTE ; si absent, dis "Je ne trouve pas…" ». C'est le **garde-fou** qui
    empêche le LLM d'inventer.
  - `"human"` : la question de l'utilisateur.
- `{context}` et `{question}` sont les **trous** remplis automatiquement au moment
  de l'appel (Étape 4.2 : injection dynamique).
- `\"...\"` : le `\"` permet d'écrire un guillemet **à l'intérieur** d'un texte
  déjà entre guillemets.

## 3.6 `format_context()` — assembler les chunks

```python
def format_context(documents):
    return "\n\n---\n\n".join(
        f"[Source : {d.metadata.get('source', 'inconnue')}]\n{d.page_content}"
        for d in documents
    )
```
- Transforme la liste de chunks en **un seul bloc de texte** (le `{context}`).
- `"séparateur".join(liste)` : colle les éléments d'une liste avec un séparateur
  entre chaque (ici `\n\n---\n\n`).
- `d.metadata.get('source', 'inconnue')` : récupère le nom du fichier ; si absent,
  met `'inconnue'`.
- C'est une **list comprehension** : `f"..." for d in documents` génère un texte
  par document.

---

# 4. L'INTERFACE (Étape 1)

```python
st.set_page_config(page_title="Clone NotebookLM (RAG Local)", page_icon="📓", layout="wide")
st.title("📓 Clone NotebookLM — RAG 100% Local")
```
- `st.set_page_config(...)` : règle le titre de l'onglet, l'icône, la largeur.
- `st.title(...)` : affiche le grand titre en haut de la page.

```python
if "messages" not in st.session_state:
    st.session_state.messages = []
```
- `st.session_state` : la **mémoire** de Streamlit qui survit aux rechargements.
- Ici : si la liste `messages` n'existe pas encore, on la crée vide. Elle stockera
  l'**historique** de la conversation.

```python
if st.session_state.get("vectorstore") is None:
    if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        st.session_state.vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=get_embeddings(),
        )
    else:
        st.session_state.vectorstore = None
```
- Si aucune base n'est en mémoire **et** qu'un dossier `chroma_db/` non vide existe
  sur le disque → on **recharge** la base automatiquement.
- Avantage : recharger la page ne fait **pas perdre** les documents déjà indexés.
- `os.path.isdir(...)` : « est-ce un dossier ? » ; `os.listdir(...)` : « contient-il
  des fichiers ? ».

## 4.1 La barre latérale (sidebar)

```python
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_files = st.file_uploader(
        "Chargez vos documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
```
- `with st.sidebar:` : tout ce qui est indenté en dessous va dans la **barre
  latérale** gauche.
- `st.file_uploader(...)` : la zone d'**upload**. `type=[...]` limite aux formats
  acceptés ; `accept_multiple_files=True` autorise plusieurs fichiers.

```python
    if st.button("📥 Indexer les documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Veuillez d'abord charger au moins un fichier.")
        else:
            with st.spinner("Indexation en cours ..."):
                vs, n_chunks = build_vectorstore(uploaded_files)
            if vs is not None:
                st.session_state.vectorstore = vs
                st.success(f"✅ {len(uploaded_files)} fichier(s) indexé(s) — {n_chunks} chunks.")
            else:
                st.error("Aucun texte n'a pu être extrait.")
```
- `st.button(...)` renvoie `True` **quand on clique**. Donc tout le bloc s'exécute
  au clic sur « Indexer ».
- `if not uploaded_files:` : si aucun fichier → avertissement.
- `with st.spinner(...)` : affiche une animation « en cours » pendant le calcul.
- `vs, n_chunks = build_vectorstore(...)` : lance le **pipeline** (Étape 2) et
  récupère la base + le nombre de chunks.
- On stocke la base en mémoire (`st.session_state.vectorstore = vs`) et on affiche
  un message de succès vert.

```python
    use_llm = st.toggle("🤖 Activer l'Assistant RAG (LLM)", value=False)
```
- **LE TOGGLE** (cœur du TP). `st.toggle(...)` renvoie `True`/`False` selon
  l'interrupteur. `value=False` = désactivé par défaut. Cette variable `use_llm`
  décide ensuite quel mode utiliser.

```python
    model_name = st.selectbox(
        "Modèle Ollama",
        options=["qwen2.5:3b", "llama3.2:1b", "mistral", "qwen2.5-coder", "llama3.2"],
        index=0,
        disabled=not use_llm,
    )
```
- `st.selectbox(...)` : un menu déroulant. `index=0` = premier choix par défaut.
- `disabled=not use_llm` : le menu est **grisé** si le LLM est désactivé.
- *(Ce menu est un BONUS ; le sujet ne demande que le toggle.)*

```python
    if use_llm:
        st.caption("Mode **Assistant RAG complet** ...")
    else:
        st.caption("Mode **Recherche Sémantique pure** ...")
```
- Affiche une petite légende qui change selon le mode actif.

## 4.2 L'affichage de l'historique

```python
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("🔎 Voir les extraits utilisés comme contexte"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Extrait {i} — `{src['source']}`**")
                    st.text(src["content"])
```
- Boucle sur **tous les messages** passés et les réaffiche (sinon ils
  disparaîtraient à chaque interaction).
- `st.chat_message(role)` : une bulle de chat (`"user"` ou `"assistant"`).
- `if msg.get("sources"):` : si le message avait des extraits attachés, on les
  remet dans un volet dépliable (`st.expander`).
- `enumerate(liste, 1)` : boucle en donnant aussi un **compteur** qui commence à 1
  (→ Extrait 1, 2, 3…).

## 4.3 La saisie de la question

```python
question = st.chat_input("Posez une question sur vos documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
```
- `st.chat_input(...)` : la barre de saisie en bas. Renvoie le texte tapé (ou
  `None` si rien).
- `if question:` : on ne fait quelque chose que **si** l'utilisateur a écrit.
- `.append({...})` : ajoute la question à l'historique. Un `{...}` est un
  **dictionnaire** (paires clé:valeur).
- On affiche aussi la question dans une bulle « user ».

```python
    if st.session_state.vectorstore is None:
        with st.chat_message("assistant"):
            warn = "⚠️ Aucun document indexé. Chargez des fichiers puis cliquez sur **Indexer**."
            st.markdown(warn)
        st.session_state.messages.append({"role": "assistant", "content": warn})
```
- **Garde-fou** : si aucune base n'est indexée, on affiche un avertissement au lieu
  de chercher.

## 4.4 La recherche (commune aux 2 modes) — Étape 4.1

```python
    else:
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": TOP_K, "fetch_k": FETCH_K, "lambda_mult": 0.5},
        )
        retrieved_docs = retriever.invoke(question)
```
- `as_retriever(...)` : transforme la base en **moteur de recherche**.
- `search_type="mmr"` : méthode **MMR** (Maximal Marginal Relevance) = elle choisit
  des extraits **pertinents ET variés** (évite les quasi-doublons).
- `k=5` : on veut 5 extraits ; `fetch_k=20` : examiner 20 candidats d'abord ;
  `lambda_mult=0.5` : équilibre entre pertinence et diversité.
- `retriever.invoke(question)` : **lance la recherche** → renvoie les chunks
  (= Extrait 1 à 5). **C'est ici que sont produits les extraits.**

```python
        sources = [
            {"source": d.metadata.get("source", "inconnue"), "content": d.page_content}
            for d in retrieved_docs
        ]
```
- Construit une liste de dictionnaires `{source, content}` pour chaque chunk
  trouvé (utilisée pour l'affichage). Encore une **list comprehension**.

## 4.5 MODE 1 — Recherche sémantique pure (Toggle OFF) — Étape 3

```python
        with st.chat_message("assistant"):
            if not use_llm:
                st.markdown("**🔍 Extraits les plus pertinents ...**")
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**Extrait {i} — source : `{src['source']}`**")
                    st.info(src["content"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "🔍 Extraits les plus pertinents ...",
                    "sources": sources,
                })
```
- `if not use_llm:` = « si le toggle est OFF ».
- On affiche **directement les extraits bruts** + le nom du fichier source.
  **Aucun LLM n'est appelé** (c'est tout l'intérêt de ce mode : vérifier que la
  recherche fonctionne — Étape 3).
- On sauvegarde le message + les sources dans l'historique.

## 4.6 MODE 2 — Assistant RAG complet (Toggle ON) — Étape 4

```python
            else:
                try:
                    context = format_context(retrieved_docs)
                    llm = get_llm(model_name)
                    chain = RAG_PROMPT | llm | StrOutputParser()
                    with st.spinner(f"🤖 {model_name} rédige la réponse..."):
                        answer = chain.invoke({"context": context, "question": question})
                    st.markdown(answer)
```
- `context = format_context(...)` : assemble les 5 chunks en un bloc de texte.
- `llm = get_llm(model_name)` : crée le modèle choisi.
- `chain = RAG_PROMPT | llm | StrOutputParser()` : **LE CŒUR DU RAG**.
  Le `|` (pipe) **enchaîne** les étapes (syntaxe LangChain « LCEL ») :
  ```
  prompt rempli  →  envoyé au LLM  →  réponse convertie en texte
  ```
- `chain.invoke({"context": ..., "question": ...})` : **remplit les trous**
  `{context}` et `{question}`, envoie au LLM, et récupère la **réponse** (Étape
  4.2 + 4.3).
- `st.markdown(answer)` : affiche la réponse.

```python
                    with st.expander("🔎 Voir les extraits utilisés comme contexte"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**Extrait {i} — `{src['source']}`**")
                            st.text(src["content"])
                    st.session_state.messages.append({
                        "role": "assistant", "content": answer, "sources": sources,
                    })
```
- **TRANSPARENCE (Étape 4.4)** : sous la réponse, un volet dépliable montre les
  extraits qui ont servi de contexte (on peut vérifier d'où vient la réponse).
- On sauvegarde la réponse + les sources dans l'historique.

```python
                except Exception as e:
                    err = (f"❌ Erreur lors de l'appel au LLM. "
                           f"Vérifiez qu'Ollama tourne et que le modèle `{model_name}` "
                           f"est téléchargé (`ollama run {model_name}`).\n\n`{e}`")
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
```
- `except Exception as e:` : si l'appel au LLM échoue (Ollama éteint, modèle non
  téléchargé, manque de RAM…), on **attrape** l'erreur dans la variable `e` et on
  affiche un message clair au lieu de planter.

---

# 5. RÉSUMÉ DU FLUX

```
QUAND ON CLIQUE SUR "INDEXER" :
   build_vectorstore()  →  load_document() [extraction]
                        →  split_documents() [chunking]
                        →  Chroma.from_documents() [embeddings + stockage]

QUAND ON POSE UNE QUESTION :
   retriever.invoke(question)  →  Extrait 1 à 5
        │
        ├── Toggle OFF (Étape 3) : on affiche les extraits bruts
        │
        └── Toggle ON  (Étape 4) : chain = prompt | llm | parser
                                   → le LLM rédige la réponse
                                   → + volet "Voir les extraits" (transparence)
```

---

# 6. NOTIONS PYTHON UTILISÉES (mémo)

| Syntaxe | Signification |
|---|---|
| `import x` / `from x import y` | Charger une librairie / un outil précis |
| `def nom(param):` | Définir une fonction |
| `return valeur` | Renvoyer un résultat |
| `if / elif / else` | Choix conditionnel |
| `for x in liste:` | Boucle sur chaque élément |
| `[... for x in liste]` | List comprehension (créer une liste) |
| `try / except / finally` | Gérer les erreurs |
| `with ... as ...:` | Ouvrir/fermer proprement une ressource |
| `{ "clé": valeur }` | Dictionnaire (paires clé-valeur) |
| `[ ]` | Liste |
| `f"texte {variable}"` | f-string (insérer une variable dans du texte) |
| `@décorateur` | Modifie le comportement d'une fonction (ex: cache) |
| `|` (LangChain) | Enchaîner les étapes : prompt → LLM → parser |

---

# 7. NOTIONS LLM / RAG UTILISÉES (mémo)

| Terme | Signification |
|---|---|
| **Embeddings** | Transformer du texte en vecteur de nombres représentant son **sens** |
| **Vecteur** | Liste de nombres ; deux textes au sens proche ont des vecteurs proches |
| **Base vectorielle (ChromaDB)** | Stocke les vecteurs et retrouve les plus proches |
| **Chunk** | Petit morceau de document |
| **Chunking / overlap** | Découpage en morceaux / chevauchement entre morceaux |
| **Retrieval** | Recherche des chunks proches de la question (→ Extrait 1 à 5) |
| **MMR** | Recherche qui privilégie pertinence **+ diversité** |
| **Top-K** | Nombre de chunks récupérés (ici 5) |
| **Prompt / PromptTemplate** | La consigne donnée au LLM, avec des trous à remplir |
| **Contexte** | Les chunks injectés dans le prompt |
| **Temperature** | Contrôle le « hasard » du LLM (0 = factuel, pas d'invention) |
| **LLM local (Ollama)** | Modèle qui tourne sur votre machine, sans Internet |
| **Garde-fou** | Consigne qui empêche le LLM de répondre hors documents |
```
