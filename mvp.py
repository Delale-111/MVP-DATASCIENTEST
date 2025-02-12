import streamlit as st
import graphviz

# Configuration de la page
st.set_page_config(page_title="Moteur de Recherche Intelligent - Méthodologies", layout="wide")

# Menu de navigation dans la sidebar
choice = st.sidebar.radio(
    "Liste des méthodologies :",
    (
        "Méthodologie 1 : Filtrage basé sur le contenu (NLP)",
        "Méthodologie 2 : Filtrage collaboratif basé sur l'historique",
        "Méthodologie 3 : Réseau de neurones léger pour le ré-ranking",
        "Synthèse"
    )
)

# =============================================================================
# MÉTHODOLOGIE 1 : Filtrage basé sur le contenu (NLP)
# =============================================================================
if choice == "Méthodologie 1 : Filtrage basé sur le contenu (NLP)":
    st.header("Méthodologie 1 : Filtrage Basé sur le Contenu avec NLP")
    st.markdown(
        """
Cette approche exploite l’analyse sémantique des descriptions et titres de produits afin d’aligner au mieux la recherche utilisateur avec le contenu textuel.  
**Objectif :** Fournir des résultats de recherche pertinents en se basant sur une représentation vectorielle des textes.
        """
    )

    st.subheader("1.1. Workflow Global de la Méthodologie 1")
    st.markdown("Le diagramme suivant décrit l’ensemble du pipeline :")
    st.image("w.png", use_container_width=True)

    st.subheader("1.2. Pipeline de Prétraitement et Vectorisation")
    with st.expander("Voir le détail du prétraitement et de la vectorisation"):
        st.markdown("### 1.2.1. Extraction et Nettoyage des Données")
        st.markdown(
            """
- **Extraction des Titres et Descriptions :**  
  Le système interroge la base de données ou reçoit un flux de données (via API REST ou pipeline ETL) afin de récupérer les textes associés aux produits.

- **Nettoyage du Texte :**  
  - *Suppression des Caractères Spéciaux* : Filtrage des symboles et caractères non alphabétiques.  
  - *Normalisation* : Transformation en minuscules.  
  - *Stopwords, Stemming et Lemmatisation* : Utilisation de bibliothèques comme [spaCy](https://spacy.io/) ou [NLTK](https://www.nltk.org/) pour éliminer les mots non informatifs et uniformiser les termes (exemple : "marcher" et "marche").
            """
        )
        st.markdown("### 1.2.2. Vectorisation des Textes")
        st.markdown(
            """
Deux techniques principales sont utilisées :
- **TF-IDF (Term Frequency-Inverse Document Frequency) :**
  
  La pondération TF-IDF pour un terme \\( t \\) dans un document \\( d \\) est donnée par :
  \\[
  \\text{TF-IDF}(t,d) = \\text{TF}(t,d) \\times \\log\\left(\\frac{N}{n_t}\\right)
  \\]
  où :
  - \\( \\text{TF}(t,d) \\) est la fréquence du terme dans le document.
  - \\( N \\) est le nombre total de documents.
  - \\( n_t \\) est le nombre de documents contenant le terme.

- **Word2Vec :**
  
  Chaque mot est représenté par un vecteur dense (ex. 300 dimensions). La représentation d’un texte est souvent la moyenne des vecteurs de chaque mot :
  \\[
  \\mathbf{v}_{\\text{texte}} = \\frac{1}{n} \\sum_{i=1}^{n} \\mathbf{v}_{w_i}
  \\]
            """
        )
        st.markdown("### 1.2.3. Encodage de la Requête")
        st.markdown(
            "La requête de l’utilisateur est prétraitée et vectorisée de la même manière, donnant lieu à un vecteur \\( \\mathbf{E}_q \\)."
        )

    st.subheader("1.3. Modèle de Similarité et Classement")
    with st.expander("Détails du calcul de similarité et du classement"):
        st.markdown("#### 1.3.1. Calcul de la Similarité par Cosine Similarity")
        st.latex(r"\text{Cosine Similarity}(\mathbf{E}_q, \mathbf{E}_p) = \frac{\mathbf{E}_q \cdot \mathbf{E}_p}{\|\mathbf{E}_q\| \, \|\mathbf{E}_p\|}")
        st.markdown(
            """
*Explication détaillée :*
- **Produit scalaire \\( \\mathbf{E}_q \\cdot \\mathbf{E}_p \\) :** Somme des produits des composantes correspondantes.
- **Norme \\( \\|\\mathbf{E}\\| \\) :** Calculée par \\( \\|\\mathbf{E}\\| = \\sqrt{\\sum_{i=1}^{d} E_i^2} \\).
  
Un score élevé (compris entre 0 et 1 si les vecteurs sont positifs) indique une forte similarité.
            """
        )
        st.markdown("#### 1.3.2. Classement et Indexation")
        st.markdown(
            "Les produits sont classés par ordre décroissant du score de similarité. Pour optimiser la recherche sur de gros volumes de données, des bibliothèques comme [FAISS](https://github.com/facebookresearch/faiss) peuvent être utilisées."
        )

    st.subheader("1.4. Amélioration par Apprentissage Supervisé")
    with st.expander("Détails sur l'apprentissage supervisé pour le raffinement"):
        st.markdown("#### 1.4.1. Constitution du Dataset")
        st.markdown(
            "Création d’un dataset d’entraînement composé de paires (requête, produit) étiquetées (pertinence binaire ou score continu), via annotations manuelles ou logs d’interaction."
        )
        st.markdown("#### 1.4.2. Modélisation de la Pertinence avec un MLP")
        st.markdown(
            """
Exemple d'architecture d'un réseau de neurones (MLP) :
- **Entrée :** Concaténation de \\( \\mathbf{E}_q \\) et \\( \\mathbf{E}_p \\) (dimension \\( 2d \\)).
- **Couches Cachées :**
  - 1ère couche : 128 neurones avec activation ReLU.
  - 2ème couche : 64 neurones avec activation ReLU.
- **Couche de Sortie :** 1 neurone avec activation sigmoïde pour obtenir une probabilité.
  
La sortie est donnée par :
\\[
\\hat{y} = \\sigma(W_3 \\mathbf{h}_2 + b_3) \\quad \\text{où} \\quad \\sigma(z) = \\frac{1}{1+e^{-z}}
\\]
            """
        )
        st.markdown("#### 1.4.3. Fonction de Perte et Entraînement")
        st.markdown("La fonction de perte utilisée est la Binary Cross-Entropy :")
        st.latex(r"\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)\right]")
        st.markdown("L'entraînement est réalisé via des algorithmes d'optimisation comme Adam ou SGD, à l'aide de frameworks tels que TensorFlow/Keras ou PyTorch.")

    # --- Ajout du MVP pour Méthodologie 1 ---
    st.subheader("1.5. MVP : Caractéristiques Fonctionnelles et Priorisation")
    st.markdown(
        """
**Caractéristiques fonctionnelles du MVP :**
- **Pipeline de Prétraitement et Vectorisation :** Mise en place d'un nettoyage, normalisation et vectorisation basique (TF-IDF) des textes.
- **Calcul de Similarité et Classement :** Implémentation d'un calcul simple de cosine similarity pour classer les produits.
- **Interface de Résultats :** Affichage rapide et lisible des résultats de recherche.

**Priorisation :**
- **Priorité haute :** Pipeline de prétraitement, vectorisation et calcul de similarité.
- **À réaliser ultérieurement :** Intégration de Word2Vec et raffinements via apprentissage supervisé (MLP).
        """
    )

# =============================================================================
# MÉTHODOLOGIE 2 : Filtrage collaboratif basé sur l'historique
# =============================================================================
elif choice == "Méthodologie 2 : Filtrage collaboratif basé sur l'historique":
    st.header("Méthodologie 2 : Filtrage Collaboratif Basé sur l’Historique Utilisateur")
    st.markdown(
        """
Cette approche exploite l’historique des interactions des utilisateurs (clics, achats, requêtes antérieures) pour modéliser leurs préférences et recommander des produits pertinents via des techniques de factorisation matricielle.
        """
    )
    st.subheader("2.1. Workflow Global de la Méthodologie 2")
    st.image("f.png", use_container_width=True)

    st.subheader("2.2. Pipeline de Collecte et Préparation des Données Utilisateur")
    with st.expander("Voir le détail de la collecte des données"):
        st.markdown(
            """
- **Extraction des Interactions :**  
  Enregistrement des clics, achats et requêtes passées.
  
- **Stockage :**  
  Les données sont stockées dans une base NoSQL ou dans un data lake pour faciliter le traitement en batch ou en temps réel.
  
- **Construction de la Matrice \\( R \\) :**  
  Matrice de dimensions \\( m \\times n \\) (\\( m \\) utilisateurs, \\( n \\) produits) avec des valeurs binaires ou pondérées.
            """
        )

    st.subheader("2.3. Factorisation Matricielle pour Extraction des Facteurs Latents")
    with st.expander("Voir les méthodes de factorisation"):
        st.markdown("#### 2.3.1. Singular Value Decomposition (SVD)")
        st.latex(r"R \approx U \Sigma V^T")
        st.markdown(
            "Chaque utilisateur \\( i \\) est représenté par le vecteur latent \\( \\mathbf{u}_i \\) et chaque produit \\( j \\) par \\( \\mathbf{v}_j \\). La prédiction de l’interaction est obtenue par :"
        )
        st.latex(r"\hat{R}_{ij} = \mathbf{u}_i \cdot \mathbf{v}_j")
        st.markdown("#### 2.3.2. Alternating Least Squares (ALS)")
        st.latex(r"\min_{U,V} \|R - UV^T\|_F^2 + \lambda(\|U\|_F^2 + \|V\|_F^2)")
    
    st.subheader("2.4. Prédiction et Recommandation")
    with st.expander("Détails sur le calcul des scores"):
        st.markdown("Pour un utilisateur \\( u \\) et un produit \\( j \\), le score est estimé par :")
        st.latex(r"\text{Score}_{u,j} = \mathbf{u}_u \cdot \mathbf{v}_j")
        st.markdown(
            "Ce score est ensuite combiné avec la similarité textuelle issue de la méthodologie 1 dans un système hybride :"
        )
        st.latex(r"\text{Score}_{\text{final}} = \alpha \times \text{Score}_{\text{collab}} + (1-\alpha) \times \text{Score}_{\text{texte}")
    
    st.subheader("2.5. Implémentation et Outils")
    st.markdown(
        """
- **Bibliothèques :**  
  Surprise, TensorFlow Recommenders, implicit.
  
- **Pipeline Global :**  
  1. Ingestion des logs pour construire la matrice \\( R \\).  
  2. Application de la factorisation (SVD ou ALS) pour obtenir \\( U \\) et \\( V \\).  
  3. Fusion avec la similarité textuelle pour générer le classement final des produits.
        """
    )
    
    # --- Ajout du MVP pour Méthodologie 2 ---
    st.subheader("2.6. MVP : Caractéristiques Fonctionnelles et Priorisation")
    st.markdown(
        """
**Caractéristiques fonctionnelles du MVP :**
- **Collecte des Interactions :** Extraction et stockage basique des données d'interaction utilisateur.
- **Construction de la Matrice Utilisateur-Produit :** Génération d'une matrice simple pour le filtrage collaboratif.
- **Factorisation de Base (SVD) :** Implémentation d'une factorisation par SVD pour obtenir des recommandations initiales.

**Priorisation :**
- **Priorité haute :** Collecte des données et mise en place d'une factorisation par SVD.
- **À réaliser ultérieurement :** Intégration d'algorithmes avancés (ALS) et fusion avec les scores textuels.
        """
    )

# =============================================================================
# MÉTHODOLOGIE 3 : Réseau de neurones léger pour le ré-ranking
# =============================================================================
elif choice == "Méthodologie 3 : Réseau de neurones léger pour le ré-ranking":
    st.header("Méthodologie 3 : Réseau de Neurones Léger pour le Ré-Ranking des Résultats")
    st.markdown(
        """
L’objectif est d’utiliser un réseau de neurones (MLP) pour réordonner les résultats de recherche en intégrant plusieurs critères tels que la similarité textuelle, la popularité du produit et l’historique utilisateur.
        """
    )
    st.subheader("3.1. Workflow Global de la Méthodologie 3")
    st.image("k.png", use_container_width=True)

    st.subheader("3.2. Extraction et Construction des Features")
    with st.expander("Détails sur l'extraction des features"):
        st.markdown(
            """
- **Features de Similarité Textuelle :**  
  Utilisation de TF-IDF ou d’extractions d’embeddings (BERT, Word2Vec) pour représenter la requête et le texte du produit.
  
  \\[
  \\text{Score}_{\\text{texte}} = \\frac{\\mathbf{E}_q \\cdot \\mathbf{E}_p}{\\|\\mathbf{E}_q\\| \\; \\|\\mathbf{E}_p\\|}
  \\]
  
- **Popularité du Produit :**  
  Mesurée via le nombre de ventes, évaluations, etc. (valeurs normalisées).
  
- **Score de Préférence Utilisateur :**  
  Par exemple, issu du filtrage collaboratif \\( \\mathbf{u}_i \\cdot \\mathbf{v}_j \\).

Ces features sont concaténées pour constituer le vecteur d'entrée \\( \\mathbf{x} \\).
            """
        )

    st.subheader("3.3. Architecture du Réseau de Neurones (MLP)")
    with st.expander("Détails de l'architecture du MLP"):
        st.markdown(
            """
**Structure du MLP :**
- **Couche d’Entrée :** Dimension \\( d_{in} \\) (nombre total de features).
- **Première Couche Cachée :** 64 neurones avec activation ReLU.
  
  \\[
  \\mathbf{h}_1 = \\text{ReLU}(W_1 \\mathbf{x} + b_1)
  \\]
  
- **Deuxième Couche Cachée :** 32 neurones avec activation ReLU.
  
  \\[
  \\mathbf{h}_2 = \\text{ReLU}(W_2 \\mathbf{h}_1 + b_2)
  \\]
  
- **Couche de Sortie :** 1 neurone avec activation sigmoïde pour obtenir un score de pertinence entre 0 et 1.
  
  \\[
  \\hat{y} = \\sigma(W_3 \\mathbf{h}_2 + b_3) \\quad \\text{où} \\quad \\sigma(z) = \\frac{1}{1+e^{-z}}
  \\]
            """
        )

    st.subheader("3.4. Fonction de Perte et Entraînement")
    with st.expander("Détails sur la fonction de perte et l'optimisation"):
        st.markdown("**Fonction de perte :** Binary Cross-Entropy")
        st.latex(r"\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]")
        st.markdown("**Optimisation :** Utilisation de l’algorithme Adam (ou SGD) pour entraîner le modèle avec TensorFlow/Keras ou PyTorch.")
        st.markdown("Lorsqu'une requête est effectuée, le système extrait les features, construit le vecteur \\( \\mathbf{x} \\), passe ce vecteur dans le MLP pour obtenir un score \\( \\hat{y} \\) et réordonne les résultats en fonction de ce score.")

    # --- Ajout du MVP pour Méthodologie 3 ---
    st.subheader("3.5. MVP : Caractéristiques Fonctionnelles et Priorisation")
    st.markdown(
        """
**Caractéristiques fonctionnelles du MVP :**
- **Extraction des Features Essentielles :** Mise en place d'un système basique pour extraire les principales features (similarité textuelle, popularité).
- **Implémentation d'un MLP Simple :** Construction d'un réseau de neurones léger pour réordonner les résultats.
- **Affichage du Ré-Ranking :** Réorganisation des résultats selon le score obtenu.

**Priorisation :**
- **Priorité haute :** Extraction des features essentielles et mise en place du MLP de base.
- **À réaliser ultérieurement :** Intégration de critères supplémentaires et optimisation de l'architecture du réseau.
        """
    )

# =============================================================================
# SYNTHÈSE
# =============================================================================
elif choice == "Synthèse":
    st.header("Synthèse des Méthodologies")
    st.markdown(
        """
Ce document présente en profondeur trois méthodologies complémentaires pour améliorer la pertinence des résultats de recherche :

1. **Filtrage basé sur le contenu avec NLP :**  
   Une approche textuelle qui transforme les descriptions et requêtes en représentations numériques (via TF-IDF ou Word2Vec), calcule la similarité par cosine similarity et peut être raffinée par un apprentissage supervisé (MLP).

2. **Filtrage collaboratif basé sur l’historique utilisateur :**  
   Exploite l’historique des interactions pour construire une matrice utilisateur-produit, puis applique une factorisation matricielle (SVD ou ALS) afin d’extraire des facteurs latents et combiner ces informations avec des mesures textuelles dans un système hybride.

3. **Réseau de neurones léger pour le ré-ranking :**  
   Utilise un MLP pour intégrer diverses features (similarité textuelle, popularité, score collaboratif) afin de réordonner les résultats de recherche en temps réel, en produisant un score de pertinence exploitable.
        """
    )

# =============================================================================
# Informations générales dans la sidebar
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.info(
    "Ce rapport présente trois méthodologies détaillées dans le cadre du développement d'une solution d'amélioration des résultats de recherche pour Rakuten. "
    "Utilisez le menu pour naviguer entre les différentes sections"
)
