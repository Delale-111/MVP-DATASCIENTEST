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

# =============================================================================
# SYNTHÈSE et Cahier des Charges du MVP
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
    
    # Nouvelle section : Cahier des Charges du MVP et Priorisation des Fonctionnalités
    st.markdown("---")
    st.header("4. Cahier des Charges du MVP et Priorisation des Fonctionnalités")
    
    st.subheader("4.1. Objectifs du MVP")
    st.markdown(
        """
L'objectif du MVP (Produit Minimal Viable) est de délivrer une version opérationnelle et fonctionnelle du moteur de recherche intelligent pour Rakuten. 
Ce MVP doit permettre de valider l'approche initiale basée sur le filtrage par contenu (NLP), en offrant une recherche pertinente et réactive, tout en posant les bases pour l'intégration future de techniques avancées (filtrage collaboratif et ré-ranking par réseau de neurones).
        """
    )
    
    st.subheader("4.2. Caractéristiques Fonctionnelles du MVP")
    st.markdown(
        """
**a) Recherche de Produits Basée sur le Contenu**
- **Saisie et Prétraitement de la Requête :**
  - Interface de saisie permettant à l'utilisateur d'entrer sa recherche.
  - Nettoyage et normalisation de la requête (mise en minuscules, suppression des caractères spéciaux, élimination des stopwords).
  
- **Prétraitement et Vectorisation des Textes Produits :**
  - Extraction des titres et descriptions des produits.
  - Application d’un pipeline de nettoyage similaire à celui de la requête.
  - Vectorisation à l’aide d’outils tels que **TF-IDF** (avec possibilité d’extension ultérieure vers Word2Vec ou d’autres embeddings).
  
- **Calcul de la Similarité et Classement :**
  - Utilisation de la cosine similarity pour mesurer la proximité entre la requête et les produits.
  - Classement des résultats par ordre décroissant de pertinence et affichage des top N résultats.

**b) Interface Utilisateur Simple et Intuitive**
- Page de recherche épurée avec une barre de saisie.
- Présentation claire des résultats (titre, description, et éventuellement image ou score de pertinence).
- Réactivité et simplicité pour garantir une prise en main rapide par l'utilisateur.

**c) Gestion Basique des Données et Logistique**
- Importation et prétraitement des données produits via des fichiers plats ou une base de données légère.
- Mise à jour périodique de l’index produit pour intégrer les nouveautés.
- Journalisation des requêtes pour collecter des retours et orienter les améliorations futures.

**d) Suivi et Mesure de la Performance**
- Monitoring des temps de réponse (objectif : moins de 2 secondes par requête).
- Enregistrement des logs d’utilisation pour analyser l’efficacité de l’algorithme de classement et la pertinence des résultats.
        """
    )
    
    st.subheader("4.3. Priorisation des Fonctionnalités")
    st.markdown(
        """
**Fonctionnalités Prioritaires (à livrer dans le MVP) :**
1. **Pipeline de Prétraitement et Vectorisation :**
   - Mise en œuvre du nettoyage, normalisation et vectorisation des textes produits et des requêtes via TF-IDF.
2. **Mécanisme de Calcul de la Similarité et Classement :**
   - Implémentation du calcul de la cosine similarity pour établir un classement initial.
3. **Interface de Recherche Basique :**
   - Conception d'une interface utilisateur simple et responsive pour saisir la requête et visualiser les résultats.
4. **Journalisation des Requêtes :**
   - Mise en place d’un système de logging permettant de collecter les interactions utilisateurs pour affiner le modèle.

**Fonctionnalités à Développer ultérieurement :**
1. **Filtrage Collaboratif (Méthodologie 2) :**
   - Intégration d’un module de recommandation basé sur l’historique utilisateur lorsque suffisamment de données seront collectées.
2. **Ré-Ranking par Réseau de Neurones (Méthodologie 3) :**
   - Déploiement d’un MLP pour réordonner les résultats en combinant plusieurs critères (similarité textuelle, popularité, préférences utilisateur).
3. **Optimisation Avancée de la Performance :**
   - Mise en place d’index inversés ou utilisation de bibliothèques spécialisées (ex. FAISS) pour le traitement de gros volumes de données.

**Fonctionnalités à Écarter Définitivement ou à Retarder :**
- Les fonctionnalités nécessitant un volume important de données ou un entraînement complexe (comme le ré-ranking par MLP) pourront être repoussées à une version ultérieure si elles impactent la réactivité ou la stabilité du MVP.
- Les options d’interface utilisateur avancées (filtres dynamiques, recommandations visuelles personnalisées) seront envisagées seulement si elles apportent une valeur significative par rapport à leur complexité de mise en œuvre.
        """
    )
    
    st.subheader("4.4. Spécifications Techniques du MVP")
    st.markdown(
        """
**Performance et Scalabilité :**
- Temps de réponse par requête inférieur à 2 secondes.
- Capacité à gérer un volume modéré de requêtes simultanées (scalabilité horizontale envisagée pour les phases ultérieures).

**Technologies et Outils :**
- **Backend :** Python avec scikit-learn (pour le TF-IDF), NumPy et éventuellement FAISS pour l'optimisation du calcul de similarité.
- **Frontend :** Streamlit pour une mise en place rapide et une interface utilisateur simple.
- **Infrastructure :** Serveur de test évolutif vers une solution cloud (selon le volume de trafic).
- **Stockage :** Fichiers plats (CSV) ou base NoSQL légère (par exemple, MongoDB) pour les données produits.

**Sécurité et Fiabilité :**
- Gestion robuste des erreurs lors du prétraitement et de la recherche.
- Journalisation détaillée des requêtes et des incidents pour faciliter la maintenance.
- Tests unitaires et d’intégration afin de garantir la stabilité du système.

**Méthodologie de Développement :**
- Développement en cycles itératifs avec des feedbacks réguliers des utilisateurs.
- Documentation technique et fonctionnelle complète pour assurer la pérennité et la maintenabilité du projet.
        """
    )
    
    st.subheader("4.5. Conclusion")
    st.markdown(
        """
Le MVP a pour vocation de valider l'approche de filtrage par contenu comme socle de l'amélioration des résultats de recherche pour Rakuten. 
En se concentrant sur les fonctionnalités essentielles – prétraitement et vectorisation des textes, calcul de similarité, interface utilisateur basique et journalisation – nous garantissons une mise en œuvre rapide et efficace, tout en ouvrant la voie à des améliorations futures (filtrage collaboratif et ré-ranking par MLP) en fonction des retours et des évolutions du projet.
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
