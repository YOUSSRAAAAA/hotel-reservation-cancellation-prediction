import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder

# Charger le modèle LightGBM
model = joblib.load('lightgbm_best_model.pkl')

# Fonction pour effectuer la prédiction avec probabilité
def predict_cancellation(data):
    # Obtenir la probabilité de prédiction (probabilité que la réservation soit annulée)
    prob = model.predict_proba(data)[0][1]
    # Effectuer la prédiction de la classe
    prediction = model.predict(data)
    return 'Canceled' if prediction[0] == 1 else 'Not Canceled', prob

# Fonction pour encoder les colonnes catégorielles
def encode_categorical_columns(data):
    # Initialisation du LabelEncoder
    le = LabelEncoder()

    # Encoder 'market_segment' et 'deposit_type' en valeurs numériques
    data['market_segment'] = le.fit_transform(data['market_segment'])
    data['deposit_type'] = le.fit_transform(data['deposit_type'])

    return data

# Fonction pour ajouter les colonnes manquantes
def align_columns(input_data):
    required_columns = ['hotel_encoded', 'booking_location_encoded', 'lead_time',
                        'market_segment', 'deposit_type', 'total_of_special_requests',
                        'is_previously_cancelled', 'is_repeated_guest', 'is_booking_changes',
                        'customer_type', 'total_stays', 'guests', 'other_column_1', 'other_column_2',
                        'other_column_3', 'other_column_4', 'other_column_5', 'other_column_6',
                        'other_column_7', 'other_column_8', 'other_column_9', 'other_column_10',
                        'other_column_11']

    # Ajouter les colonnes manquantes avec des valeurs par défaut
    for col in required_columns:
        if col not in input_data.columns:
            input_data[col] = 0  # Ajouter une colonne avec valeur par défaut 0 ou d'autres valeurs si nécessaire

    # S'assurer que les colonnes sont dans le bon ordre
    input_data = input_data[required_columns]

    return input_data


# Titre de l'application
st.markdown("""
    <h1 style='text-align: center; color: #2C3E50; font-weight: bold; font-size: 48px; font-family: "Arial", sans-serif; padding-top: 20px;'>
        Prédiction des Annulations de Réservations d'Hôtel
    </h1>
""", unsafe_allow_html=True)

# Ajouter une description stylisée
st.markdown("""
    <p style='font-size: 15px; text-align: center; color: #7F8C8D; font-family: "Arial", sans-serif; margin-top: 10px;'>
        Entrez les détails de la réservation pour prédire si elle sera annulée.
        Les prévisions sont basées sur des informations clés concernant la réservation et les clients.
    </p>
""", unsafe_allow_html=True)

# Ajouter une image large et stylisée
st.image("hotel.jpg", use_container_width=True, caption="Réservations d'Hôtel", output_format="auto")

## Ajout d'un design supplémentaire avec un fond bleu clair et des bordures
st.markdown("""
    <style>
    .stButton>button {
        background-color: #1ABC9C;
        color: white;
        font-size: 18px;
        padding: 15px;
        border-radius: 12px;
        width: 100%;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #16A085;
    }
    .stSidebar {
        background-color: #AED6F1;  /* Bleu clair pour la barre latérale */
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.1);
    }
    .stTextInput>label {
        font-size: 18px;
        color: #34495E;
    }
    .stTextInput>div>input {
        font-size: 18px;
        border-radius: 8px;
    }
    .stRadio>div>label {
        font-size: 18px;
        color: #34495E;
    }
    </style>
""", unsafe_allow_html=True)


# Création de la barre latérale avec des outils de saisie
with st.sidebar:
    st.header("Détails de la réservation")

    # Champs supplémentaires pour saisir des informations détaillées
    hotel = st.selectbox("Type d'hôtel", ["City Hotel", "Resort Hotel"], key="hotel_selectbox")
    is_repeated_guest = st.selectbox("Client répétitif", ["Non", "Oui"], key="repeated_guest_selectbox")
    adults = st.number_input("Nombre d'adultes", min_value=1, max_value=10, step=1, key="adults_input")
    children = st.number_input("Nombre d'enfants", min_value=0, max_value=10, step=1, key="children_input")
    adr = st.number_input("ADR (Prix par nuit)", min_value=0.0, max_value=1000.0, step=0.1, key="adr_input")
    booking_changes = st.number_input("Modifications de réservation", min_value=0, max_value=10, step=1,
                                      key="booking_changes_input")
    special_requests = st.number_input("Demandes spéciales", min_value=0, max_value=5, step=1,
                                       key="special_requests_input")

    # Autres champs pour plus de détails
    lead_time = st.number_input("Temps de réservation avant l'arrivée (en jours)", min_value=1, max_value=365, step=1)
    deposit_type = st.selectbox("Type de dépôt", ["No Deposit", "Non Refund", "Refundable"])
    market_segment = st.selectbox("Segment de marché", ["Online", "Offline", "Corporate", "Direct"])
    is_previously_cancelled = st.selectbox("A été annulé auparavant", ["Non", "Oui"])

# Design supplémentaire avec un fond et des bordures pour les boutons
st.markdown("""
    <style>
    .stButton>button {
        background: linear-gradient(to right, #1D4ED8, #60A5FA); /* Dégradé de bleu */
        color: white;
        font-size: 18px;
        padding: 12px;
        border-radius: 8px;
        width: 100%;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        border: none;
        outline: none;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #1E3A8A, #3B82F6); /* Bleu foncé pour effet hover */
        transform: scale(1.02);
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:active {
        transform: scale(0.98);
    }
    .result-box {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        padding: 20px;
        text-align: center;
        margin-top: 20px;
    }
    .result-box.canceled {
        border-left: 6px solid #FF5733; /* Rouge pour annulation */
    }
    .result-box.not-canceled {
        border-left: 6px solid #28B463; /* Vert pour non-annulation */
    }
    .reset-btn {
        background-color: #F39C12;
        color: white;
        font-size: 18px;
        padding: 12px;
        border-radius: 8px;
        width: 100%;
    }
    .reset-btn:hover {
        background-color: #E67E22;
    }
    </style>
""", unsafe_allow_html=True)


# Ajouter une section avec un bouton de prédiction
col1, col2 = st.columns([1, 3])  # Répartition pour un design mobile
with col1:
    # Ajouter un bouton pour revenir ou réinitialiser
    if st.button("Réinitialiser", key="reset_btn", help="Réinitialiser les champs"):
        # Réinitialiser les valeurs de session
        st.session_state.clear()  # Réinitialise tout le session_state

        # Optionnellement, vous pouvez réinitialiser les champs individuels
        st.session_state['hotel_selectbox'] = 'City Hotel'
        st.session_state['repeated_guest_selectbox'] = 'Non'
        st.session_state['adults_input'] = 1
        st.session_state['children_input'] = 0
        st.session_state['adr_input'] = 0.0
        st.session_state['booking_changes_input'] = 0
        st.session_state['special_requests_input'] = 0
        st.session_state['lead_time'] = 1
        st.session_state['deposit_type'] = 'No Deposit'
        st.session_state['market_segment'] = 'Online'
        st.session_state['is_previously_cancelled'] = 'Non'

with col2:
    # Ajouter un bouton de prédiction centré
    if st.button("Prédire l'annulation", key="predict_btn"):
        # Préparer les données pour la prédiction
        input_data = pd.DataFrame({
            'hotel_encoded': [0 if hotel == "City Hotel" else 1],
            'booking_location_encoded': [0],  # Remplacer avec vos données
            'lead_time': [lead_time],
            'market_segment': [market_segment],
            'deposit_type': [deposit_type],
            'total_of_special_requests': [special_requests],
            'is_previously_cancelled': [1 if is_previously_cancelled == "Oui" else 0],
            'is_repeated_guest': [1 if is_repeated_guest == "Oui" else 0],
            'is_booking_changes': [booking_changes],
            'customer_type': [0],  # Remplacer avec vos données
            'total_stays': [adults + children],
            'guests': [adults + children],
        })

        # Compléter les colonnes manquantes
        input_data = align_columns(input_data)

        # Encoder les colonnes catégorielles
        input_data = encode_categorical_columns(input_data)

        # Effectuer la prédiction et obtenir la probabilité
        result, prob = predict_cancellation(input_data)

        # Afficher le résultat avec couleur en fonction de l'annulation
        if result == 'Canceled':
            st.markdown(f"<h2 style='color: red; text-align: center;'>La réservation est : <strong>{result}</strong></h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color: green; text-align: center;'>La réservation est : <strong>{result}</strong></h2>", unsafe_allow_html=True)

        st.markdown(f"<h3 style='text-align: center;'>Probabilité d'annulation : {prob * 100:.2f}%</h3>",
                    unsafe_allow_html=True)
        # Affichage de la probabilité sous forme de barre
        st.progress(prob)

# Messages d'information
st.info(
    "Les prévisions sont basées sur les informations entrées. Veuillez remplir tous les champs pour une prédiction plus précise.")
