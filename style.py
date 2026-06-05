import streamlit as st

def aplicar_estilo():
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top left, #202959 0%, #060A21 30%, #040214 100%);
        }

        .block-container {
            padding-top: 5rem;
            animation: fadeIn 0.8s ease-in-out;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(8px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        h1 {
            background: radial-gradient(circle at top left, #75B8FF 0%, #DCEDFC 60%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        [data-testid="stMetric"] {
            background-color: #060A21;
            border: 1px solid rgba(117, 184, 255, 0.25);
            padding: 18px;
            border-radius: 18px;
            box-shadow: 0 0 18px rgba(37, 64, 143, 0.25);
        }

        .stButton > button {
            border-radius: 12px;
            border: 1px solid rgba(37, 64, 143, 0.5);
            background: linear-gradient(135deg, #25408F, #202959);
            padding: 0.6rem 1.2rem;
            transition: all 0.25s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 0 18px rgba(117, 184, 255, 0.5);
            border-color: #75B8FF;
        }
        </style>
        """, unsafe_allow_html=True
    )