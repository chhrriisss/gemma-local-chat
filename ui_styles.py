import streamlit as st

def apply_custom_styling():
    """Apply modern UI styling to the Streamlit app"""
    st.markdown("""
    <style>

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Main app styling */
        .main > div {
            padding-top: 1rem;
        }
        
        /* Header styling */
        .app-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        }
        
        .app-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }
        
        .app-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        /* Chat styling */
        .chat-message {
            padding: 1rem 1.5rem;
            border-radius: 18px;
            margin: 0.5rem 0;
            max-width: 80%;
            word-wrap: break-word;
        }
        
        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: auto;
            border-radius: 18px 18px 5px 18px;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }
        
        .assistant-message {
            background: white;
            color: #333;
            border-radius: 18px 18px 18px 5px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        /* Status cards */
        .status-card {
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            text-align: center;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .status-success {
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
        }
        
        .status-warning {
            background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
            color: white;
        }
        
        .status-info {
            background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
            color: white;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            padding: 0.75rem 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        .stButton > button:active {
            transform: translateY(0);
        }
        
        /* File uploader styling */
        .uploadedFile {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 10px;
            border: 2px dashed #667eea;
            padding: 1rem;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .uploadedFile:hover {
            border-color: #764ba2;
            background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
        }
        
        /* Dataframe styling */
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        /* Code block styling */
        .stCode {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid #00d4aa;
            border-radius: 10px;
            box-shadow: 0 8px 25px rgba(0, 212, 170, 0.3);
        }
        
        /* Metrics styling */
        .metric-container {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            border-top: 4px solid #667eea;
            margin: 0.5rem 0;
        }
        
        /* Input styling */
        .stTextInput > div > div > input {
            border-radius: 25px;
            border: 2px solid #e0e0e0;
            padding: 0.75rem 1rem;
            transition: all 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #667eea;
            box-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 10px;
            border: 1px solid #dee2e6;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 10px;
            padding: 0.5rem 1rem;
            border: 1px solid #dee2e6;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        /* Success/Error message styling */
        .stSuccess {
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            border-radius: 10px;
            border: none;
        }
        
        .stError {
            background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
            border-radius: 10px;
            border: none;
        }
        
        .stWarning {
            background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
            border-radius: 10px;
            border: none;
        }
        
        .stInfo {
            background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
            border-radius: 10px;
            border: none;
        }
        
        /* Spinner styling */
        .stSpinner {
            color: #667eea;
        }
        
        /* Remove padding from main container */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .app-header h1 {
                font-size: 2rem;
            }
            
            .chat-message {
                max-width: 95%;
            }
            
            .stButton > button {
                font-size: 0.9rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """Render a beautiful header"""
    st.markdown("""
    <div class="app-header">
        <h1>🤖 AI Data Assistant</h1>
        <p>Upload CSV files and get instant Excel formulas & data insights</p>
    </div>
    """, unsafe_allow_html=True)

def render_status_card(status_type, message):
    """Render status cards with different styles"""
    st.markdown(f'<div class="status-card status-{status_type}">{message}</div>', 
                unsafe_allow_html=True)