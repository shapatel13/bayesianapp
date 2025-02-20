import streamlit as st
import json
from agno.agent import Agent
from agno.models.google import Gemini
from typing import List, Dict

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'agent' not in st.session_state:
        st.session_state.agent = Agent(
            name="BayesianClinicalReasoner",
            model=Gemini(
                id="gemini-2.0-flash-exp",
                api_key='AIzaSyDmcPbEDAEojTomYs7vLKu107fOa7c6500'
            ),
            markdown=True,
            read_chat_history=True,
            add_history_to_messages=True,
            description="""You are an expert clinical reasoner using advanced Bayesian thinking to assist physicians with complex medical decision-making. 
            You combine sophisticated probabilistic reasoning with deep medical knowledge to provide precise quantitative assessments in clinical scenarios.""",
            instructions=[
                "Probabilistic Reasoning:",
                "- Always start with explicit pre-test probability estimates based on epidemiology and patient factors",
                "- Provide specific likelihood ratios (LR+/LR-) for key clinical findings, labs, and imaging",
                "- Calculate post-test probabilities using Bayes theorem, showing your work",
                "- State probability thresholds for key clinical decisions (testing vs treating)",
                
                "Clinical Decision Analysis:", 
                "- Break down complex decisions into clear components: actions, uncertainties, outcomes",
                "- Quantify utilities for different clinical outcomes where relevant",
                "- Calculate expected utility of different clinical strategies",
                "- Present clear decision thresholds based on probability/utility analysis",
                
                "Risk Assessment:",
                "- Provide specific probability estimates with explicit confidence intervals",
                "- Distinguish between population statistics and individual patient factors",
                "- Acknowledge uncertainty while still providing actionable probability estimates",
                "- Factor in both sensitivity/specificity of tests and patient-specific factors",
                
                "Cognitive Bias Mitigation:",
                "- Flag potential availability bias and representativeness heuristic traps",
                "- Encourage consideration of base rates and pre-test probabilities",
                "- Question intuitive probability estimates that may be biased",
                "- Push back on anecdotal reasoning with statistical thinking",
                
                "Communication:",
                "- Present probability calculations clearly showing each step",
                "- Express probabilities as natural frequencies (e.g., '20 out of 100 patients')",
                "- Show your mathematical work for all probability calculations",
                "- Explain statistical concepts in clinically relevant terms"
            ]
        )

def clear_chat():
    st.session_state.messages = []
    if 'agent' in st.session_state:
        st.session_state.agent.memory.clear()

def main():
    st.set_page_config(
        page_title="BayesianClinicalReasoner",
        page_icon="🏥",
        layout="wide"
    )

    initialize_session_state()

    st.title("🏥 BayesianClinicalReasoner")
    st.markdown("""
    > Advanced Clinical Decision Support using Bayesian reasoning and probabilistic analysis
    """)

    # Sidebar
    with st.sidebar:
        st.header("Commands")
        if st.button("🔄 New Case", key="new_case"):
            clear_chat()
            st.rerun()
            
        if st.button("📋 View History", key="history"):
            if st.session_state.messages:
                st.json([m for m in st.session_state.agent.memory.messages])
            else:
                st.info("No chat history yet")
                
        st.header("Example Queries")
        st.markdown("""
        1. Calculate pre/post-test probability for pulmonary embolism given these findings...
        2. What's the testing threshold for bacterial meningitis given risks/benefits...
        3. Help me estimate likelihood ratios for this complex neurological presentation...
        """)
        
        st.header("About")
        st.markdown("""
        This AI assistant provides:
        - 📊 Probabilistic reasoning
        - 🧮 Bayesian calculations
        - 🔍 Clinical decision analysis
        - ⚖️ Risk assessment
        - 🎯 Bias mitigation
        """)

    # Main chat interface
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Enter your clinical query..."):
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    response = st.session_state.agent.run(prompt)
                    st.markdown(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})

    # Footer
    st.markdown("---")
    st.markdown(
        "*Powered by advanced AI for evidence-based clinical decision support*",
        help="Uses Gemini with Bayesian reasoning capabilities"
    )

if __name__ == "__main__":
    main()
