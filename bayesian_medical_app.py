import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini

# Configure API key
API_KEY = "AIzaSyDmcPbEDAEojTomYs7vLKu107fOa7c6500"

# Initialize the Bayesian Clinical Reasoner
clinical_reasoner = Agent(
    name="BayesianClinicalReasoner",
    model=Gemini(id="gemini-2.0-flash-exp", api_key=API_KEY),
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

def initialize_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

def format_message_for_agent(role: str, content: str) -> dict:
    return {
        "role": "user" if role == "user" else "assistant",
        "content": content
    }

def main():
    st.set_page_config(
        page_title="BayesianClinicalReasoner",
        page_icon="🏥",
        layout="wide"
    )

    initialize_chat_history()

    st.title("🏥 BayesianClinicalReasoner")
    st.markdown("""
    > Advanced Clinical Decision Support using Bayesian reasoning and probabilistic analysis
    """)

    # Sidebar
    with st.sidebar:
        st.header("Commands")
        if st.button("🔄 New Case"):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.rerun()
            
        st.header("Example Queries")
        st.markdown("""
        1. Calculate pre/post-test probability for pulmonary embolism given these findings...
        2. What's the testing threshold for bacterial meningitis given risks/benefits...
        3. Help me estimate likelihood ratios for this complex neurological presentation...
        """)

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your clinical query..."):
        # Add user message to display
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Add to conversation history
        st.session_state.conversation_history.append(
            format_message_for_agent("user", prompt)
        )

        # Generate response with context
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = clinical_reasoner.run(
                    prompt,
                    messages=st.session_state.conversation_history  # Pass conversation history
                )
                st.markdown(response.content)
                
        # Add assistant response to histories
        st.session_state.messages.append({"role": "assistant", "content": response.content})
        st.session_state.conversation_history.append(
            format_message_for_agent("assistant", response.content)
        )

    st.markdown("---")
    st.markdown(
        "*Powered by Gemini with Bayesian reasoning capabilities*",
        help="Uses advanced AI for clinical decision support"
    )

if __name__ == "__main__":
    main()
