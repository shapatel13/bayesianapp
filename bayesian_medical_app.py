import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini

# --- Configuration & Setup ---
try:
    API_KEY = 'AIzaSyDmcPbEDAEojTomYs7vLKu107fOa7c6500'
except (FileNotFoundError, KeyError):
    st.error("API Key not found. Please create a .streamlit/secrets.toml file with your API_KEY.")
    st.stop()

# --- Agent Definition (Simplified but Powerful) ---
clinical_reasoner = Agent(
    name="ProbabilisticRounds",
    model=Gemini(id="gemini-2.5-flash-preview-04-17", api_key=API_KEY),
    markdown=True,
    read_chat_history=True,
    add_history_to_messages=True,
    description="You are 'Probabilistic Rounds', a senior academic physician and biostatistician...",
    # The prompt is kept, but modified to remove references to tools.
    instructions=[
        """
        You are 'Probabilistic Rounds', a senior academic physician and biostatistician. Your role is to guide the user, a capable physician, through complex cases using rigorous quantitative methods. 
        
        You must base your analysis on established medical knowledge and state the basis for your quantitative estimates (e.g., 'based on landmark studies like PIOPED II,' or 'derived from general epidemiological data'). You do not have access to live internet search.

        You must structure EVERY response using the following Markdown template precisely.

        ---

        **Executive Summary:** [Start with a single, bolded sentence summarizing the most critical clinical conclusion.]

        ---

        ### 1. Pre-Test Probability Assessment
        * **Initial Estimate:** [State the pre-test probability as a percentage]
        * **Rationale & Sources:** [Explain WHY you chose this probability, citing well-known epidemiological data or clinical prediction rules from your internal knowledge.]

        ### 2. Bayesian Update with New Information
        *You must present this as a table:*
        | Finding/Test | Result | Likelihood Ratio (95% CI) | Source |
        | :--- | :--- | :--- | :--- |
        | [Symptom 1] | [e.g., Present] | [e.g., 2.5 (1.8-3.4)] | [e.g., From established literature] |

        ### 3. Post-Test Probability Calculation
        * **Pre-Test Odds:** [Show conversion: P / (1 - P)]
        * **Likelihood Multiplier:** [Show the product of all LRs]
        * **Post-Test Odds:** [Show calculation: Pre-Test Odds * Likelihood Multiplier]
        * **Final Post-Test Probability:** [Show conversion back to percentage: Odds / (1 + Odds)]
        * **Plausible Probability Range:** [State a range if CIs are available from your knowledge base]

        ### 4. Clinical Interpretation & Next Steps
        * **Interpretation:** [Explain what this new probability means clinically.]
        * **Recommended Action:** [Suggest the single most logical next step.]
        * **Cognitive Bias Check:** [Explicitly flag a potential bias.]
        * **Devil's Advocate:** [Briefly argue for an alternative diagnosis.]
        """
    ]
    # NOTE: No tools or validator are passed to the agent for maximum stability.
)

# --- App Logic (Simplified) ---

def initialize_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

def format_message_for_agent(role: str, content: str) -> dict:
    return {"role": "user" if role == "user" else "assistant", "content": content}

def main():
    st.set_page_config(page_title="BayesianClinicalReasoner (Stable)", page_icon="🩺", layout="wide")
    initialize_chat_history()
    st.title("🩺 BayesianClinicalReasoner (Stable Version)")
    st.markdown("> A streamlined clinical reasoning partner. Focused on stability and quality of output.")

    with st.sidebar:
        st.header("Commands")
        if st.button("🔄 New Case"):
            st.session_state.messages, st.session_state.conversation_history = [], []
            st.rerun()
        st.header("Example Queries")
        st.markdown("""
        1.  `55yo M with acute pleuritic chest pain and a heart rate of 105. Calculate the post-test probability for PE based on known LRs.`
        2.  `A patient has a new rash. They recently started taking Lamotrigine. Work up the probability of SJS.`
        """)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your clinical query..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.conversation_history.append(format_message_for_agent("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Reverted to the simple, stable .run() method with a spinner.
            with st.spinner("Analyzing..."):
                try:
                    response = clinical_reasoner.run(
                        prompt, messages=st.session_state.conversation_history
                    )
                    response_content = response.content
                    st.markdown(response_content)
                except Exception as e:
                    response_content = f"An error occurred: {str(e)}"
                    st.error(response_content)

        st.session_state.messages.append({"role": "assistant", "content": response_content})
        st.session_state.conversation_history.append(
            format_message_for_agent("assistant", response_content)
        )

if __name__ == "__main__":
    main()
