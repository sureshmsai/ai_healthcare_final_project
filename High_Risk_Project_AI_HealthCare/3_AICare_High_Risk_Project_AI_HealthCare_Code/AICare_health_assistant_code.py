from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

import pandas as pd
import networkx as nx
import google.generativeai as genai
from dotenv import load_dotenv
import os
import gradio as gr
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score


nltk.download('wordnet')

load_dotenv()  # Load environment variables from .env file

# --- 1. Data Acquisition & Knowledge Graph Construction ---
def create_knowledge_graph():
    """
    Create a knowledge graph from cardiac arrest data.

    Reads cardiac arrest data from a CSV file, processes it, and constructs a directed graph
    where patients and diagnoses are represented as nodes, and relationships between them are
    represented as edges.

    Returns:
        networkx.DiGraph: A directed graph representing the knowledge graph.
    """
    # Read the MIMIC-III sample dataset (cardiac arrest, Acute kidney injury, and sepsis)
    mimic_data = pd.read_csv("mimic_health_data.csv")
    mimic_data_sample = mimic_data.drop_duplicates()

    graph = nx.DiGraph()

    # Add patients as nodes with additional attributes
    for index, row in mimic_data_sample.iterrows():
        Patient = int(row['Patient'])  # Ensure Patient ID is an integer
        graph.add_node(Patient, type="patient", 
                       gender=row['Gender'], dob=row['DOB'], dod=row['DOD'], expire_flag=row['Expire_Flag'],
                       admit_time=row['Admit_Time'], discharge_time=row['Discharge_Time'], death_time=row['Death_Time'],
                       admission_type=row['Admission_Type'], admission_location=row['Admission_location'], 
                       discharge_location=row['Discharge_location'], insurance=row['Insurance'], 
                       marital_status=row['Marital_Status'], diagnosis=row['Diagnosis'], 
                       hospital_expire_flag=row['Hospital_Expire_Flag'], drug=row['Drug'])

    # Add diagnoses as nodes and edges
    for index, row in mimic_data_sample.iterrows():
        Patient = int(row['Patient'])  # Ensure Patient ID is an integer
        ICD9_Code = row['ICD9_Code']
        graph.add_node(ICD9_Code, type="diagnosis", lab_label=row['Lab_Label'], chart_time=row['Chart_Time'], 
                       lab_value=row['Lab_Value'], lab_flag=row['Lab_Flag'], category=row['Category'], 
                       drug=row['Drug'], text=row['Text'])
        graph.add_edge(Patient, ICD9_Code, relation="has_diagnosis", admission=row['Admission'])

    return graph  # Return the graph object

# --- 2. Multi-Agent System Implementation ---

class MedicalHistoryAgent:
    """
    Agent responsible for retrieving patient medical history from the knowledge graph.
    """
    def __init__(self, graph):
        """
        Initialize the MedicalHistoryAgent.

        Args:
            graph (networkx.DiGraph): The knowledge graph containing patient and diagnosis data.
        """
        self.graph = graph
        self.icd9_descriptions = {
            '4275': 'Cardiac arrest',
            '5849': 'Acute kidney failure',
            '99592': 'Severe sepsis'
        }

    def get_patient_history(self, patient_id):
        """
        Retrieve the medical history of a patient.

        Args:
            patient_id (int): The ID of the patient.

        Returns:
            str: A string describing the patient's diagnoses.
        """
        diagnoses = []
        for n, d in self.graph.nodes(data=True):
            if d["type"] == "diagnosis" and self.graph.has_edge(patient_id, n):
                diagnosis_code = str(n)
                diagnosis_desc = self.icd9_descriptions.get(diagnosis_code, "Unknown diagnosis")
                diagnoses.append(f"{diagnosis_code} ({diagnosis_desc})")
        return f"Patient {patient_id} has the following diagnoses: {', '.join(diagnoses)}"

class RiskAssessmentAgent:
    """
    Agent responsible for assessing health risks based on patient medical history.
    """
    def __init__(self):
        """
        Initialize the RiskAssessmentAgent and configure the generative AI model.
        """
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-pro')

    def assess_risk(self, history):
        """
        Assess potential health risks based on medical history.

        Args:
            history (str): The medical history of the patient.

        Returns:
            str: A string describing the potential health risks.
        """
        prompt = f"Given the medical history: '{history}', identify potential health risks."
        response = self.model.generate_content(prompt)
        return response.text

class RecommendationAgent:
    """
    Agent responsible for generating health recommendations based on assessed risks.
    """
    def __init__(self):
        """
        Initialize the RecommendationAgent and configure the generative AI model.
        """
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-pro')

    def generate_recommendations(self, risks):
        """
        Generate health recommendations based on assessed risks.

        Args:
            risks (str): The assessed health risks.

        Returns:
            str: A string describing the health recommendations.
        """
        prompt = f"Based on the risks: '{risks}', suggest preventive measures or recommendations."
        response = self.model.generate_content(prompt)
        return response.text

class KnowledgeRetrievalAgent:
    """
    Agent responsible for retrieving context and information from the knowledge graph.
    """
    def __init__(self, graph):
        """
        Initialize the KnowledgeRetrievalAgent.

        Args:
            graph (networkx.DiGraph): The knowledge graph containing patient and diagnosis data.
        """
        self.graph = graph

    def retrieve_context(self, patient_id, keywords):
        """
        Retrieve context information for a patient based on keywords.

        Args:
            patient_id (int): The ID of the patient.
            keywords (list): A list of keywords to search for in the patient's data.

        Returns:
            str: A string describing the context information.
        """
        context = []
        patient_data = self.graph.nodes[patient_id]
        
        for keyword in keywords:
            if keyword.lower() in patient_data:
                context.append(f"{keyword.capitalize()}: {patient_data[keyword.lower()]}")
        
        if not context:
            # Include all patient data if no specific keyword matches
            for key, value in patient_data.items():
                context.append(f"{key.capitalize()}: {value}")
        
        return "\n".join(context)

class TextSummarizationAgent:
    """
    Agent responsible for summarizing medical text.
    """
    def __init__(self):
        """
        Initialize the TextSummarizationAgent and configure the generative AI model.
        """
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-pro')

    def summarize_text(self, text):
        """
        Summarize the given medical text.

        Args:
            text (str): The medical text to be summarized.

        Returns:
            str: A string containing the summarized text.
        """
        prompt = f"Summarize the following medical text: '{text}'"
        response = self.model.generate_content(prompt)
        return response.text

# --- Modify the QueryHandlingAgent class ---
class QueryHandlingAgent:
    """
    Agent responsible for handling user queries and coordinating responses from other agents.
    """
    def __init__(self, history_agent, risk_agent, recommendation_agent, knowledge_agent, summarization_agent):
        """
        Initialize the QueryHandlingAgent.

        Args:
            history_agent (MedicalHistoryAgent): The agent for retrieving patient history.
            risk_agent (RiskAssessmentAgent): The agent for assessing health risks.
            recommendation_agent (RecommendationAgent): The agent for generating recommendations.
            knowledge_agent (KnowledgeRetrievalAgent): The agent for retrieving context from the knowledge graph.
            summarization_agent (TextSummarizationAgent): The agent for summarizing medical text.
        """
        self.history_agent = history_agent
        self.risk_agent = risk_agent
        self.recommendation_agent = recommendation_agent
        self.knowledge_agent = knowledge_agent
        self.summarization_agent = summarization_agent
        self.conversation_history = []
        self.patient_memory = {}  # Store patient-specific context and insights

    def handle_query(self, patient_id, query):
        """
        Handle a user query and generate responses using various agents.

        Args:
            patient_id (int): The ID of the patient.
            query (str): The user query.

        Returns:
            tuple: A tuple containing the AI response, summary, risk assessment, recommendations, and relevance score.
        """
        # Convert patient_id to integer
        patient_id = int(patient_id)
        
        # Check if patient_id exists in the knowledge graph
        if patient_id not in self.knowledge_agent.graph.nodes:
            message = "Patient ID not found in the knowledge graph."
            return message, message, message, message, 0.0

        # 1. Retrieve Patient History
        history = self.history_agent.get_patient_history(patient_id)
        self.conversation_history.append(("System", history))

        # 2. Extract Keywords from Query (Simple example)
        keywords = query.split()

        # 3. Retrieve Context from Knowledge Graph
        context = self.knowledge_agent.retrieve_context(patient_id, keywords)
        self.patient_memory[patient_id] = {"context": context}  # Store in patient memory

        # 4. Generate Few-Shot Examples
        few_shot_examples = """
        Example 1:
        Patient ID: 83752
        Context: Gender: Male, Age: 65, Diagnosis: Cardiac arrest
        Medical History: Patient 83752 has the following diagnoses: 4275 (Cardiac arrest)
        User Query: What are the patient's risks?
        Response: The patient is at risk of heart failure due to the cardiac arrest.

        Example 2:
        Patient ID: 67890
        Context: Gender: Female, Age: 70, Diagnosis: Acute kidney failure
        Medical History: Patient 67890 has the following diagnoses: 5849 (Acute kidney failure)
        User Query: What are the patient's risks?
        Response: The patient is at risk of electrolyte imbalance and infection due to acute kidney failure.
        """

        # 4.1 Generate Prompt
        system_message = "You are a helpful AI assistant. Provide clear and concise responses to the user query."
        prompt = f"{system_message}\n\n{few_shot_examples}\n\nPatient ID: {patient_id}\nContext:\n{context}\n\nMedical History:\n{history}\n\nUser Query: {query}\nResponse:"

        # 4. Generate Prompt
        #system_message = "You are a helpful AI assistant. Provide clear and concise responses to the user query."
        #prompt = f"{system_message}\n\nPatient ID: {patient_id}\nContext:\n{context}\n\nMedical History:\n{history}\n\nUser Query: {query}\nResponse:"

        # 5. Interact with Google Generative AI 
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)  # Get response from LLM

        # Extract the plain text response
        response_text = response.text if response else "The AI model did not return a valid response."

        # Content moderation check
        if "HARM_CATEGORY" in response_text:
            response_text = "The response contains potentially harmful content and has been filtered."

        self.conversation_history.append(("User", query))
        self.conversation_history.append(("AI", response_text))

        # 6. Risk Assessment and Recommendations (using other agents)
        risk_assessment = self.risk_agent.assess_risk(history + "\n" + str(response_text))
        recommendations = self.recommendation_agent.generate_recommendations(risk_assessment)

        # Store insights in patient memory
        self.patient_memory[patient_id]["risks"] = risk_assessment
        self.patient_memory[patient_id]["recommendations"] = recommendations

        # 7. Summarize Text from Cardiac Data
        summary = self.summarization_agent.summarize_text(context)

        state_graph = gr.Image("./agent_state_graph.png", width=300, height=300)

        # Evaluation Metrics
        def evaluate_metrics(reference, candidate):
            """
            Evaluate the Precision, Recall, F1-Score, and AUC-ROC of the candidate response
            against the reference response.

            Args:
                reference (str): The reference response.
                candidate (str): The candidate response.

            Returns:
                dict: The evaluation metrics.
            """
            reference_tokens = reference.split()
            candidate_tokens = candidate.split()

            # Convert to binary format for evaluation
            y_true = [1 if token in reference_tokens else 0 for token in reference_tokens]
            y_pred = [1 if token in candidate_tokens else 0 for token in reference_tokens]

            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_true, y_pred)
            except ValueError:
                auc = float("nan")  # or any other default value or handling

            # Ensure y_true and y_pred have both positive and negative samples
            if len(set(y_true)) > 1 and len(set(y_pred)) > 1:
                auc = roc_auc_score(y_true, y_pred)
            else:
                auc = float("nan")

            return {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "auc": auc
            }

        # Example usage
        reference = "The patient has cardiac arrest and acute kidney failure."
        candidate = "The patient is suffering from cardiac arrest and acute kidney failure."

        metrics = evaluate_metrics(reference, candidate)

        return response_text, summary, risk_assessment, recommendations, state_graph, metrics

# --- Execution ---
if __name__ == "__main__":
    # 1. Create Knowledge Graph from MIMIC-III
    knowledge_graph = create_knowledge_graph()  # Assign the returned graph

    # 2. Initialize Agents
    history_agent = MedicalHistoryAgent(knowledge_graph)
    risk_agent = RiskAssessmentAgent()
    recommendation_agent = RecommendationAgent()
    knowledge_agent = KnowledgeRetrievalAgent(knowledge_graph)
    summarization_agent = TextSummarizationAgent()
    query_agent = QueryHandlingAgent(history_agent, risk_agent, recommendation_agent, knowledge_agent, summarization_agent)

    # --- Modify the Gradio interface function ---
    def gradio_interface(patient_id, query):
        """
        Gradio interface function to handle user input and generate responses.

        Args:
            patient_id (str): The ID of the patient.
            query (str): The user query.

        Returns:
            tuple: A tuple containing the AI response, summary, risk assessment, recommendations, and relevance score.
        """
        response, summary, risks, recommendations, state_graph, metrics = query_agent.handle_query(patient_id, query)
        return response, metrics["precision"], metrics["recall"], metrics["f1_score"], metrics["auc"], summary, risks, recommendations, state_graph

    # --- Add Human Feedback Handling ---
    def handle_feedback(patient_id, query, feedback):
        """
        Handle human feedback for the AI response.

        Args:
            patient_id (str): The ID of the patient.
            query (str): The user query.
            feedback (str): The human feedback.

        Returns:
            str: Acknowledgment message.
        """
        # Store feedback in patient memory
        if patient_id not in query_agent.patient_memory:
            query_agent.patient_memory[patient_id] = {}
        if "feedback" not in query_agent.patient_memory[patient_id]:
            query_agent.patient_memory[patient_id]["feedback"] = []
        query_agent.patient_memory[patient_id]["feedback"].append((query, feedback))
        
        return "Thank you for your feedback!"

    # --- Modify the Gradio UI to include feedback input ---
    with gr.Blocks() as gr_ui:
        gr.Markdown("<h1 style='text-align: center;'>AICare: AI Powered Health Assistance 🩺 🧑‍⚕️</h1>")
        gr.Markdown("---")
        gr.Markdown("Enter Patient ID and your query to get AI-generated responses, risk assessments, recommendations, and text summaries.")
        with gr.Row():
            with gr.Column():
                patient_id_input = gr.Textbox(label="Patient ID")
                query_input = gr.Textbox(label="Query")
                with gr.Row():
                    clear_button = gr.Button("Clear")
                    submit_button = gr.Button("Submit", variant="primary")
            with gr.Column():
                ai_response_output = gr.Textbox(label="AI Response")
            with gr.Row():
                precision_output = gr.Textbox(label="Precision")
                recall_output = gr.Textbox(label="Recall")
                f1_score_output = gr.Textbox(label="F1-Score")
                auc_output = gr.Textbox(label="AUC-ROC")
            with gr.Row():
                thumbs_up_button = gr.Button("👍")
                thumbs_down_button = gr.Button("👎")
        with gr.Row():
            summary_output = gr.Textbox(label="Discharge Notes(Summary)")
            risk_assessment_output = gr.Textbox(label="Risk Assessment")
            recommendations_output = gr.Textbox(label="Recommendations")
            state_graph_output = gr.Image(label="State Graph", width=300, height=300)

        submit_button.click(
            fn=gradio_interface,
            inputs=[patient_id_input, query_input],
            outputs=[ai_response_output, precision_output, recall_output, f1_score_output, auc_output, summary_output, risk_assessment_output, recommendations_output, state_graph_output]
        )
        
        clear_button.click(
            fn=lambda: ("", 0.0, 0.0, 0.0, 0.0, "", "", "", None),
            inputs=[],
            outputs=[ai_response_output, precision_output, recall_output, f1_score_output, auc_output, summary_output, risk_assessment_output, recommendations_output, state_graph_output]
        )

        thumbs_up_button.click(
            fn=lambda patient_id, query: handle_feedback(patient_id, query, "positive"),
            inputs=[patient_id_input, query_input],
            outputs=[ai_response_output]
        )

        thumbs_down_button.click(
            fn=lambda patient_id, query: handle_feedback(patient_id, query, "negative"),
            inputs=[patient_id_input, query_input],
            outputs=[ai_response_output]
        )

    gr_ui.launch()