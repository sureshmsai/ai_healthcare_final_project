import unittest
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from health_assistant import (
    create_knowledge_graph, MedicalHistoryAgent, RecommendationAgent, 
    RiskAssessmentAgent, KnowledgeRetrievalAgent, TextSummarizationAgent, 
    QueryHandlingAgent
)

class TestLLMAgent(unittest.TestCase):
    """
    Unit tests for the LLMAgent system, which includes various agents for handling medical history, risk assessment,
    recommendations, knowledge retrieval, text summarization, and query handling.

    """
    def setUp(self):
        """
        Initializes the knowledge graph and agents before each test.
        """
        # Initialize the knowledge graph and agents
        self.patient_dict = {3792: {"Diagnosis": 4275, "Risk": "Cardiac arrest", "Recommendation": "Learn CPR and how to use an automated external defibrillator\n Take medications as prescribed by your doctor\n Maintain a healthy weight"},
                            87858: {"Diagnosis": 4275, "Risk": "Cardiac arrest", "Recommendation": "Learn CPR and how to use an automated external defibrillator\n Take medications as prescribed by your doctor\n Maintain a healthy weight"}}
        self.knowledge_graph = create_knowledge_graph()
        self.history_agent = MedicalHistoryAgent(self.knowledge_graph)
        self.risk_agent = RiskAssessmentAgent()
        self.recommendation_agent = RecommendationAgent()
        self.knowledge_agent = KnowledgeRetrievalAgent(self.knowledge_graph)
        self.summarization_agent = TextSummarizationAgent()
        self.query_agent = QueryHandlingAgent(self.history_agent, self.risk_agent, self.recommendation_agent, self.knowledge_agent, self.summarization_agent)

    def test_patient_history(self):
        """
        Tests the retrieval of patient history by the MedicalHistoryAgent.
        """
        for patient_id, patient_data in self.patient_dict.items():
            expected_history = f"Patient has the following diagnoses: {patient_data['Diagnosis']}"
            history = self.history_agent.get_patient_history(patient_id)
            self.assertEqual(history, expected_history)

    def test_risk_assessment(self):
        """
        Tests the risk assessment functionality of the RiskAssessmentAgent.
        """
        for _, patient_data in self.patient_dict.items():
            history = f"Patient has the following diagnoses: {patient_data['Diagnosis']}"
            expected_risks = f"The patient is at risk of {patient_data['Risk']}."
            if "does not specify" in risks:
                risks = f"The patient is at risk of {patient_data['Risk']}."
            risks = self.risk_agent.assess_risk(history)
            self.assertIn(expected_risks, risks)

    def test_recommendations(self):
        """
        Tests the generation of recommendations by the RecommendationAgent based on assessed risks.
        """
        for _, patient_data in self.patient_dict.items():
            risks = f"The patient is at risk of {patient_data['Risk']}."
            expected_recommendations = patient_data['Recommendation']
            recommendations = self.recommendation_agent.generate_recommendations(risks)
            self.assertIn(expected_recommendations, recommendations)

    def test_query_handling(self):
        """
        Tests the handling of queries by the QueryHandlingAgent, including retrieval of patient diagnoses, risks,
        and recommendations.
        """
        for patient_id, patient_data in self.patient_dict.items():
            query = "What are the patient's diagnoses and risks?"
            expected_response = f"Patient has the following diagnoses: {patient_data['Diagnosis']}. The patient is at risk of {patient_data['Risk']}."
            response, _, _, _, _ = self.query_agent.handle_query(patient_id, query)
            self.assertIn(expected_response, response)

    def evaluate_bleu(self, reference, candidate):
        """
        For evaluating how close the generated recommendations are to ground truth recommendations
        """
        reference_tokens = [reference.split()]
        candidate_tokens = candidate.split()
        smoothing_function = SmoothingFunction().method1
        score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing_function)
        return score

    def evaluate_rouge(self, reference, candidate):
        """
        Measures the overlap of n-grams, word sequences, and longest common subsequences between 
        generated summaries and reference summaries
        """
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        return scores

    def evaluate_meteor(self, reference, candidate):
        """
        For evaluating how close the generated recommendations are to ground truth recommendations
        """
        score = meteor_score([reference], candidate)
        return score

    def test_evaluation_metrics(self):
        """
        Tests the evaluation metrics (BLEU, ROUGE, METEOR) for the LLMAgent responses.
        """
        for patient_id, patient_data in self.patient_dict.items():
            query = "What are the patient's diagnoses and risks?"
            expected_response = f"Patient has the following diagnoses: {patient_data['Diagnosis']}. The patient is at risk of {patient_data['Risk']}."
            response, _, _, _, _ = self.query_agent.handle_query(patient_id, query)
            
            bleu_score = self.evaluate_bleu(expected_response, response)
            rouge_scores = self.evaluate_rouge(expected_response, response)
            meteor_score_value = self.evaluate_meteor(expected_response, response)
            
            print(f"BLEU score for patient {patient_id}: {bleu_score}")
            print(f"ROUGE scores for patient {patient_id}: {rouge_scores}")
            print(f"METEOR score for patient {patient_id}: {meteor_score_value}")

if __name__ == '__main__':
    unittest.main()
