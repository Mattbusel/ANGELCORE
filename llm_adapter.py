import os
import openai

class LLMAdapter:
    def __init__(self, model="gpt-4", api_key=None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key

    def query(self, prompt, temperature=0.7, max_tokens=512):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response["choices"][0]["message"]["content"].strip()

# --- Sample usage in Raven and Seraph classes ---

class RavenIntelligence:
    def __init__(self, llm):
        self.llm = llm

    def interpret_pattern(self, pattern):
        prompt = f"Interpret this neural lattice pattern in symbolic language:\n{pattern}"
        return self.llm.query(prompt)

class SeraphIntelligence:
    def __init__(self, llm):
        self.llm = llm

    def evaluate_ethics(self, action):
        prompt = f"Evaluate the ethical implications of this action in a sensitive biological AI system:\n{action}"
        return self.llm.query(prompt)

# --- Demo Run ---
if __name__ == "__main__":
    llm = LLMAdapter()
    raven = RavenIntelligence(llm)
    seraph = SeraphIntelligence(llm)

    pattern = "110010011001 - Synaptic burst encoding - Phase alignment: Positive"
    interpretation = raven.interpret_pattern(pattern)
    print("[RAVEN INTERPRETATION]\n", interpretation)

    action = f"Trigger memory recall based on: {interpretation}"
    ethics = seraph.evaluate_ethics(action)
    print("\n[SERAPH ETHICAL EVALUATION]\n", ethics)
