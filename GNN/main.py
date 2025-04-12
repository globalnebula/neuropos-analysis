import requests


def gnnLLM(drug, history):
    system_prompt = "You are a medical AI expert understanding medical history of the patient and the current drug intake an dprovode the possible side effects with probablities and give the remedy too. You also act like a GNN providing the Drug Interaction path with various protiens in the body"
    user_prompt = f"Here is the drug and patient history:\n{drug, history}\n\nProvide a diagnosis."

    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3.1",
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False
        })
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"❌ LLaMA backend error. Status code: {response.status_code}"
    except Exception as e:
        return f"❌ Failed to connect to LLaMA: {str(e)}"