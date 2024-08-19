from fastapi import FastAPI, Request, HTTPException
from transformers import pipeline
import os
import json

app = FastAPI()

# Path to the file where prompts will be stored
# PROMPT_FILE_PATH = "prompts.json"
PROMPT_FILE_PATH = "/app/data/prompts.json"


# Ensure the prompts file exists
if not os.path.exists(PROMPT_FILE_PATH):
    with open(PROMPT_FILE_PATH, 'w') as f:
        json.dump([], f)

# Load the transformer model pipeline
generator = pipeline("text-generation", model="openai-community/gpt2-medium", trust_remote_code=True)
# Alternatively, you can switch to the other model if needed:
# generator = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)

@app.post("/generate")
async def generate_text(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt")
        if not prompt:
            raise HTTPException(status_code=400, detail="No prompt provided")
        
        # Save the prompt to the file
        try:
            with open(PROMPT_FILE_PATH, 'r+') as f:
                prompts = json.load(f)
                prompts.append(prompt)
                f.seek(0)
                json.dump(prompts, f)
            print(f"Saved prompt: {prompt}")
        except Exception as e:
            print(f"Error saving prompt: {e}")
            raise HTTPException(status_code=500, detail="Error saving prompt")

        # Generate text using the model
        result = generator(prompt, max_length=50, num_return_sequences=1)
        return {"generated_text": result[0]['generated_text']}
    
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/prompts")
async def get_prompts():
    # Retrieve the stored prompts
    try:
        with open(PROMPT_FILE_PATH, 'r') as f:
            prompts = json.load(f)
        return {"prompts": prompts}
    except Exception as e:
        print(f"Error retrieving prompts: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
