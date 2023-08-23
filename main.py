# 2UCYD5FB0zA20cZ4jmgoITSvXnT_3n24xQHA7GmpXkjpxMEFS
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, T5ForConditionalGeneration
import time
import json

with open("config.json", "r", encoding="utf8") as config_file:
    config = json.load(config_file)


tokenizer = AutoTokenizer.from_pretrained(config["model_l"])
# model = T5ForConditionalGeneration.from_pretrained(config["model_name"], device_map = "auto") if config["gpu"] else T5ForConditionalGeneration.from_pretrained(config["model_name"])

app = FastAPI()

# CORS middleware configuration
origins = config["cors_origins"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model during startup
@app.on_event("startup")
async def startup_event():
    app.state.model = T5ForConditionalGeneration.from_pretrained(config["model_name"], device_map = "auto") if config["gpu"] else T5ForConditionalGeneration.from_pretrained(config["model_name"])

# Unload the model during shutdown
@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, "grammar_llm"):
        del app.state.model

# Model for JSON input
class InputData(BaseModel):
    text: str = config["sample_text"]
    prompt: str = config["prompt"]


async def get_predictions(prompt, text):
    text = f"{prompt} {text}"
    input_ids = tokenizer(text, return_tensors="pt").to("cuda").input_ids if config["gpu"] else tokenizer(text, return_tensors="pt").input_ids
    length = input_ids.shape[1] + 10
    if length >= 512:
        length = 512
    outputs = app.state.model.generate(input_ids, max_length=length)
    edited_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"prediction": edited_text, "word_count": length}


@app.post("/predict")
async def process_input(input_data: InputData):
    start = time.time()
    result = await get_predictions(input_data.prompt, input_data.text)
    end = time.time()
    return {"input": input_data.text, "result": result, "total_time": end-start}

