from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictionRequest(BaseModel):
    externalStatus: str

class PredictionResponse(BaseModel):
    internalStatus: str

@app.post("/predict", response_model=PredictionResponse)
async def predict_status(request: PredictionRequest):
    text = request.externalStatus
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=maxlen)
    prediction = model.predict(padded_sequence)
    predicted_class = np.argmax(prediction)
    predicted_status = label_encoder.inverse_transform([predicted_class])[0]
    return {"internalStatus": "Predicted status"}