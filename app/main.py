from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.ml.inference import predict_spam

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": None}
    )

@app.post("/", response_class=HTMLResponse)
def predict(request: Request, message: str = Form(...)):
    if not message.strip():
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": "Please enter a message",
                "message": message
            }
        )

    prediction = predict_spam(message)
    result = "SPAM 🚨" if prediction == 1 else "NOT SPAM ✅"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "message": message
        }
    )
