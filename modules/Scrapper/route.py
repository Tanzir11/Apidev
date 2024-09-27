from .service import Scrapper_obj
from .dto import Scrapper
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

EnbeddingGenerator = APIRouter(tags=["Processes"])

@EnbeddingGenerator.post("/process_url")
def generate_resignation_letter(body: Scrapper):
    return Scrapper_obj.embedding_generator_url(body.url)

@EnbeddingGenerator.post("/process_pdf")
async def upload_resume(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Please upload a valid PDF file.")
    pdf_bytes = await file.read()
    return Scrapper_obj.embedding_generator_pdf(pdf_bytes, file.filename)