import json
import tempfile
import google.generativeai as genai
from pydantic import BaseModel, Field
from settings_service import SettingsService
from pdf2image import convert_from_bytes
import logging
import pdfplumber
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pickle
import hashlib
from typing import List, Any, Optional
import base64
from PIL import Image
import io
from page_processor import PageProcessor
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetectedLayoutItem(BaseModel):
    element_type: str = Field(
        ...,
        description="Type of detected item (Table, Figure, Image, or Text-block). Return at most 10 items.",
    )
    summary: str = Field(..., description="A summary of the layout item.")

class LayoutElements(BaseModel):
    layout_items: list[DetectedLayoutItem] = Field(default_factory=list)

class DocumentLayoutParsingState(BaseModel):
    document_path: str
    pages_as_base64_jpeg_images: list[str] = Field(default_factory=list)
    pages_processed_as_text: list[int] = Field(default_factory=list)
    extracted_layouts: list = Field(default_factory=list)
    page_analysis: dict = Field(default_factory=dict)

class FindLayoutItemsInput(BaseModel):
    document_path: str
    base64_jpeg: Optional[str] = None
    page_number: int
    page_text: Optional[str] = None
    is_visual_page: bool = True

def extract_images_from_pdf(pdf_path: str, pages: List[int] = None, dpi: int = 150):
    logger.info(f"Extracting images from PDF: {pdf_path}, pages: {pages}")
    with open(pdf_path, "rb") as f:
        with tempfile.TemporaryDirectory() as temp_dir:
            if pages:
                page_numbers = [p+1 for p in pages]  # pdf2image uses 1-indexed pages
                images = convert_from_bytes(f.read(), output_folder=temp_dir, fmt="jpeg", dpi=dpi, 
                                           first_page=min(page_numbers), last_page=max(page_numbers))
                # Map the images back to their original page numbers
                page_to_image = {}
                for i, page_num in enumerate(range(min(page_numbers), max(page_numbers) + 1)):
                    if page_num - 1 in pages:  # Convert back to 0-indexed
                        page_to_image[page_num - 1] = images[i]
                
                # Sort images by original page number
                result = [page_to_image[p] for p in sorted(page_to_image.keys())]
                logger.info(f"Extracted {len(result)} images for selected pages")
                return result
            else:
                images = convert_from_bytes(f.read(), output_folder=temp_dir, fmt="jpeg", dpi=dpi)
                logger.info(f"Extracted {len(images)} images from all PDF pages")
                return images

def pil_image_to_base64_jpeg(image) -> str:
    import io, base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def get_document_hash(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

class DocumentParser:
    def __init__(self, model_name="gemini-2.0-flash", cache_dir="doc_cache"):
        self.model_name = model_name
        self.model = genai.GenerativeModel(
            self.model_name,
            generation_config={
                "response_mime_type": "application/json",
            },
        )
        self.page_processor = PageProcessor()
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def process_document(self, document_path: str, use_cache: bool = True) -> DocumentLayoutParsingState:
        if use_cache:
            cache_result = self._check_cache(document_path)
            if cache_result:
                logger.info(f"Using cached results for {document_path}")
                return cache_result
        
        # Create initial state
        state = DocumentLayoutParsingState(document_path=document_path)
        
        # Identify pages requiring visual processing
        complex_pages = self.page_processor.identify_complex_pages(document_path)
        logger.info(f"Complex pages requiring visual processing: {complex_pages}")
        
        # Extract images only for complex pages
        if complex_pages:
            images = extract_images_from_pdf(document_path, pages=complex_pages)
            base64_images = [pil_image_to_base64_jpeg(img) for img in images]
            
            # Map images back to their original page numbers
            page_to_base64 = {page_num: base64_images[i] for i, page_num in enumerate(complex_pages)}
            state.pages_as_base64_jpeg_images = base64_images
        else:
            page_to_base64 = {}
        
        # Get total page count
        with pdfplumber.open(document_path) as pdf:
            total_pages = len(pdf.pages)
        
        # Create processing tasks for each page
        tasks = []
        for page_num in range(total_pages):
            # Determine processing method for this page
            is_visual_page = page_num in complex_pages
            
            if is_visual_page:
                # Process as image
                page_input = FindLayoutItemsInput(
                    document_path=document_path,
                    base64_jpeg=page_to_base64[page_num],
                    page_number=page_num,
                    is_visual_page=True
                )
            else:
                # Process as text only
                with pdfplumber.open(document_path) as pdf:
                    page_text = pdf.pages[page_num].extract_text() or ""
                
                page_input = FindLayoutItemsInput(
                    document_path=document_path,
                    page_number=page_num,
                    page_text=page_text,
                    is_visual_page=False
                )
                state.pages_processed_as_text.append(page_num)
            
            tasks.append(page_input)
        
        # Process pages in parallel
        state.extracted_layouts = [None] * total_pages  # Pre-allocate to maintain order
        
        with ProcessPoolExecutor() as executor:
            # Submit all tasks
            future_to_page = {executor.submit(self._process_page_wrapper, task): task.page_number 
                             for task in tasks}
            
            # Process results as they complete
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    layout_data = future.result()
                    state.extracted_layouts[page_num] = layout_data
                    logger.info(f"Completed processing page {page_num + 1}")
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {str(e)}")
                    # Add placeholder for failed page
                    state.extracted_layouts[page_num] = {"error": str(e)}
                    logger.error(traceback.format_exc())
        
        # Cache the results
        if use_cache:
            self._save_to_cache(document_path, state)
        
        return state
    
    def _process_page_wrapper(self, page_input: FindLayoutItemsInput) -> dict:
        if page_input.is_visual_page:
            return self.process_visual_page(page_input)
        else:
            return self.process_text_page(page_input)
    
    def process_visual_page(self, page_input: FindLayoutItemsInput) -> dict:
        logger.info(f"Processing visual page {page_input.page_number + 1}")
        prompt = (
            f"Extract and label all the relevant layout elements on this PDF page. For each element, "
            "provide the following information:\n"
            "- element_type: one of Table, Figure, Image, or Text-block\n"
            "- summary: a detailed description of the element\n\n"
            "Return the output as a JSON object that follows this schema: "
            f"{LayoutElements.model_json_schema()}\n\n"
            "Be especially thorough with tables and figures."
        )
        messages = [
            {
                "parts": [
                    {"text": prompt},
                    {"mime_type": "image/jpeg", "data": page_input.base64_jpeg}
                ]
            }
        ]
        result = self.model.generate_content(messages)
        data = json.loads(result.text)
        return data
    
    def process_text_page(self, page_input: FindLayoutItemsInput) -> dict:
        logger.info(f"Processing text-only page {page_input.page_number + 1}")
        
        # For text-only pages, we can use a simpler approach
        text_blocks = self._extract_text_blocks(page_input.page_text)
        
        # Format as LayoutElements
        layout_items = [
            {"element_type": "Text-block", "summary": block}
            for block in text_blocks if block.strip()
        ]
        
        return {"layout_items": layout_items}
    
    def _extract_text_blocks(self, text: str) -> List[str]:
        """Split text into logical blocks/paragraphs"""
        if not text:
            return []
            
        # Split by double newlines to get paragraphs
        paragraphs = text.split('\n\n')
        
        # Further process paragraphs with single newlines that might be lists
        result = []
        for para in paragraphs:
            if '\n' in para and len(para) < 1000:
                result.append(para)
            else:
                result.append(para)
                
        return [block for block in result if block.strip()]
    
    def _check_cache(self, document_path: str) -> Optional[DocumentLayoutParsingState]:
        doc_hash = get_document_hash(document_path)
        cache_path = os.path.join(self.cache_dir, f"{doc_hash}.pickle")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache: {str(e)}")
                return None
        return None
    
    def _save_to_cache(self, document_path: str, state: DocumentLayoutParsingState):
        doc_hash = get_document_hash(document_path)
        cache_path = os.path.join(self.cache_dir, f"{doc_hash}.pickle")
        
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(state, f)
            logger.info(f"Saved results to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Error saving to cache: {str(e)}")

def save_base64_image(base64_str: str, output_path: str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    image.save(output_path, format="JPEG")
    print(f"Saved image to {output_path}")

if __name__ == "__main__":
    document_path = "testing/LoRA-GA.pdf"
    genai.configure(api_key=SettingsService().settings.google_api_key)
    
    parser = DocumentParser()
    state = parser.process_document(document_path)
    
    print(f"\n--- Document Processing Summary ---")
    print(f"Total pages: {len(state.extracted_layouts)}")
    print(f"Pages processed as images: {len(state.pages_as_base64_jpeg_images)}")
    print(f"Pages processed as text: {len(state.pages_processed_as_text)}")
    
    for i in range(len(state.extracted_layouts)):
        print(f"\n--- Sample from Page {i+1} ---")
        layout = state.extracted_layouts[i]
        if isinstance(layout, dict) and "layout_items" in layout:
            items = layout["layout_items"]
            print(f"Number of elements: {len(items)}")
            if items:
                print(f"Content: {json.dumps(items, indent=2)}")
    
    # verify_dir = "verify"
    # os.makedirs(verify_dir, exist_ok=True)
    
    # for page_num, base64_img in enumerate(state.pages_as_base64_jpeg_images):
    #     image_data = base64.b64decode(base64_img)
    #     from PIL import Image
    #     import io
    #     image = Image.open(io.BytesIO(image_data))
    #     output_path = os.path.join(verify_dir, f"page_{page_num+1}.jpg")
    #     image.save(output_path, format="JPEG")
    #     logger.info(f"Saved verification image for page {page_num+1} to {output_path}")