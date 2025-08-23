import pdfplumber
import numpy as np
import cv2
import re
from typing import Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class PageProcessor:
    def __init__(self, 
                 table_text_density_threshold: float = 0.1,
                 whitespace_pattern_threshold: float = 0.4,
                 min_image_area_ratio: float = 0.03,
                 text_density_threshold: float = 0.6):

        self.table_text_density_threshold = table_text_density_threshold
        self.whitespace_pattern_threshold = whitespace_pattern_threshold
        self.min_image_area_ratio = min_image_area_ratio
        self.text_density_threshold = text_density_threshold
    
    def should_process_as_image(self, pdf_path: str, page_num: int) -> Tuple[bool, Dict[str, Any]]:
        logger.info(f"Analyzing page {page_num + 1} of {pdf_path}")
        
        # Analysis results dictionary to store reasoning
        analysis = {
            "has_tables": False,
            "has_figures": False,
            "has_complex_layout": False,
            "text_density": 0.0,
            "reasoning": []
        }
        
        # Extract text and visual features
        with pdfplumber.open(pdf_path) as pdf:
            if page_num >= len(pdf.pages):
                logger.warning(f"Page {page_num + 1} out of range for document with {len(pdf.pages)} pages")
                return False, {"reasoning": ["Page number out of range"]}
            
            page = pdf.pages[page_num]
            page_text = page.extract_text() or ""
            page_width, page_height = float(page.width), float(page.height)
            page_area = page_width * page_height
            
            # Check for table indicators in text patterns
            if self._has_table_patterns(page_text):
                analysis["has_tables"] = True
                analysis["reasoning"].append("Detected table patterns in text")
            
            # Extract and analyze images
            images = page.images
            if images:
                total_image_area = sum(img["width"] * img["height"] for img in images)
                image_area_ratio = total_image_area / page_area
                
                if image_area_ratio > self.min_image_area_ratio:
                    analysis["has_figures"] = True
                    analysis["reasoning"].append(f"Found images covering {image_area_ratio:.1%} of page")
            
            # Check text density
            if page_text:
                words = page_text.split()
                chars = len(page_text)
                text_density = chars / page_area
                analysis["text_density"] = text_density
                
                # Low text density but content exists suggests figures/charts
                if text_density < self.text_density_threshold and chars > 50:
                    analysis["has_complex_layout"] = True
                    analysis["reasoning"].append(f"Low text density ({text_density:.5f}) with content suggests figures/charts")
            
            # Check for figure/table mentions in text
            figure_pattern = re.compile(r'(figure|fig\.?|chart|graph|plot|diagram)', re.IGNORECASE)
            table_pattern = re.compile(r'(table|tbl\.?|tabular)', re.IGNORECASE)
            
            if figure_pattern.search(page_text):
                analysis["has_figures"] = True
                analysis["reasoning"].append("Text mentions figures/charts")
                
            if table_pattern.search(page_text):
                analysis["has_tables"] = True
                analysis["reasoning"].append("Text mentions tables")
            
            # Check for whitespace patterns that suggest tables
            try:
                page_image = page.to_image()
                img = page_image.original
                gray = np.array(img.convert('L'))
                
                # Check for grid-like structures using horizontal/vertical line detection
                if self._detect_grid_structures(gray):
                    analysis["has_tables"] = True
                    analysis["reasoning"].append("Detected grid-like structures")
            except Exception as e:
                logger.warning(f"Error analyzing page image: {str(e)}")
        
        # Decision making based on the analysis
        needs_visual_processing = (
            analysis["has_tables"] or 
            analysis["has_figures"] or 
            analysis["has_complex_layout"]
        )
        
        if needs_visual_processing:
            logger.info(f"Page {page_num + 1} needs visual processing: {', '.join(analysis['reasoning'])}")
        else:
            logger.info(f"Page {page_num + 1} can be processed as text only")
        
        return needs_visual_processing, analysis
    
    def _has_table_patterns(self, text: str) -> bool:
        if not text:
            return False
        
        # Check for repeating whitespace patterns (common in tables)
        lines = text.split('\n')
        if len(lines) < 3:
            return False
        
        # Count lines with similar whitespace patterns
        whitespace_patterns = []
        for line in lines:
            # Create a binary pattern where spaces are 1 and text is 0
            pattern = ''.join(['1' if c.isspace() else '0' for c in line])
            whitespace_patterns.append(pattern)
        
        # Count lines with similar patterns
        pattern_counts = {}
        for pattern in whitespace_patterns:
            if pattern in pattern_counts:
                pattern_counts[pattern] += 1
            else:
                pattern_counts[pattern] = 1
        
        # If a significant portion of lines share patterns, likely a table
        if len(lines) > 0:
            max_pattern_ratio = max(pattern_counts.values()) / len(lines)
            return max_pattern_ratio > self.whitespace_pattern_threshold
        
        return False
    
    def _detect_grid_structures(self, gray_img: np.ndarray) -> bool:
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # Count detected lines
        h_lines = cv2.HoughLinesP(horizontal_lines, 1, np.pi/180, 
                                 threshold=50, minLineLength=50, maxLineGap=10)
        v_lines = cv2.HoughLinesP(vertical_lines, 1, np.pi/180, 
                                 threshold=50, minLineLength=50, maxLineGap=10)
        
        # Check if we have enough lines to form a grid
        h_count = 0 if h_lines is None else len(h_lines)
        v_count = 0 if v_lines is None else len(v_lines)
        
        return h_count >= 3 and v_count >= 3
    
    def identify_complex_pages(self, pdf_path: str) -> List[int]:
        complex_pages = []
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"Analyzing {total_pages} pages in {pdf_path}")
            
            for page_num in range(total_pages):
                needs_visual, _ = self.should_process_as_image(pdf_path, page_num)
                if needs_visual:
                    complex_pages.append(page_num)
        
        logger.info(f"Identified {len(complex_pages)} complex pages requiring visual processing")
        return complex_pages