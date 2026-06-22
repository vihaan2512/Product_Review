import io
import base64
import time
import requests
from pathlib import Path
from typing import Optional

import torch
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from PIL import Image
from loguru import logger

from src.utils.logger import setup_logger
from src.fusion.pipeline import AnalysisPipeline
from src.utils.scraper import extract_asin_from_url, scrape_amazon_reviews, scrape_ecommerce_reviews
import numpy as np
import threading
from database import init_db, save_report, get_history

# ── App setup ─────────────────────────────────────────────────
setup_logger()

# ── In-Memory ASIN Cache ──────────────────────────────────────
ASIN_CACHE = {}
CACHE_TTL = 24 * 60 * 60 

# ── Lifespan Context Manager ──────────────────────────────────
global pipeline
pipeline = None

def warm_up_models():
    global pipeline
    if pipeline is not None:
        try:
            logger.info("Background model parallel pre-warm starting...")
            import concurrent.futures
            
            def load_model(name, load_fn):
                t = time.time()
                logger.info(f"Pre-warming model: {name}...")
                load_fn()
                logger.info(f"Model {name} pre-warmed in {time.time() - t:.2f}s")
                
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(load_model, "Sentiment", lambda: pipeline.sentiment): "Sentiment",
                    executor.submit(load_model, "Defect", lambda: pipeline.defect): "Defect",
                    executor.submit(load_model, "Fake", lambda: pipeline.fake): "Fake",
                    executor.submit(load_model, "ABSA", lambda: pipeline.absa): "ABSA",
                }
                concurrent.futures.wait(futures)
                
            logger.success("All deep learning models pre-warmed successfully.")
        except Exception as e:
            logger.warning(f"Error during background model pre-warm: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Initializing SQLite database...")
    init_db()
    
    logger.info("Initializing AnalysisPipeline...")
    pipeline = AnalysisPipeline()

    threading.Thread(target=warm_up_models, daemon=True).start()
    
    logger.success("API ready.")
    yield

app = FastAPI(
    title="Multimodal E-commerce Product Review Analysis Platform",
    description=(
        "AI-powered product quality analysis combining sentiment analysis, "
        "defect detection, fake review detection, and aspect-based sentiment."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ─────────────────────────────────

class TextAnalysisRequest(BaseModel):
    reviews:         list[str]            = Field(..., min_length=1, max_length=500,
                                                  description="Customer review texts")
    ratings:         Optional[list[float]] = Field(None, description="Star ratings (1-5)")
    asin:            Optional[str]         = Field("UNKNOWN", description="Product ASIN")
    product_name:    Optional[str]         = Field("Product", description="Product name")
    run_absa:        Optional[bool]        = Field(True, description="Run ABSA (slower)")
    category:        Optional[str]         = Field("Auto (detect from reviews)",
                                                   description="Product category for ABSA aspect selection")
    custom_aspects:  Optional[list[str]]   = Field(None, description="Custom aspect terms (overrides category)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "reviews": [
                    "Battery life is amazing, lasts all day!",
                    "Sound quality is great but build feels cheap.",
                    "Terrible product, broke after one week.",
                ],
                "ratings": [5, 4, 1],
                "asin": "B08XYZ123",
                "product_name": "Wireless Earbuds Pro",
                "category": "Electronics & Gadgets",
            }
        }
    )

class URLAnalysisRequest(BaseModel):
    url:             str                   = Field(..., description="Product URL (Amazon or Flipkart)")
    run_absa:        Optional[bool]        = Field(True, description="Run ABSA (slower)")
    category:        Optional[str]         = Field("Auto (detect from reviews)",
                                                   description="Product category for ABSA aspect selection")
    custom_aspects:  Optional[list[str]]   = Field(None, description="Custom aspect terms (overrides category)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "url": "https://www.amazon.com/dp/B08Earbuds",
                "run_absa": True
            }
        }
    )

class QualityScoreResponse(BaseModel):
    asin:          str
    product_name:  str
    quality_score: float
    grade:         str
    breakdown:     dict
    flags:         list[str]
    summary:       str
    defect_source: str
    n_reviews:     int
    timing:        dict
    device:        str
    module_outputs: Optional[dict] = None


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "N/A"

    model_status = {
        "sentiment":   Path("models/sentiment_best.pt").exists(),
        "defect":      Path("models/defect_resnet_best.pt").exists(),
        "fake_review": Path("models/fake_review_detector.joblib").exists(),
        "absa":        True,  
        "pipeline":    pipeline is not None,
    }

    return {
        "status":       "healthy" if all(model_status.values()) else "degraded",
        "device":       device,
        "gpu_name":     gpu_name,
        "models":       model_status,
        "version":      "1.0.0",
    }


@app.post("/analyze/text", response_model=QualityScoreResponse)
async def analyze_text(request: TextAnalysisRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    modules = ["sentiment", "fake"]
    if request.run_absa:
        modules.append("absa")

    try:
        result = pipeline.analyze(
            reviews=request.reviews,
            ratings=request.ratings,
            image_path=None,
            asin=request.asin or "UNKNOWN",
            product_name=request.product_name or "Product",
            run_modules=modules,
            generate_heatmap=False,
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    def sanitize_data(val):
        if isinstance(val, Image.Image):
            return None
        elif isinstance(val, dict):
            return {k: sanitize_data(v) for k, v in val.items()}
        elif isinstance(val, (list, tuple)):
            return [sanitize_data(v) for v in val]
        elif hasattr(val, "model_dump"):
            return sanitize_data(val.model_dump())
        elif hasattr(val, "dict") and callable(getattr(val, "dict")):
            return sanitize_data(val.dict())
        elif isinstance(val, (np.float32, np.float64)):
            return float(val)
        elif isinstance(val, (np.int32, np.int64)):
            return int(val)
        elif isinstance(val, np.ndarray):
            return sanitize_data(val.tolist())
        elif hasattr(val, "__dict__"):
            try:
                return sanitize_data(vars(val))
            except Exception:
                return str(val)
        return val

    sanitized = sanitize_data(result)
    return QualityScoreResponse(
        asin=sanitized["asin"],
        product_name=sanitized["product_name"],
        quality_score=sanitized["quality_score"],
        grade=sanitized["grade"],
        breakdown=sanitized["breakdown"],
        flags=sanitized["flags"],
        summary=sanitized["summary"],
        defect_source=sanitized["defect_source"],
        n_reviews=sanitized["n_reviews"],
        timing=sanitized["timing"],
        device=sanitized["device"],
        module_outputs=sanitized.get("module_outputs"),
    )


@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image, got: {file.content_type}"
        )

    try:
        contents  = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        result = pipeline.defect.predict(
            pil_image,
            generate_heatmap=True,
        )

        def sanitize_data(val):
            if isinstance(val, Image.Image):
                return None
            elif isinstance(val, dict):
                return {k: sanitize_data(v) for k, v in val.items()}
            elif isinstance(val, (list, tuple)):
                return [sanitize_data(v) for v in val]
            elif hasattr(val, "model_dump"):
                return sanitize_data(val.model_dump())
            elif hasattr(val, "dict") and callable(getattr(val, "dict")):
                return sanitize_data(val.dict())
            elif isinstance(val, (np.float32, np.float64)):
                return float(val)
            elif isinstance(val, (np.int32, np.int64)):
                return int(val)
            elif isinstance(val, np.ndarray):
                return sanitize_data(val.tolist())
            elif hasattr(val, "__dict__"):
                try:
                    return sanitize_data(vars(val))
                except Exception:
                    return str(val)
            return val

        return sanitize_data({
            "label":                result["label"],
            "confidence":           result["confidence"],
            "uncertain":            result["uncertain"],
            "scores":               result["scores"],
            "defect_type":          result.get("defect_type"),
            "defect_type_confidence": result.get("defect_type_confidence"),
            "overlay_b64":          result.get("overlay_b64"),  
        })
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
async def analyze_full(
    reviews:      str = None,
    ratings:      str = None,
    asin:         str = "UNKNOWN",
    product_name: str = "Product",
    file: Optional[UploadFile] = File(None),
):
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    import json

    try:
        review_list = json.loads(reviews) if reviews else []
        rating_list = json.loads(ratings) if ratings else None
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid reviews/ratings JSON")

    if not review_list:
        raise HTTPException(status_code=400, detail="reviews cannot be empty")

    pil_image = None
    if file and file.content_type.startswith("image/"):
        contents  = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

    try:
        result = pipeline.analyze(
            reviews=review_list,
            ratings=rating_list,
            image_path=pil_image,
            asin=asin,
            product_name=product_name,
            run_modules=["sentiment", "defect", "fake", "absa"],
            generate_heatmap=pil_image is not None,
        )
    except Exception as e:
        logger.error(f"Full analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if "module_outputs" in result:
        defect_out = result["module_outputs"].get("defect", {})
        if defect_out and "overlay_pil" in defect_out:
            del defect_out["overlay_pil"]

    def sanitize_data(val):
        if isinstance(val, Image.Image):
            return None
        elif isinstance(val, dict):
            return {k: sanitize_data(v) for k, v in val.items()}
        elif isinstance(val, (list, tuple)):
            return [sanitize_data(v) for v in val]
        elif hasattr(val, "model_dump"):
            return sanitize_data(val.model_dump())
        elif hasattr(val, "dict") and callable(getattr(val, "dict")):
            return sanitize_data(val.dict())
        elif isinstance(val, (np.float32, np.float64)):
            return float(val)
        elif isinstance(val, (np.int32, np.int64)):
            return int(val)
        elif isinstance(val, np.ndarray):
            return sanitize_data(val.tolist())
        elif hasattr(val, "__dict__"):
            try:
                return sanitize_data(vars(val))
            except Exception:
                return str(val)
        return val

    return sanitize_data(result)


@app.post("/analyze/defect")
async def analyze_defect_only(file: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    if not file or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="A valid image file is required")

    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

    import tempfile as _tf, os as _os
    tmp = _tf.NamedTemporaryFile(delete=False, suffix=".jpg")
    pil_image.save(tmp.name)
    tmp.close()

    try:
        result = pipeline.analyze(
            reviews=["placeholder"],       
            ratings=None,
            image_path=tmp.name,
            asin="DEFECT_ONLY",
            product_name="Image Analysis",
            run_modules=["defect"],      
            generate_heatmap=True,
        )
    except Exception as e:
        logger.error(f"Defect-only analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if _os.path.exists(tmp.name):
            _os.remove(tmp.name)

    defect_out = result.get("module_outputs", {}).get("defect", {})
    defect_out.pop("overlay_pil", None)

    def sanitize_data(val):
        if isinstance(val, Image.Image):
            return None
        elif isinstance(val, dict):
            return {k: sanitize_data(v) for k, v in val.items()}
        elif isinstance(val, (list, tuple)):
            return [sanitize_data(v) for v in val]
        elif hasattr(val, "model_dump"):
            return sanitize_data(val.model_dump())
        elif hasattr(val, "dict") and callable(getattr(val, "dict")):
            return sanitize_data(val.dict())
        elif isinstance(val, (np.float32, np.float64)):
            return float(val)
        elif isinstance(val, (np.int32, np.int64)):
            return int(val)
        elif isinstance(val, np.ndarray):
            return sanitize_data(val.tolist())
        elif hasattr(val, "__dict__"):
            try:
                return sanitize_data(vars(val))
            except Exception:
                return str(val)
        return val

    return sanitize_data(defect_out)



@app.post("/analyze/url")
async def analyze_url_endpoint(request: URLAnalysisRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    clean_url = request.url.strip()
    asin_val = extract_asin_from_url(clean_url)
    if not asin_val:
        raise HTTPException(status_code=400, detail="Invalid Amazon URL: Could not extract ASIN.")

    cached = ASIN_CACHE.get(asin_val)
    if cached:
        if time.time() < cached["expiry"]:
            logger.info(f"Cache Hit for ASIN: {asin_val}... Returning instantly.")
            return cached["data"]
        else:
            logger.info("Cache entry expired. Re-scraping...")
            del ASIN_CACHE[asin_val]

    scraped = scrape_ecommerce_reviews(request.url, num_pages=2)
    
    asin = scraped["asin"]
    reviews = scraped["reviews"]
    ratings = scraped["ratings"]
    product_name = scraped["product_name"]
    image_url = scraped.get("image_url")
    review_image_urls = scraped.get("review_image_urls", [])

    if not reviews:
        raise HTTPException(
            status_code=400,
            detail=(
                "Amazon temporarily restricted review access.\n\n"
                "Possible reasons:\n"
                "• Login required\n"
                "• CAPTCHA\n"
                "• Region restriction\n"
                "• Product variant\n\n"
                "Try again later."
            )
        )

    run_modules = ["sentiment", "fake"]
    if request.run_absa:
        run_modules.append("absa")

    pipeline._absa_category      = request.category or "Auto (detect from reviews)"
    pipeline._absa_custom_aspects = request.custom_aspects or None

    # ── Run main analysis pipeline ─────────────────────────────
    try:
        result = pipeline.analyze(
            reviews=reviews,
            ratings=ratings,
            image_path=None,  
            asin=asin,
            product_name=product_name,
            run_modules=run_modules,
            generate_heatmap=False,
        )
    except Exception as e:
        logger.error(f"Scraper analysis pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if "module_outputs" in result:
        defect_out = result["module_outputs"].get("defect", {})
        if defect_out and "overlay_pil" in defect_out:
            del defect_out["overlay_pil"]

    def sanitize_data(val):
        if isinstance(val, Image.Image):
            return None
        elif isinstance(val, dict):
            return {k: sanitize_data(v) for k, v in val.items()}
        elif isinstance(val, (list, tuple)):
            return [sanitize_data(v) for v in val]
        elif hasattr(val, "model_dump"):
            return sanitize_data(val.model_dump())
        elif hasattr(val, "dict") and callable(getattr(val, "dict")):
            return sanitize_data(val.dict())
        elif isinstance(val, (np.float32, np.float64)):
            return float(val)
        elif isinstance(val, (np.int32, np.int64)):
            return int(val)
        elif isinstance(val, np.ndarray):
            return sanitize_data(val.tolist())
        elif hasattr(val, "__dict__"):
            try:
                return sanitize_data(vars(val))
            except Exception:
                return str(val)
        return val

    sanitized = sanitize_data(result)

    actual_rating = 80.0
    if ratings:
        actual_rating = float(np.mean(ratings)) * 20.0

    # ── Persist output report to database ──────────────────────
    save_report(
        asin=asin,
        name=product_name,
        url=clean_url,
        score=sanitized["quality_score"],
        grade=sanitized["grade"],
        n_reviews=sanitized["n_reviews"],
        breakdown=sanitized["breakdown"],
        flags=sanitized["flags"],
        summary=sanitized["summary"],
        full_report=sanitized,
        actual_rating=actual_rating
    )

    ASIN_CACHE[asin] = {
        "data": sanitized,
        "expiry": time.time() + CACHE_TTL
    }

    return sanitized


@app.get("/history")
async def history_endpoint(limit: int = 10):
    """Retrieves list of recently analyzed product quality reports."""
    return get_history(limit=limit)


@app.post("/analyze/compare")
async def analyze_compare_endpoint(
    reference: UploadFile = File(...),
    test:      UploadFile = File(...),
):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    defect_predictor = getattr(pipeline, "defect_predictor", None)
    if defect_predictor is None:
        defect_predictor = pipeline.modules.get("defect")
    if defect_predictor is None:
        raise HTTPException(status_code=503, detail="Defect predictor not available")

    ref_bytes  = await reference.read()
    test_bytes = await test.read()
    ref_pil    = Image.open(io.BytesIO(ref_bytes)).convert("RGB")
    test_pil   = Image.open(io.BytesIO(test_bytes)).convert("RGB")

    try:
        result = defect_predictor.compare_with_reference(
            reference_image=ref_pil,
            test_image=test_pil,
            generate_heatmap=True,
        )
    except Exception as e:
        logger.error(f"Reference comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    result.pop("overlay_pil", None)

    buf = io.BytesIO()
    test_pil.save(buf, format="JPEG")
    result["test_image_b64"] = base64.b64encode(buf.getvalue()).decode("utf-8")
    return result


if __name__ == "__main__":
    import uvicorn
    import sys
    from pathlib import Path
    
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True, app_dir=project_root)