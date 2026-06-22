import os
import re
import io
import requests
from bs4 import BeautifulSoup
from loguru import logger


def scrape_product_image_url(product_page_url: str, api_key: str = None) -> str:
   
    import urllib.parse
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        if api_key:
            encoded = urllib.parse.quote_plus(product_page_url)
            fetch_url = f"http://api.scraperapi.com?api_key={api_key}&url={encoded}&render=true"
        else:
            fetch_url = product_page_url

        resp = requests.get(fetch_url, headers=headers, timeout=30)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        img = soup.find("img", {"id": "landingImage"})
        if img:
            return img.get("data-old-hires") or img.get("src")

        img = soup.find("img", {"id": "main-image"})
        if img:
            return img.get("src")

        # Generic fallback: first large img
        for tag in soup.find_all("img"):
            src = tag.get("src", "")
            if src.startswith("http") and any(ext in src for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                return src

    except Exception as e:
        logger.warning(f"Image scraping failed: {e}")
    return None


def extract_asin_from_url(url: str) -> str:
    if not url:
        return None
    match = re.search(r"/dp/([A-Z0-9]{10})", url, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"/gp/product/([A-Z0-9]{10})", url, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"/product-reviews/([A-Z0-9]{10})", url, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

def verify_product_exists(asin: str, domain: str, api_key: str) -> bool:
    import urllib.parse
    url = f"https://{domain}/dp/{asin}"
    encoded_url = urllib.parse.quote_plus(url)
    scraper_url = f"http://api.scraperapi.com?api_key={api_key}&url={encoded_url}"

    try:
        response = requests.get(scraper_url, timeout=70)
        if response.status_code == 200 and len(response.text) > 5000:
            return True
        logger.warning(
            f"Pre-flight check failed for {url} — "
            f"status: {response.status_code}, "
            f"content length: {len(response.text)}"
        )
        return False
    except Exception as e:
        logger.warning(f"Pre-flight check error: {e}")
        return False


def scrape_amazon_reviews_apify(asin: str, domain: str = "www.amazon.com", max_reviews: int = 20) -> dict:
    import time, urllib.parse

    token = os.getenv("APIFY_TOKEN", "").strip("'\"")
    if not token:
        return None  

    clean_domain = domain.replace("www.", "")  # e.g. amazon.in

    product_url = f"https://{domain}/dp/{asin}"
    logger.info(f"Scraping via Apify actor for ASIN {asin} on {clean_domain}...")

    run_url = "https://api.apify.com/v2/acts/junglee~amazon-reviews-scraper/runs"
    payload = {
        "productUrls": [{"url": product_url}],
        "maxReviews": max_reviews,
        "proxy": {"useApifyProxy": True},
    }
    headers = {"Content-Type": "application/json"}

    try:
        run_resp = requests.post(
            run_url,
            json=payload,
            headers=headers,
            params={"token": token},
            timeout=30,
        )
        if run_resp.status_code not in (200, 201):
            logger.error(f"Apify run start failed: {run_resp.status_code} — {run_resp.text[:200]}")
            return None

        run_data = run_resp.json()
        run_id = run_data.get("data", {}).get("id")
        if not run_id:
            logger.error("Apify did not return a run ID.")
            return None

        logger.info(f"Apify run started (id={run_id}). Polling for completion...")

        status_url = f"https://api.apify.com/v2/actor-runs/{run_id}"
        for attempt in range(30):  
            time.sleep(3)
            status_resp = requests.get(status_url, params={"token": token}, timeout=15)
            status = status_resp.json().get("data", {}).get("status", "")
            if status == "SUCCEEDED":
                break
            if status in ("FAILED", "ABORTED", "TIMED-OUT"):
                logger.error(f"Apify run {run_id} ended with status: {status}")
                return None

        dataset_id = status_resp.json().get("data", {}).get("defaultDatasetId")
        if not dataset_id:
            logger.error("Apify run succeeded but no dataset ID found.")
            return None

        items_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
        items_resp = requests.get(
            items_url,
            params={"token": token, "format": "json", "limit": max_reviews},
            timeout=20,
        )
        items = items_resp.json()

        if not items:
            logger.warning("Apify returned empty dataset.")
            return None

        reviews, ratings, product_name = [], [], f"Amazon Product ({asin})"
        review_image_urls = []   

        for item in items:
            if not reviews:
                product_info = item.get("product") or {}
                product_name = (
                    item.get("productTitle")
                    or item.get("productName")
                    or item.get("title")
                    or product_info.get("title")
                    or product_info.get("productTitle")
                    or product_name
                )

            text = (
                item.get("reviewDescription")
                or item.get("text")
                or item.get("reviewText")
                or item.get("body", "")
            )
            rating = (
                item.get("ratingScore")
                or item.get("rating")
                or item.get("starRating")
                or item.get("stars", 5)
            )
            if text:
                reviews.append(str(text).strip())
                try:
                    ratings.append(int(float(str(rating).split(" ")[0])))
                except Exception:
                    ratings.append(5)

            imgs = (
                item.get("reviewImages")
                or item.get("images")
                or item.get("attachments")
                or []
            )
            if isinstance(imgs, list):
                for img in imgs:
                    if isinstance(img, str) and img.startswith("http"):
                        review_image_urls.append(img)
                    elif isinstance(img, dict):
                        url = img.get("url") or img.get("src") or img.get("large") or img.get("thumbnail")
                        if url and url.startswith("http"):
                            review_image_urls.append(url)

        image_url = None
        if items:
            product_info = items[0].get("product") or {}
            image_url = (
                product_info.get("mainImage")
                or product_info.get("thumbnailImage")
                or product_info.get("image")
                or (product_info.get("images") or [None])[0]
            )
            if not image_url:
                api_key = os.getenv("SCRAPERAPI_KEY", "").strip("'\"")
                image_url = scrape_product_image_url(
                    f"https://{domain}/dp/{asin}",
                    api_key=api_key or None
                )

        logger.info(
            f"Apify returned {len(reviews)} reviews, "
            f"{len(review_image_urls)} review images for ASIN {asin}. "
            f"Listing image: {'found' if image_url else 'not found'}"
        )
        return {
            "asin": asin,
            "product_name": product_name,
            "reviews": reviews,
            "ratings": ratings,
            "image_url": image_url,
            "review_image_urls": review_image_urls,  
            "source": "apify",
        }

    except Exception as e:
        logger.error(f"Apify scraping failed: {e}")
        return None


def _extract_review_divs(soup) -> list:
    selectors = [
        "review-body", "review-text-content", "review-text", "reviewContent", 
        "a-expander-partial-collapse-content", "cr-original-review-content",
        "review-collapsed", "review-description"
    ]
    for selector in selectors:
        divs = soup.find_all(attrs={"data-hook": selector})
        if divs:
            return divs
        divs = soup.find_all(class_=selector)
        if divs:
            return divs
    return []


def scrape_amazon_reviews_scraperapi(asin: str, domain: str = "www.amazon.com", num_pages: int = 1) -> dict:
    api_key = os.getenv("SCRAPERAPI_KEY", "").strip("'\"")
    if not api_key:
        logger.warning("SCRAPERAPI_KEY not found.")
        return None

    reviews = []
    ratings = []
    product_name = f"Amazon Product ({asin})"
    image_url = None
    review_image_urls = []
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }

    country_code = "us"
    if "amazon.in" in domain:
        country_code = "in"
    elif "amazon.co.uk" in domain:
        country_code = "gb"
    elif "amazon.ca" in domain:
        country_code = "ca"

    import urllib.parse

    for page in range(1, num_pages + 1):
        url = f"https://{domain}/product-reviews/{asin}/ref=cm_cr_arp_d_viewopt_srt?reviewerType=all_reviews&pageNumber={page}&sortBy=helpful"
        encoded_url = urllib.parse.quote_plus(url)
        scraper_url = f"http://api.scraperapi.com?api_key={api_key}&url={encoded_url}&country_code={country_code}"

        try:
            logger.info(f"Requesting page {page} for ASIN {asin} via ScraperAPI...")
            response = requests.get(scraper_url, headers=headers, timeout=40)

            if response.status_code == 404 and page == 1:
                logger.warning(f"Product reviews page returned 404. Attempting main product page fallback...")
                url = f"https://{domain}/dp/{asin}"
                encoded_url = urllib.parse.quote_plus(url)
                scraper_url = f"http://api.scraperapi.com?api_key={api_key}&url={encoded_url}&country_code={country_code}"
                response = requests.get(scraper_url, headers=headers, timeout=40)

            if response.status_code != 200:
                logger.error(f"ScraperAPI returned status code: {response.status_code}")
                break

            soup = BeautifulSoup(response.text, "html.parser")

            if page == 1:
                title_el = soup.find("a", {"data-hook": "product-link"})
                if title_el:
                    product_name = title_el.text.strip()
                else:
                    dp_title = soup.find(id="productTitle") or soup.find("h1", id="title")
                    if dp_title:
                        product_name = dp_title.text.strip()
                    elif soup.title:
                        product_name = soup.title.text.replace("Amazon.com: ", "").replace("Amazon.in: ", "").strip()

                img_el = soup.find("img", id="landingImage") or soup.find("img", id="imgBlkFront")
                if img_el:
                    dyn_img = img_el.get("data-a-dynamic-image")
                    if dyn_img:
                        try:
                            import json as _js
                            urls = list(_js.loads(dyn_img).keys())
                            if urls:
                                image_url = urls[0]
                        except Exception:
                            pass
                    if not image_url:
                        image_url = img_el.get("src") or img_el.get("data-old-hires")

                for img in soup.find_all("img", {"data-hook": "review-image-img"}):
                    src = img.get("src") or img.get("data-src")
                    if src and src.startswith("http"):
                        import re as _re
                        large_src = _re.sub(r"\._[A-Z0-9_]+\.(jpg|jpeg|png)", ".\\1", src)
                        review_image_urls.append(large_src)

            review_divs = _extract_review_divs(soup)
            rating_elements = soup.find_all("i", {"data-hook": "review-star-rating"}) or soup.find_all("i", class_="review-rating")

            if not review_divs:
                logger.warning(f"No reviews found on page {page}.")
                break
                
            for idx, el in enumerate(review_divs):
                text = el.text.strip()
                if text.endswith("Read more"):
                    text = text[:-9].strip()
                reviews.append(text)
                
                if idx < len(rating_elements):
                    rating_text = rating_elements[idx].text.strip()
                    try:
                        val = float(rating_text.split(" ")[0])
                        ratings.append(int(val))
                    except Exception:
                        ratings.append(5)
                else:
                    ratings.append(5)

        except Exception as e:
            logger.error(f"Error scraping reviews via ScraperAPI on page {page}: {e}")
            break

    if not reviews:
        return None

    return {
        "asin": asin,
        "product_name": product_name,
        "reviews": reviews,
        "ratings": ratings,
        "image_url": image_url,
        "review_image_urls": review_image_urls,
        "source": "scraper_api"
    }


def scrape_amazon_reviews(asin: str, domain: str = "www.amazon.com", num_pages: int = 2) -> dict:

    apify_token = os.getenv("APIFY_TOKEN", "").strip("'\"")
    if apify_token:
        logger.info(f"APIFY_TOKEN found. Using Apify for ASIN {asin}...")
        try:
            apify_res = scrape_amazon_reviews_apify(asin, domain=domain, max_reviews=num_pages * 20)
            if apify_res and apify_res.get("reviews"):
                logger.success(f"Apify scrape success: Selected Apify dataset ({len(apify_res['reviews'])} reviews)")
                return apify_res
        except Exception as e:
            logger.error(f"Apify scrape error: {e}")
        logger.warning("Apify returned no reviews. Falling back to ScraperAPI...")

    try:
        scraper_res = scrape_amazon_reviews_scraperapi(asin, domain=domain, num_pages=num_pages)
        if scraper_res and scraper_res.get("reviews"):
            logger.success(f"ScraperAPI scrape success: Selected ScraperAPI dataset ({len(scraper_res['reviews'])} reviews)")
            return scraper_res
    except Exception as e:
        logger.error(f"ScraperAPI scrape error: {e}")
    logger.warning("ScraperAPI scraping was unsuccessful. Trying direct fallback...")

    logger.warning("All premium parallel API services failed. Attempting direct scrape fallback...")
    return direct_scrape_amazon(asin, domain=domain, num_pages=num_pages)


def direct_scrape_amazon(asin: str, domain: str = "www.amazon.com", num_pages: int = 1) -> dict:
    reviews = []
    ratings = []
    product_name = f"Amazon Product ({asin})"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Device-Memory": "8",
    }
    
    for page in range(1, num_pages + 1):
        url = f"https://{domain}/product-reviews/{asin}/ref=cm_cr_arp_d_viewopt_srt?reviewerType=all_reviews&pageNumber={page}&sortBy=recent"
        try:
            logger.info(f"Requesting page {page} for ASIN {asin} directly...")
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 503 or "api-services-support@amazon.com" in response.text:
                logger.error("Amazon direct request blocked (503 Service Unavailable / CAPTCHA).")
                break
                
            if response.status_code != 200:
                logger.error(f"Direct request failed with status: {response.status_code}")
                break
                
            soup = BeautifulSoup(response.text, "html.parser")
            
            if page == 1:
                title_el = soup.find("a", {"data-hook": "product-link"})
                if title_el:
                    product_name = title_el.text.strip()
                    
            review_divs = _extract_review_divs(soup)
            rating_elements = soup.find_all("i", {"data-hook": "review-star-rating"})
            
            if not review_divs:
                logger.warning("No review bodies found in HTML. Direct requests blocked.")
                break
                
            for idx, el in enumerate(review_divs):
                text = el.text.strip()
                if text.endswith("Read more"):
                    text = text[:-9].strip()
                reviews.append(text)
                
                if idx < len(rating_elements):
                    rating_text = rating_elements[idx].text.strip()
                    try:
                        val = float(rating_text.split(" ")[0])
                        ratings.append(int(val))
                    except Exception:
                        ratings.append(5)
                else:
                    ratings.append(5)
                    
        except Exception as e:
            logger.error(f"Error scraping reviews directly on page {page}: {e}")
            break
            
    return {
        "asin": asin,
        "product_name": product_name,
        "reviews": reviews,
        "ratings": ratings,
        "source": "direct_scrape"
    }


def scrape_ecommerce_reviews(url: str, num_pages: int = 2) -> dict:
    url_lower = url.lower()
    if "amazon." in url_lower:
        logger.info(f"Routing URL to Amazon scraper: {url}")
        asin = extract_asin_from_url(url)
        if not asin:
            return {
                "asin": None,
                "product_name": "Unknown Product",
                "reviews": [],
                "ratings": [],
                "image_url": None,
                "review_image_urls": [],
                "source": "failed",
            }
        domain_match = re.search(r"amazon\.(com|in|co\.uk|de|ca)", url, re.IGNORECASE)
        domain = "www.amazon." + (domain_match.group(1).lower() if domain_match else "com")
        
        result = scrape_amazon_reviews(asin, domain=domain, num_pages=num_pages)
        
        return result
    else:
        logger.error(f"Unsupported URL domain — only Amazon is supported: {url}")
        return {
            "asin": None,
            "product_name": "Unknown Product",
            "reviews": [],
            "ratings": [],
            "image_url": None,
            "review_image_urls": [],
            "source": "unsupported",
        }