import asyncio
import aiohttp
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import bs4
import pandas as pd
import json
import time
import random
from urllib.parse import urljoin, urlparse
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Set
import re
from fake_useragent import UserAgent
from datetime import datetime
import os
import hashlib
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import gc
import warnings
import ssl
import certifi
import tldextract
import nest_asyncio
from bs4 import XMLParsedAsHTMLWarning
import socket
from urllib.robotparser import RobotFileParser

# Fix warnings and SSL
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
nest_asyncio.apply()

# GPU/CPU Detection
def detect_device():
    """Detect processing capability"""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 GPU Detected: {gpu_name} ({gpu_memory:.1f} GB)")
            return device, 'gpu', int(gpu_memory)
    except ImportError:
        pass
    
    cpu_count = mp.cpu_count()
    print(f"💻 Using CPU: {cpu_count} cores")
    return None, 'cpu', cpu_count

DEVICE, DEVICE_TYPE, DEVICE_CAPACITY = detect_device()

# Enhanced logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class ScrapedContent:
    """Enhanced data class for scraped content"""
    url: str
    title: str
    content: str
    word_count: int
    quality_score: float
    timestamp: str
    domain: str
    content_type: str
    language: str
    links_found: List[str]
    hash_id: str

class UltraRobustJSONLDatabase:
    """Ultra-robust JSONL database with comprehensive error handling"""
    
    def __init__(self, db_file: str = "llm_dataset.jsonl"):
        self.db_file = db_file
        self.links_file = "discovered_links.jsonl"
        self.scraped_urls = set()
        self.failed_urls = set()
        self.robots_cache = {}
        self.domain_delays = {}
        self.load_existing_data()
    
    def load_existing_data(self):
        """Load existing data with error handling"""
        # Load scraped URLs
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            try:
                                data = json.loads(line)
                                self.scraped_urls.add(data.get('url', ''))
                            except json.JSONDecodeError:
                                logging.warning(f"Invalid JSON at line {line_num}")
                logging.info(f"✅ Loaded {len(self.scraped_urls)} existing URLs")
            except Exception as e:
                logging.error(f"Error loading database: {e}")
        
        # Load failed URLs
        failed_file = "failed_urls.jsonl"
        if os.path.exists(failed_file):
            try:
                with open(failed_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                self.failed_urls.add(data.get('url', ''))
                            except:
                                pass
            except Exception as e:
                logging.warning(f"Could not load failed URLs: {e}")
    
    def save_content(self, content: ScrapedContent) -> bool:
        """Save content with atomic operations"""
        if content.url in self.scraped_urls:
            return False
        
        try:
            # Write to temporary file first
            temp_file = f"{self.db_file}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(json.dumps(asdict(content), ensure_ascii=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
            
            # Atomic append
            with open(self.db_file, 'a', encoding='utf-8') as main_file:
                with open(temp_file, 'r', encoding='utf-8') as temp:
                    main_file.write(temp.read())
            
            os.remove(temp_file)
            self.scraped_urls.add(content.url)
            return True
            
        except Exception as e:
            logging.error(f"Error saving content: {e}")
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            return False
    
    def save_failed_url(self, url: str, error: str):
        """Save failed URLs for analysis"""
        try:
            with open("failed_urls.jsonl", 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'url': url,
                    'error': str(error)[:200],  # Limit error message length
                    'timestamp': datetime.now().isoformat()
                }, ensure_ascii=False) + '\n')
            self.failed_urls.add(url)
        except Exception as e:
            logging.error(f"Error saving failed URL: {e}")
    
    def save_links(self, links: Set[str], source_url: str):
        """Save discovered links efficiently"""
        if not links:
            return
            
        try:
            with open(self.links_file, 'a', encoding='utf-8') as f:
                for link in links:
                    if (link not in self.scraped_urls and 
                        link not in self.failed_urls and 
                        len(link) < 500):  # Avoid extremely long URLs
                        
                        link_data = {
                            'url': link,
                            'source_url': source_url,
                            'discovered_at': datetime.now().isoformat()
                        }
                        f.write(json.dumps(link_data, ensure_ascii=False) + '\n')
        except Exception as e:
            logging.error(f"Error saving links: {e}")
    
    def get_unscraped_links(self, limit: int = 1000) -> List[str]:
        """Get unique unscraped links with domain balancing"""
        unscraped = []
        seen_links = set()
        domain_count = {}
        
        if not os.path.exists(self.links_file):
            return unscraped
        
        try:
            with open(self.links_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            url = data.get('url', '')
                            
                            if (url and url not in self.scraped_urls and 
                                url not in self.failed_urls and 
                                url not in seen_links and len(unscraped) < limit):
                                
                                # Balance domains to avoid overwhelming single domain
                                try:
                                    domain = urlparse(url).netloc
                                    if domain_count.get(domain, 0) < 50:  # Max 50 per domain
                                        unscraped.append(url)
                                        seen_links.add(url)
                                        domain_count[domain] = domain_count.get(domain, 0) + 1
                                except:
                                    continue
                                    
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logging.error(f"Error reading links: {e}")
        
        return unscraped
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        stats = {
            'total_scraped': len(self.scraped_urls),
            'total_failed': len(self.failed_urls),
            'high_quality': 0,
            'total_words': 0,
            'avg_quality': 0.0,
            'domains': {},
            'content_types': {}
        }
        
        if not os.path.exists(self.db_file):
            return stats
        
        quality_scores = []
        try:
            with open(self.db_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            quality = data.get('quality_score', 0)
                            quality_scores.append(quality)
                            
                            if quality >= 0.8:
                                stats['high_quality'] += 1
                            
                            stats['total_words'] += data.get('word_count', 0)
                            
                            domain = data.get('domain', 'unknown')
                            stats['domains'][domain] = stats['domains'].get(domain, 0) + 1
                            
                            content_type = data.get('content_type', 'unknown')
                            stats['content_types'][content_type] = stats['content_types'].get(content_type, 0) + 1
                            
                        except json.JSONDecodeError:
                            continue
            
            if quality_scores:
                stats['avg_quality'] = sum(quality_scores) / len(quality_scores)
                
        except Exception as e:
            logging.error(f"Error calculating stats: {e}")
        
        return stats

class UltraAdvancedWebScraper:
    """Ultra-advanced web scraper with comprehensive fixes"""
    
    def __init__(self, use_selenium: bool = False):
        self.device_type = DEVICE_TYPE
        self.device_capacity = DEVICE_CAPACITY
        self.use_selenium = use_selenium
        self.user_agent = UserAgent()
        self.db = UltraRobustJSONLDatabase()
        
        # Optimize workers based on device
        if self.device_type == 'gpu':
            self.max_workers = min(12, int(self.device_capacity * 1.5))  # More conservative
        else:
            self.max_workers = min(6, self.device_capacity)
        
        print(f"🔧 Configured with {self.max_workers} workers for {self.device_type.upper()}")
        
        # Setup session with enhanced configuration
        self.setup_session()
        
        if self.use_selenium:
            self.setup_selenium()
    
    def setup_session(self):
        """Setup ultra-robust HTTP session"""
        self.session = requests.Session()
        
        # Enhanced headers for better compatibility
        self.session.headers.update({
            'User-Agent': self.user_agent.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
            'DNT': '1'
        })
        
        # Enhanced retry strategy
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=5,  # More retries
            backoff_factor=2,  # Longer backoff
            status_forcelist=[429, 500, 502, 503, 504, 520, 521, 522, 523, 524],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=20,
            pool_maxsize=20
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Enhanced SSL configuration
        self.session.verify = certifi.where()
        
        # Set reasonable timeout
        self.session.timeout = (10, 30)  # (connect, read)
    
    def is_valid_url(self, url: str) -> bool:
        """Ultra-enhanced URL validation with comprehensive filtering"""
        if not url or len(url) < 10 or len(url) > 2000:
            return False
        
        try:
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                return False
            
            # Check scheme
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Enhanced file extension filtering
            unwanted_extensions = {
                '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
                '.zip', '.rar', '.tar', '.gz', '.7z', '.exe', '.dmg', '.pkg',
                '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.ico', '.webp',
                '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv',
                '.css', '.js', '.xml', '.rss', '.atom', '.json', '.txt', '.csv',
                '.ps', '.eps', '.ai', '.psd', '.tiff', '.raw', '.wav', '.flac',
                '.bin', '.iso', '.deb', '.rpm', '.apk', '.ipa', '.msi'
            }
            
            path_lower = parsed.path.lower()
            if any(path_lower.endswith(ext) for ext in unwanted_extensions):
                return False
            
            # Enhanced domain filtering
            domain = parsed.netloc.lower()
            
            # Block problematic domains
            blocked_domains = {
                'web.archive.org',  # Causing connection issues
                'archive.org',
                'wayback.machine.org',
                'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com', 'tiktok.com',
                'youtube.com', 'vimeo.com', 'dailymotion.com',
                'doubleclick.net', 'googletagmanager.com', 'google-analytics.com',
                'ads.', 'analytics.', 'tracking.', 'metrics.',
                'login.', 'signin.', 'signup.', 'register.'
            }
            
            if any(blocked in domain for blocked in blocked_domains):
                return False
            
            # Block citation and reference patterns
            unwanted_patterns = [
                'doi.org', 'dx.doi.org', 'hdl.handle.net', 'orcid.org',
                'api.semanticscholar.org', 'citeseerx.ist.psu.edu',
                'static/favicon', '/favicon.', '.ico', '#cite', '#ref',
                'javascript:', 'mailto:', 'tel:', 'ftp://', 'file://',
                'upload.wikimedia.org/wikipedia/commons/thumb',
                '/wiki/File:', '/wiki/Category:', '/wiki/Template:',
                'edit?', 'action=edit', 'oldid=', 'diff='
            ]
            
            if any(pattern in url.lower() for pattern in unwanted_patterns):
                return False
            
            # Check for extremely long query parameters (often junk)
            if len(parsed.query) > 200:
                return False
            
            return True
            
        except Exception:
            return False
    
    def extract_links(self, soup: bs4.BeautifulSoup, base_url: str) -> Set[str]:
        """Enhanced and safe link extraction"""
        links = set()
        
        try:
            # Focus on main content links only
            main_selectors = [
                'main a[href]',
                'article a[href]',
                '.content a[href]',
                '.post-content a[href]',
                '.entry-content a[href]',
                'p a[href]',
                'div a[href]'
            ]
            
            for selector in main_selectors:
                try:
                    elements = soup.select(selector)
                    for element in elements[:50]:  # Limit per selector
                        href = element.get('href')
                        if href:
                            try:
                                full_url = urljoin(base_url, href)
                                if self.is_valid_url(full_url) and len(links) < 100:
                                    links.add(full_url)
                            except Exception:
                                continue
                except Exception:
                    continue
            
            # If no main content links, fallback to general links
            if len(links) < 20:
                try:
                    elements = soup.select('a[href]')
                    for element in elements[:100]:
                        href = element.get('href')
                        if href:
                            try:
                                full_url = urljoin(base_url, href)
                                if self.is_valid_url(full_url) and len(links) < 100:
                                    links.add(full_url)
                            except Exception:
                                continue
                except Exception:
                    pass
                    
        except Exception as e:
            logging.warning(f"Link extraction error: {e}")
        
        return links
    
    def calculate_quality(self, content: str, title: str = "") -> float:
        """Enhanced quality calculation with better metrics"""
        if not content or len(content.strip()) < 20:
            return 0.0
        
        score = 0.0
        words = content.split()
        word_count = len(words)
        
        # Enhanced word count scoring
        if 20 <= word_count <= 50:
            score += 0.2
        elif 50 <= word_count <= 200:
            score += 0.4
        elif 200 <= word_count <= 1000:
            score += 0.5
        elif 1000 <= word_count <= 3000:
            score += 0.4
        elif word_count > 3000:
            score += 0.3
        
        # Content quality indicators
        sentences = content.split('.')
        if len(sentences) > 3:
            score += 0.1
        
        # Check for proper capitalization
        if any(c.isupper() for c in content[:500]):
            score += 0.1
        
        # Check for punctuation variety
        punct_count = sum(content.count(p) for p in '.!?,:;')
        if punct_count >= 5:
            score += 0.1
        
        # Vocabulary diversity (enhanced)
        if word_count > 20:
            sample_words = words[:200]  # Sample for efficiency
            unique_words = set(w.lower().strip('.,!?;:') for w in sample_words if len(w) > 2)
            diversity = len(unique_words) / len(sample_words) if sample_words else 0
            score += diversity * 0.3
        
        # Title quality bonus
        if title and len(title.split()) >= 3 and len(title) <= 200:
            score += 0.1
        
        # Educational content indicators
        educational_indicators = [
            'research', 'study', 'analysis', 'theory', 'method', 'approach',
            'algorithm', 'technology', 'science', 'university', 'academic',
            'journal', 'paper', 'article', 'learning', 'intelligence'
        ]
        
        content_lower = content.lower()
        edu_count = sum(1 for indicator in educational_indicators if indicator in content_lower)
        if edu_count >= 3:
            score += 0.1
        elif edu_count >= 5:
            score += 0.2
        
        # Penalize low-quality indicators
        spam_indicators = [
            'click here', 'buy now', 'limited time', 'act now', 'free money',
            'make money fast', '100% free', 'guaranteed', 'miracle',
            'subscribe now', 'sign up', 'download now'
        ]
        
        spam_count = sum(1 for spam in spam_indicators if spam.lower() in content_lower)
        score -= spam_count * 0.15
        
        # Penalize repetitive content
        if word_count > 50:
            word_freq = {}
            for word in words[:100]:  # Sample for efficiency
                word_clean = word.lower().strip('.,!?;:')
                if len(word_clean) > 3:
                    word_freq[word_clean] = word_freq.get(word_clean, 0) + 1
            
            if word_freq:
                max_freq = max(word_freq.values())
                if max_freq / len(word_freq) > 0.3:  # Too repetitive
                    score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def clean_content(self, content: str) -> str:
        """Ultra-comprehensive content cleaning"""
        if not content:
            return ""
        
        try:
            # Remove extra whitespace and normalize
            content = re.sub(r'\s+', ' ', content)
            
            # Remove URLs (enhanced pattern)
            content = re.sub(r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?', '', content)
            
            # Remove email addresses
            content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', content)
            
            # Remove phone numbers (enhanced)
            content = re.sub(r'(\+\d{1,3}[-.\s]?)?\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}', '', content)
            
            # Remove excessive punctuation
            content = re.sub(r'[.]{3,}', '...', content)
            content = re.sub(r'[!]{2,}', '!', content)
            content = re.sub(r'[?]{2,}', '?', content)
            content = re.sub(r'[-]{3,}', '---', content)
            
            # Remove citations in brackets/parentheses
            content = re.sub(r'\[[^\]]*\]', '', content)
            content = re.sub(r'\([^)]*\d[^)]*\)', '', content)  # Remove citations with numbers
            
            # Remove wiki-style markup
            content = re.sub(r'\{\{[^}]*\}\}', '', content)
            content = re.sub(r'\[\[([^|]+\|)?([^\]]+)\]\]', r'\2', content)
            
            # Remove navigation text
            nav_patterns = [
                r'jump to navigation', r'jump to search', r'edit source',
                r'view history', r'main page', r'contents', r'current events',
                r'random article', r'about wikipedia', r'contact us', r'donate'
            ]
            for pattern in nav_patterns:
                content = re.sub(pattern, '', content, flags=re.IGNORECASE)
            
            # Clean special characters but keep essential punctuation
            content = re.sub(r'[^\w\s.!?,:;()\-\'\"]+', ' ', content)
            
            # Fix spacing around punctuation
            content = re.sub(r'\s+([.!?,:;)])', r'\1', content)
            content = re.sub(r'([(])\s+', r'\1', content)
            
            # Remove very short lines and fragments
            lines = [line.strip() for line in content.split('\n') if len(line.strip()) > 15]
            content = ' '.join(lines)
            
            # Remove repeated phrases
            sentences = content.split('.')
            unique_sentences = []
            seen = set()
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:
                    # Create a fingerprint of the sentence
                    fingerprint = ' '.join(sorted(sentence.lower().split()[:10]))
                    if fingerprint not in seen:
                        unique_sentences.append(sentence)
                        seen.add(fingerprint)
            
            content = '. '.join(unique_sentences)
            
            # Final cleanup
            content = re.sub(r'\s+', ' ', content).strip()
            
            return content
            
        except Exception as e:
            logging.warning(f"Content cleaning error: {e}")
            return content.strip() if content else ""
    
    def detect_content_encoding(self, response_content: bytes, content_type: str = "") -> str:
        """Enhanced content encoding detection"""
        # Try to get encoding from content-type header
        if content_type:
            charset_match = re.search(r'charset=([^;]+)', content_type.lower())
            if charset_match:
                try:
                    return response_content.decode(charset_match.group(1).strip())
                except:
                    pass
        
        # Try common encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'ascii', 'utf-16']
        
        for encoding in encodings:
            try:
                decoded = response_content.decode(encoding)
                # Validate that it looks like text
                if len(decoded) > 0 and not any(ord(c) < 32 and c not in '\t\n\r' for c in decoded[:1000]):
                    return decoded
            except UnicodeDecodeError:
                continue
        
        # Fallback: decode with errors ignored
        try:
            return response_content.decode('utf-8', errors='ignore')
        except:
            return ""
    
    async def scrape_url_async(self, session: aiohttp.ClientSession, url: str) -> Optional[ScrapedContent]:
        """Ultra-robust async URL scraping with comprehensive error handling"""
        if url in self.db.scraped_urls or url in self.db.failed_urls:
            return None
        
        try:
            # Enhanced timeout and connection settings
            timeout = aiohttp.ClientTimeout(
                total=45,
                connect=15,
                sock_read=30
            )
            
            # Create custom SSL context
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Enhanced connector
            connector = aiohttp.TCPConnector(
                ssl=ssl_context,
                limit=100,
                limit_per_host=10,
                keepalive_timeout=30,
                enable_cleanup_closed=True,
                family=socket.AF_INET  # Force IPv4
            )
            
            async with session.get(
                url,
                timeout=timeout,
                allow_redirects=True,
                max_redirects=5,
                ssl=ssl_context
            ) as response:
                
                if response.status != 200:
                    self.db.save_failed_url(url, f"HTTP {response.status}")
                    return None
                
                # Enhanced content type checking
                content_type = response.headers.get('content-type', '').lower()
                if not any(ct in content_type for ct in ['text/html', 'application/xhtml', 'text/plain']):
                    self.db.save_failed_url(url, f"Invalid content type: {content_type}")
                    return None
                
                # Read content with size limit
                content_bytes = await response.read()
                if len(content_bytes) > 10 * 1024 * 1024:  # 10MB limit
                    self.db.save_failed_url(url, "Content too large")
                    return None
                
                # Enhanced encoding detection
                html_content = self.detect_content_encoding(content_bytes, content_type)
                
                if not html_content or len(html_content) < 200:
                    self.db.save_failed_url(url, "Empty or too short content")
                    return None
                
                # Parse with BeautifulSoup
                soup = bs4.BeautifulSoup(html_content, 'html.parser')
                
                # Extract links first (before content modification)
                new_links = self.extract_links(soup, url)
                if new_links:
                    self.db.save_links(new_links, url)
                
                # Enhanced content extraction
                title_element = soup.find('title')
                title_text = title_element.get_text().strip() if title_element else ""
                
                # Remove unwanted elements more aggressively
                unwanted_tags = [
                    'script', 'style', 'nav', 'footer', 'header', 'aside',
                    'form', 'button', 'input', 'select', 'textarea',
                    'iframe', 'embed', 'object', 'applet', 'audio', 'video',
                    'canvas', 'svg', 'map', 'area', 'noscript',
                    '.navbox', '.infobox', '.sidebar', '.toc', '.references'
                ]
                
                for tag_name in unwanted_tags:
                    if tag_name.startswith('.'):
                        # CSS class selector
                        for elem in soup.select(tag_name):
                            elem.decompose()
                    else:
                        # Tag name
                        for tag in soup.find_all(tag_name):
                            tag.decompose()
                
                # Enhanced main content extraction
                main_content = ""
                
                # Try multiple content extraction strategies
                content_strategies = [
                    # Strategy 1: Semantic HTML5 elements
                    lambda: ' '.join([elem.get_text().strip() for elem in 
                                    soup.select('main, article, [role="main"]') if elem.get_text().strip()]),
                    
                    # Strategy 2: Common content classes
                    lambda: ' '.join([elem.get_text().strip() for elem in 
                                    soup.select('.content, .post-content, .entry-content, .article-body, .story-body') if elem.get_text().strip()]),
                    
                    # Strategy 3: Wikipedia-specific
                    lambda: ' '.join([elem.get_text().strip() for elem in 
                                    soup.select('.mw-parser-output p, .mw-parser-output div') if elem.get_text().strip()]),
                    
                    # Strategy 4: General paragraphs and divs
                    lambda: ' '.join([elem.get_text().strip() for elem in 
                                    soup.select('p, div') if elem.get_text().strip() and len(elem.get_text().strip()) > 50])
                ]
                
                for strategy in content_strategies:
                    try:
                        content = strategy()
                        if content and len(content.split()) > 50:
                            main_content = content
                            break
                    except Exception:
                        continue
                
                # Fallback: all text content
                if not main_content:
                    main_content = soup.get_text()
                
                # Clean content thoroughly
                cleaned_content = self.clean_content(main_content)
                
                if not cleaned_content or len(cleaned_content.split()) < 30:
                    self.db.save_failed_url(url, "Insufficient quality content after cleaning")
                    return None
                
                # Calculate quality with stricter threshold
                quality_score = self.calculate_quality(cleaned_content, title_text)
                
                if quality_score < 0.4:  # Increased threshold
                    self.db.save_failed_url(url, f"Low quality score: {quality_score:.3f}")
                    return None
                
                # Extract domain info safely
                try:
                    domain_info = tldextract.extract(url)
                    domain = domain_info.domain or urlparse(url).netloc
                except:
                    domain = urlparse(url).netloc or 'unknown'
                
                # Create content object
                scraped_content = ScrapedContent(
                    url=url,
                    title=title_text[:300],  # Increased title limit
                    content=cleaned_content,
                    word_count=len(cleaned_content.split()),
                    quality_score=quality_score,
                    timestamp=datetime.now().isoformat(),
                    domain=domain,
                    content_type='text/html',
                    language='en',  # Could be enhanced with language detection
                    links_found=list(new_links)[:30],  # Increased links limit
                    hash_id=hashlib.md5(cleaned_content.encode()).hexdigest()
                )
                
                logging.info(f"✅ Scraped: {url} (Q: {quality_score:.3f}, Words: {scraped_content.word_count})")
                return scraped_content
                
        except asyncio.TimeoutError:
            self.db.save_failed_url(url, "Connection timeout")
            logging.warning(f"⏰ Timeout: {url}")
        except aiohttp.ClientError as e:
            error_msg = str(e)[:100]  # Limit error message length
            self.db.save_failed_url(url, f"Client error: {error_msg}")
            logging.warning(f"🌐 Client error for {url}: {error_msg}")
        except Exception as e:
            error_msg = str(e)[:100]
            self.db.save_failed_url(url, f"Unexpected error: {error_msg}")
            logging.error(f"❌ Unexpected error for {url}: {error_msg}")
        
        return None
    
    async def scrape_batch_async(self, urls: List[str]) -> List[ScrapedContent]:
        """Ultra-optimized batch scraping with enhanced error handling"""
        results = []
        
        # Enhanced connector configuration
        connector = aiohttp.TCPConnector(
            limit=self.max_workers * 2,
            limit_per_host=3,  # Very conservative
            keepalive_timeout=60,
            enable_cleanup_closed=True,
            family=socket.AF_INET,  # Force IPv4 for better compatibility
            ssl=False  # We handle SSL manually
        )
        
        timeout = aiohttp.ClientTimeout(total=60, connect=15)
        
        # Enhanced headers
        headers = {
            'User-Agent': self.user_agent.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'DNT': '1'
        }
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers
        ) as session:
            
            # Enhanced rate limiting
            semaphore = asyncio.Semaphore(self.max_workers)
            domain_semaphores = {}
            
            async def scrape_with_domain_limit(url):
                domain = urlparse(url).netloc
                
                # Create per-domain semaphore for politeness
                if domain not in domain_semaphores:
                    domain_semaphores[domain] = asyncio.Semaphore(2)  # Max 2 concurrent per domain
                
                async with semaphore:
                    async with domain_semaphores[domain]:
                        # Enhanced respectful delay
                        base_delay = random.uniform(1.0, 3.0)
                        if 'wikipedia' in domain:
                            base_delay = random.uniform(0.5, 1.5)  # Faster for Wikipedia
                        elif domain in self.db.domain_delays:
                            base_delay = max(base_delay, self.db.domain_delays[domain])
                        
                        await asyncio.sleep(base_delay)
                        return await self.scrape_url_async(session, url)
            
            # Filter and prioritize URLs
            valid_urls = []
            for url in urls:
                if self.is_valid_url(url):
                    # Prioritize educational domains
                    priority = 0
                    domain = urlparse(url).netloc.lower()
                    if any(edu in domain for edu in ['wikipedia', 'edu', 'ac.', 'stanford', 'mit', 'harvard']):
                        priority = 1
                    valid_urls.append((priority, url))
            
            # Sort by priority
            valid_urls.sort(key=lambda x: x[0], reverse=True)
            urls_to_process = [url for _, url in valid_urls]
            
            logging.info(f"🔍 Processing {len(urls_to_process)} valid URLs out of {len(urls)}")
            
            # Create tasks
            tasks = [scrape_with_domain_limit(url) for url in urls_to_process]
            
            # Process with enhanced progress tracking
            completed = 0
            batch_size = min(15, self.max_workers)  # Smaller batches
            
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i:i + batch_size]
                
                try:
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    for result in batch_results:
                        if isinstance(result, ScrapedContent):
                            results.append(result)
                            self.db.save_content(result)
                        elif isinstance(result, Exception):
                            logging.error(f"Task exception: {result}")
                    
                    completed += len(batch_tasks)
                    
                    if completed % 10 == 0:
                        success_rate = len(results) / completed * 100 if completed > 0 else 0
                        logging.info(f"🔄 Progress: {completed}/{len(tasks)} ({len(results)} successful, {success_rate:.1f}% success rate)")
                        
                        # Enhanced memory cleanup
                        if self.device_type == 'gpu':
                            try:
                                import torch
                                torch.cuda.empty_cache()
                            except:
                                pass
                        gc.collect()
                            
                except Exception as e:
                    logging.error(f"Batch processing error: {e}")
                    continue
        
        logging.info(f"✅ Batch completed: {len(results)} pages scraped successfully")
        return results
    
    def crawl_website(self, start_url: str, max_pages: int = 500, max_depth: int = 2):
        """Enhanced website crawling with better error handling and recovery"""
        print(f"\n🚀 Starting crawl from: {start_url}")
        print(f"📊 Device: {self.device_type.upper()}, Workers: {self.max_workers}")
        print(f"🎯 Target: {max_pages} pages, Depth: {max_depth}")
        
        depth = 0
        total_scraped = 0
        current_urls = [start_url]
        failed_batches = 0
        max_failed_batches = 3
        
        while current_urls and depth < max_depth and total_scraped < max_pages and failed_batches < max_failed_batches:
            print(f"\n🔍 Depth {depth}: Processing {len(current_urls)} URLs")
            
            # Process in smaller, manageable batches
            batch_size = min(30, self.max_workers * 2)  # Smaller batches for stability
            batch_results = []
            
            for i in range(0, len(current_urls), batch_size):
                if total_scraped >= max_pages:
                    break
                    
                batch = current_urls[i:i + batch_size]
                print(f"📦 Batch {i//batch_size + 1}: {len(batch)} URLs")
                
                try:
                    # Run async batch with timeout
                    loop = asyncio.get_event_loop()
                    results = asyncio.wait_for(
                        self.scrape_batch_async(batch),
                        timeout=300  # 5 minute timeout per batch
                    )
                    results = loop.run_until_complete(results)
                    
                    batch_results.extend(results)
                    total_scraped += len(results)
                    
                    print(f"✅ Batch completed: +{len(results)} pages (Total: {total_scraped})")
                    
                    # Reset failed batch counter on success
                    if len(results) > 0:
                        failed_batches = 0
                    else:
                        failed_batches += 1
                        
                except asyncio.TimeoutError:
                    logging.error(f"Batch timeout after 5 minutes")
                    failed_batches += 1
                    continue
                except Exception as e:
                    logging.error(f"Batch error: {e}")
                    failed_batches += 1
                    continue
            
            # Get new URLs for next depth with better filtering
            if depth < max_depth - 1 and total_scraped < max_pages:
                remaining_quota = max_pages - total_scraped
                new_urls = self.db.get_unscraped_links(min(300, remaining_quota * 3))
                
                # Filter out already processed URLs
                current_urls = [url for url in new_urls if url not in self.db.scraped_urls and url not in self.db.failed_urls]
                
                print(f"🔗 Found {len(current_urls)} new URLs for next depth")
            else:
                current_urls = []
            
            depth += 1
            
            # Show comprehensive stats
            stats = self.db.get_stats()
            success_rate = stats['total_scraped'] / (stats['total_scraped'] + stats['total_failed']) * 100 if (stats['total_scraped'] + stats['total_failed']) > 0 else 0
            
            print(f"📈 Stats: {stats['total_scraped']} scraped, {stats['total_failed']} failed ({success_rate:.1f}% success)")
            print(f"⭐ Quality: {stats['avg_quality']:.3f} avg, {stats['high_quality']} high-quality")
            print(f"📝 Words: {stats['total_words']:,} total")
    
    def export_dataset(self, min_quality: float = 0.4, output_format: str = 'jsonl'):
        """Enhanced dataset export with better filtering and validation"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print(f"💾 Exporting dataset (min_quality: {min_quality})...")
        
        if output_format == 'jsonl':
            output_file = f'llm_dataset_filtered_{timestamp}.jsonl'
            exported_count = 0
            
            try:
                with open(output_file, 'w', encoding='utf-8') as outfile:
                    with open(self.db.db_file, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            if line.strip():
                                try:
                                    data = json.loads(line)
                                    quality = data.get('quality_score', 0)
                                    word_count = data.get('word_count', 0)
                                    
                                    # Enhanced filtering
                                    if (quality >= min_quality and 
                                        word_count >= 30 and 
                                        word_count <= 5000 and
                                        data.get('content', '') and
                                        len(data.get('title', '')) > 5):
                                        
                                        # Format for LLM training
                                        llm_record = {
                                            'text': data['content'],
                                            'title': data['title'],
                                            'url': data['url'],
                                            'quality_score': round(quality, 3),
                                            'word_count': word_count,
                                            'domain': data['domain'],
                                            'timestamp': data['timestamp']
                                        }
                                        outfile.write(json.dumps(llm_record, ensure_ascii=False) + '\n')
                                        exported_count += 1
                                        
                                except json.JSONDecodeError:
                                    continue
                
                print(f"📁 JSONL Dataset exported: {output_file} ({exported_count} records)")
                
            except Exception as e:
                logging.error(f"Export error: {e}")
                return None
        
        elif output_format == 'csv':
            output_file = f'llm_dataset_filtered_{timestamp}.csv'
            data_list = []
            
            try:
                with open(self.db.db_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                quality = data.get('quality_score', 0)
                                word_count = data.get('word_count', 0)
                                
                                if (quality >= min_quality and 
                                    word_count >= 30 and 
                                    word_count <= 5000 and
                                    data.get('content', '') and
                                    len(data.get('title', '')) > 5):
                                    data_list.append(data)
                                    
                            except json.JSONDecodeError:
                                continue
                
                if data_list:
                    df = pd.DataFrame(data_list)
                    df.to_csv(output_file, index=False, encoding='utf-8')
                    print(f"📊 CSV Dataset exported: {output_file} ({len(data_list)} records)")
                
            except Exception as e:
                logging.error(f"CSV export error: {e}")
                return None
        
        return output_file
    
    def cleanup(self):
        """Enhanced cleanup with better error handling"""
        try:
            if hasattr(self, 'driver') and self.driver:
                self.driver.quit()
            if hasattr(self, 'session') and self.session:
                self.session.close()
        except Exception as e:
            logging.error(f"Cleanup error: {e}")

# Enhanced test prompts and quality assessment
def test_dataset_quality():
    """Enhanced test dataset with comprehensive prompts"""
    test_prompts = [
        "What is artificial intelligence and how does it work?",
        "Explain the difference between machine learning and deep learning.",
        "What are the main applications of AI in healthcare?",
        "How do neural networks learn and adapt?",
        "What are the ethical considerations in AI development?",
        "Describe the history and evolution of artificial intelligence.",
        "What is the role of data in machine learning algorithms?",
        "How does natural language processing work in AI systems?",
        "What are the challenges and limitations of current AI technology?",
        "Explain the concept of artificial general intelligence (AGI)."
    ]
    
    print("\n🧪 Dataset Quality Test Prompts:")
    print("=" * 50)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"{i:2d}. {prompt}")
    
    print("\n💡 Use these prompts to test your LLM model's performance!")
    print("📊 A high-quality dataset should enable good responses to these questions.")
    
    # Sample responses for quality benchmarking
    print("\n🎯 Expected Response Quality Indicators:")
    print("✅ Comprehensive explanations (200+ words)")
    print("✅ Technical accuracy and detail")
    print("✅ Structured information presentation")
    print("✅ Real-world examples and applications")
    print("✅ Balanced perspective on benefits and challenges")

# Main execution function with enhanced error handling
def create_ultra_robust_dataset():
    """Ultra-robust main function for dataset creation"""
    
    print("🚀 UltraAdvancedWebScraper - Production Ready++")
    print(f"💾 Using Ultra-Robust JSONL Database")
    print(f"🔧 Device: {DEVICE_TYPE.upper()}")
    print(f"🛡️ Enhanced Error Handling & Recovery")
    
    # Diversified starting URLs for better coverage
    start_urls = [
        'https://en.wikipedia.org/wiki/Artificial_intelligence',
        'https://en.wikipedia.org/wiki/Machine_learning',
        'https://en.wikipedia.org/wiki/Deep_learning',
        'https://en.wikipedia.org/wiki/Natural_language_processing',
        'https://en.wikipedia.org/wiki/Computer_science',
        'https://en.wikipedia.org/wiki/Data_science'
    ]
    
    scraper = UltraAdvancedWebScraper(use_selenium=False)
    
    try:
        # Crawl each starting URL with enhanced error recovery
        for i, start_url in enumerate(start_urls, 1):
            print(f"\n{'='*60}")
            print(f"🌐 Crawling {i}/{len(start_urls)}: {start_url}")
            print(f"{'='*60}")
            
            try:
                scraper.crawl_website(start_url, max_pages=300, max_depth=2)
            except Exception as e:
                logging.error(f"Error crawling {start_url}: {e}")
                print(f"⚠️ Skipping {start_url} due to error: {e}")
                continue
            
            # Show progress after each URL
            stats = scraper.db.get_stats()
            success_rate = stats['total_scraped'] / (stats['total_scraped'] + stats['total_failed']) * 100 if (stats['total_scraped'] + stats['total_failed']) > 0 else 0
            
            print(f"\n📊 Current Progress:")
            print(f"   📄 Total Scraped: {stats['total_scraped']}")
            print(f"   ❌ Total Failed: {stats['total_failed']}")
            print(f"   📈 Success Rate: {success_rate:.1f}%")
            print(f"   ⭐ High Quality: {stats['high_quality']}")
        
        # Final comprehensive statistics
        final_stats = scraper.db.get_stats()
        final_success_rate = final_stats['total_scraped'] / (final_stats['total_scraped'] + final_stats['total_failed']) * 100 if (final_stats['total_scraped'] + final_stats['total_failed']) > 0 else 0
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"{'='*50}")
        print(f"📊 Total Pages Scraped: {final_stats['total_scraped']}")
        print(f"❌ Total Failed URLs: {final_stats['total_failed']}")
        print(f"📈 Overall Success Rate: {final_success_rate:.1f}%")
        print(f"⭐ High Quality Pages: {final_stats['high_quality']}")
        print(f"📈 Average Quality: {final_stats['avg_quality']:.3f}")
        print(f"🔤 Total Words: {final_stats['total_words']:,}")
        print(f"🌐 Unique Domains: {len(final_stats['domains'])}")
        
        # Show top domains
        if final_stats['domains']:
            print(f"\n🏆 Top Domains:")
            sorted_domains = sorted(final_stats['domains'].items(), key=lambda x: x[1], reverse=True)
            for domain, count in sorted_domains[:15]:
                print(f"   {domain}: {count} pages")
        
        # Export datasets with multiple quality thresholds
        print(f"\n💾 Exporting Datasets...")
        print(f"{'='*30}")
        
        # Export premium quality dataset
        premium_jsonl = scraper.export_dataset(min_quality=0.7, output_format='jsonl')
        premium_csv = scraper.export_dataset(min_quality=0.7, output_format='csv')
        
        # Export high quality dataset
        high_jsonl = scraper.export_dataset(min_quality=0.6, output_format='jsonl')
        high_csv = scraper.export_dataset(min_quality=0.6, output_format='csv')
        
        # Export medium quality for more data
        medium_jsonl = scraper.export_dataset(min_quality=0.4, output_format='jsonl')
        
        print(f"\n✅ EXPORT COMPLETE!")
        print(f"🏆 Premium Quality (0.7+):")
        print(f"   📄 JSONL: {premium_jsonl}")
        print(f"   📊 CSV: {premium_csv}")
        print(f"⭐ High Quality (0.6+):")
        print(f"   📄 JSONL: {high_jsonl}")
        print(f"   📊 CSV: {high_csv}")
        print(f"📚 Medium Quality (0.4+):")
        print(f"   📄 JSONL: {medium_jsonl}")
        
        # Enhanced test prompts
        test_dataset_quality()
        
        print(f"\n🎉 Dataset Creation Complete!")
        print(f"🚀 Ready for LLM Training!")
        
    except Exception as e:
        logging.error(f"Main execution error: {e}")
        print(f"❌ Critical error occurred: {e}")
    
    finally:
        scraper.cleanup()
        print(f"\n🧹 Cleanup completed!")

if __name__ == "__main__":
    create_ultra_robust_dataset()
