# src/web_crawler.py - Real-time News & Updates Crawler

import requests
from bs4 import BeautifulSoup
import feedparser
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import json
from textblob import TextBlob
import re

import config
from .cache_manager import CacheManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebCrawler:
    """Crawls web for renewable energy and cross-border energy news."""
    
    def __init__(self):
        self.cache_manager = CacheManager()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        self.timeout = config.CRAWLER_TIMEOUT_SECONDS
    
    def scrape_news_site(self, url: str) -> List[Dict]:
        """
        Scrape news articles from a website.
        
        Args:
            url: Website URL to scrape
            
        Returns:
            List of article dictionaries
        """
        articles = []
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Generic article extraction (adjust selectors based on site)
            article_elements = soup.find_all('article', limit=10)
            if not article_elements:
                article_elements = soup.find_all('div', class_=re.compile('article|post|news', re.I), limit=10)
            
            for element in article_elements:
                try:
                    title_tag = element.find(['h1', 'h2', 'h3', 'a'])
                    title = title_tag.get_text(strip=True) if title_tag else "No title"
                    
                    link_tag = element.find('a', href=True)
                    link = link_tag['href'] if link_tag else url
                    if not link.startswith('http'):
                        link = url + link
                    
                    summary_tag = element.find(['p', 'summary', 'description'])
                    summary = summary_tag.get_text(strip=True)[:200] if summary_tag else "No summary"
                    
                    date_tag = element.find(['time', 'span'], class_=re.compile('date|time', re.I))
                    date_str = date_tag.get_text(strip=True) if date_tag else str(datetime.now())
                    
                    article = {
                        "title": title,
                        "link": link,
                        "summary": summary,
                        "source": url,
                        "date": date_str,
                        "scraped_at": datetime.now().isoformat(),
                        "sentiment": None
                    }
                    
                    articles.append(article)
                except Exception as e:
                    logger.warning(f"Error parsing article element: {e}")
                    continue
            
            logger.info(f"Scraped {len(articles)} articles from {url}")
            return articles
        
        except requests.RequestException as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return []
    
    def fetch_rss_feeds(self, url: str) -> List[Dict]:
        """
        Fetch articles from RSS feed.
        
        Args:
            url: RSS feed URL
            
        Returns:
            List of article dictionaries
        """
        articles = []
        try:
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:config.MAX_NEWS_ARTICLES]:
                article = {
                    "title": entry.get('title', 'No title'),
                    "link": entry.get('link', ''),
                    "summary": entry.get('summary', '')[:200],
                    "source": feed.feed.get('title', url),
                    "date": entry.get('published', str(datetime.now())),
                    "scraped_at": datetime.now().isoformat(),
                    "sentiment": None
                }
                articles.append(article)
            
            logger.info(f"Fetched {len(articles)} articles from RSS feed: {url}")
            return articles
        
        except Exception as e:
            logger.error(f"Failed to fetch RSS feed {url}: {e}")
            return []
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of text using TextBlob.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        try:
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity  # -1 to 1
            subjectivity = analysis.sentiment.subjectivity  # 0 to 1
            
            if polarity > 0.1:
                sentiment = "positive"
            elif polarity < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            return {
                "sentiment": sentiment,
                "polarity": round(polarity, 3),
                "subjectivity": round(subjectivity, 3)
            }
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"sentiment": "unknown", "polarity": 0, "subjectivity": 0}
    
    def filter_relevant_articles(self, articles: List[Dict], keywords: List[str] = None) -> List[Dict]:
        """
        Filter articles by relevance using keywords.
        
        Args:
            articles: List of articles to filter
            keywords: Keywords to search for
            
        Returns:
            Filtered list of relevant articles
        """
        if keywords is None:
            keywords = [
                "renewable energy", "solar", "wind", "grid integration",
                "cross-border", "electricity trade", "interconnection",
                "power transmission", "energy efficiency", "sustainability",
                "ISA", "International Solar Alliance", "microgrid"
            ]
        
        relevant = []
        for article in articles:
            text = (article['title'] + ' ' + article['summary']).lower()
            if any(keyword.lower() in text for keyword in keywords):
                relevant.append(article)
        
        logger.info(f"Filtered {len(relevant)} relevant articles from {len(articles)}")
        return relevant
    
    def crawl_all_sources(self) -> List[Dict]:
        """
        Crawl all configured news sources.
        
        Returns:
            Combined list of all articles
        """
        all_articles = []
        
        # Try to get from cache first
        cached = self.cache_manager.get("all_news")
        if cached:
            return cached
        
        # Scrape from configured sources
        for source_url in config.NEWS_SOURCES:
            try:
                logger.info(f"Crawling: {source_url}")
                articles = self.scrape_news_site(source_url)
                
                # Try RSS feed if HTML scraping fails
                if not articles:
                    rss_url = source_url.replace('https://', '').replace('http://', '')
                    articles = self.fetch_rss_feeds(f"https://{rss_url}/feed/")
                
                all_articles.extend(articles)
            except Exception as e:
                logger.error(f"Error crawling {source_url}: {e}")
                continue
        
        # Filter for relevance
        relevant_articles = self.filter_relevant_articles(all_articles)
        
        # Analyze sentiment for each article
        for article in relevant_articles:
            sentiment_analysis = self.analyze_sentiment(
                article['title'] + ' ' + article['summary']
            )
            article['sentiment'] = sentiment_analysis
        
        # Sort by date
        relevant_articles.sort(
            key=lambda x: datetime.fromisoformat(x['scraped_at']),
            reverse=True
        )
        
        # Limit to max articles
        relevant_articles = relevant_articles[:config.MAX_NEWS_ARTICLES]
        
        # Cache the results
        self.cache_manager.set("all_news", relevant_articles, ttl_hours=config.CRAWLER_UPDATE_INTERVAL_MINUTES // 60)
        
        logger.info(f"Crawl completed. Found {len(relevant_articles)} relevant articles")
        return relevant_articles
    
    def get_topic_news(self, topic: str) -> List[Dict]:
        """
        Get news articles for a specific topic.
        
        Args:
            topic: Topic to search for
            
        Returns:
            List of relevant articles
        """
        all_articles = self.crawl_all_sources()
        
        topic_articles = [
            article for article in all_articles
            if topic.lower() in article['title'].lower() or 
               topic.lower() in article['summary'].lower()
        ]
        
        return sorted(
            topic_articles,
            key=lambda x: x.get('sentiment', {}).get('polarity', 0),
            reverse=True
        )
    
    def get_sentiment_summary(self, articles: List[Dict]) -> Dict:
        """
        Get overall sentiment summary for a list of articles.
        
        Args:
            articles: List of articles
            
        Returns:
            Sentiment statistics
        """
        if not articles:
            return {"positive": 0, "neutral": 0, "negative": 0, "total": 0}
        
        sentiments = {"positive": 0, "neutral": 0, "negative": 0}
        total_polarity = 0
        
        for article in articles:
            sentiment = article.get('sentiment', {})
            sentiment_type = sentiment.get('sentiment', 'neutral')
            sentiments[sentiment_type] += 1
            total_polarity += sentiment.get('polarity', 0)
        
        return {
            "positive": sentiments["positive"],
            "neutral": sentiments["neutral"],
            "negative": sentiments["negative"],
            "total": len(articles),
            "average_polarity": round(total_polarity / len(articles), 3)
        }


# Example usage function
def get_latest_news():
    """Convenience function to get latest news."""
    crawler = WebCrawler()
    return crawler.crawl_all_sources()
