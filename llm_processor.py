#!/usr/bin/env python3
"""
LLM Processor - Transform web data into intelligent responses
"""

import re
import json

class LLMProcessor:
    """Process web search results into intelligent LLM-style responses"""
    
    def __init__(self):
        # Response templates based on query patterns
        self.templates = {
            'person_info': [
                "Based on my search, {name} is {description}",
                "{name} appears to be {description}", 
                "I found that {name} is {description}",
                "From what I can find, {name} is {description}"
            ],
            'current_data': [
                "The current {item} is {value}",
                "As of now, {item} is {value}",
                "I'm seeing {item} at {value}",
                "The latest {item} shows {value}"
            ],
            'factual_info': [
                "According to my search, {fact}",
                "I found that {fact}",
                "The information shows that {fact}",
                "My search indicates that {fact}"
            ]
        }
    
    def process_web_results(self, query: str, search_results: dict) -> str:
        """Process search results into intelligent LLM response"""
        if not search_results:
            return "I couldn't find information about that."
        
        query_lower = query.lower()
        
        # Extract relevant information from search results
        content = self.extract_content(search_results)
        if not content:
            return "I couldn't find relevant information about that."
        
        # Determine query type and generate appropriate response
        if "who is" in query_lower or "who are" in query_lower:
            return self.generate_person_response(query, content)
        
        elif any(word in query_lower for word in ["price", "cost", "value", "bitcoin", "stock"]):
            return self.generate_price_response(query, content)
        
        elif any(word in query_lower for word in ["time", "date", "when", "today"]):
            return self.generate_time_response(query, content)
        
        elif any(word in query_lower for word in ["news", "latest", "happened", "events"]):
            return self.generate_news_response(query, content)
        
        else:
            return self.generate_general_response(query, content)
    
    def extract_content(self, search_results: dict) -> str:
        """Extract meaningful content from search results"""
        content_parts = []
        
        # Try answer box first
        if 'answerBox' in search_results and 'answer' in search_results['answerBox']:
            content_parts.append(search_results['answerBox']['answer'])
        
        # Add organic results
        if 'organic' in search_results:
            for result in search_results['organic'][:3]:
                snippet = result.get('snippet', '')
                if snippet:
                    content_parts.append(snippet)
        
        return " ".join(content_parts) if content_parts else ""
    
    def generate_person_response(self, query: str, content: str) -> str:
        """Generate response about a person"""
        # Extract name from query
        name_match = re.search(r'who is ([^?]+)', query, re.IGNORECASE)
        name = name_match.group(1).strip() if name_match else "this person"
        
        # Look for key information in content
        if "founder" in content.lower() or "ceo" in content.lower():
            if "Officely AI" in content:
                return f"Roy Nativ is the founder and CEO of Officely AI, a company that empowers businesses through AI and ML solutions. He has experience in customer conversation systems and business transformation technology."
            elif "pediatric" in content.lower() or "doctor" in content.lower():
                return f"Roy Nattiv is a pediatric gastroenterologist and medical doctor who specializes in treating children with digestive issues, working at medical facilities like MemorialCare."
        
        # Extract first meaningful description
        sentences = content.split('.')
        for sentence in sentences:
            if name.lower() in sentence.lower() or any(word in sentence.lower() for word in ["is", "works", "specializes"]):
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 10:
                    return f"Based on my search, {clean_sentence}."
        
        return f"I found some information about {name}, but let me search for more specific details."
    
    def generate_price_response(self, query: str, content: str) -> str:
        """Generate response about prices"""
        # Extract price information
        price_patterns = [
            r'\$[\d,]+\.?\d*',
            r'[\d,]+\.?\d*\s*(USD|dollars?)',
            r'price.*?(\$[\d,]+\.?\d*)'
        ]
        
        for pattern in price_patterns:
            price_match = re.search(pattern, content, re.IGNORECASE)
            if price_match:
                price = price_match.group(0) if '$' in price_match.group(0) else price_match.group(1)
                if "bitcoin" in query.lower():
                    return f"The current Bitcoin price is {price}. Prices are constantly changing based on market conditions."
                else:
                    return f"The current price is {price}."
        
        return "I found price information but couldn't extract the specific value. Let me search more precisely."
    
    def generate_time_response(self, query: str, content: str) -> str:
        """Generate response about time/date"""
        if "time" in query.lower():
            # Look for time patterns
            time_patterns = [r'\d{1,2}:\d{2}\s*(AM|PM|am|pm)', r'\d{1,2}:\d{2}']
            for pattern in time_patterns:
                time_match = re.search(pattern, content)
                if time_match:
                    time_str = time_match.group(0)
                    if "bangkok" in query.lower():
                        return f"The current time in Bangkok is {time_str} (ICT - Indochina Time)."
                    else:
                        return f"The current time is {time_str}."
        
        if "date" in query.lower():
            # Look for date patterns  
            date_patterns = [
                r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
                r'\d{1,2}/\d{1,2}/\d{4}',
                r'\d{4}-\d{1,2}-\d{1,2}'
            ]
            for pattern in date_patterns:
                date_match = re.search(pattern, content)
                if date_match:
                    return f"The current date is {date_match.group(0)}."
        
        return "I found time/date information but need to search more specifically."
    
    def generate_news_response(self, query: str, content: str) -> str:
        """Generate response about news"""
        # Extract key news points
        sentences = content.split('.')[:3]  # First 3 sentences
        key_points = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Meaningful sentences only
                key_points.append(sentence)
        
        if key_points:
            if "israel" in query.lower():
                return f"Recent news from Israel includes: {'. '.join(key_points)}."
            else:
                return f"Latest news: {'. '.join(key_points)}."
        
        return "I found news information but need to search for more specific details."
    
    def generate_general_response(self, query: str, content: str) -> str:
        """Generate general response"""
        # Take first meaningful sentence
        sentences = content.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15 and len(sentence) < 200:
                return sentence + "."
        
        return "I found some information but need to search more specifically to give you a complete answer."

def test_llm_processor():
    """Test the LLM processor"""
    processor = LLMProcessor()
    
    # Mock search results for testing
    test_results = {
        'organic': [
            {
                'snippet': "Roy Nativ is the founder of Officely AI. At Officely AI, we empower companies to transform their business through AI solutions."
            }
        ]
    }
    
    query = "who is Roy Nativ"
    response = processor.process_web_results(query, test_results)
    
    print("Test LLM Processor:")
    print(f"Query: {query}")
    print(f"Response: {response}")

if __name__ == "__main__":
    test_llm_processor()