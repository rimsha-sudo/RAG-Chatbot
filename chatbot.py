import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging
import re
from typing import List

logger = logging.getLogger(__name__)

class ChatBot:
    def __init__(self, document_chunks):
        self.document_chunks = document_chunks
        self.full_text = " ".join(document_chunks)
        
        # Initializing  embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Created vector store if multiple chunks
        if len(document_chunks) > 1:
            self._create_vector_store()
        else:
            self.use_vector_search = False
    
    def _create_vector_store(self):
       #creating a vector store and using FAISS
        try:
            embeddings = self.embedding_model.encode(self.document_chunks)
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))
            self.use_vector_search = True
            logger.info(f"Vector store created with {len(self.document_chunks)} chunks")
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            self.use_vector_search = False
    
    def _find_relevant_content(self, query: str) -> str:
        """Find relevant content using multiple strategies"""
        query_lower = query.lower()
        
        #to get better responses using different techniques and making sure
        # the chatbot handles different types of questions  accuractely 
        # Vector search if available
        if hasattr(self, 'use_vector_search') and self.use_vector_search:
            try:
                query_embedding = self.embedding_model.encode([query])
                k = min(3, len(self.document_chunks))
                distances, indices = self.index.search(query_embedding.astype('float32'), k)
                relevant_chunks = [self.document_chunks[i] for i in indices[0]]
                return " ".join(relevant_chunks)
            except:
                pass
        
        # Strategy 2: Keyword matching with context
        return self._keyword_search_with_context(query_lower)
    
    def _keyword_search_with_context(self, query_lower: str) -> str:
        """Smart keyword search that finds relevant sections with context"""
        
        # Extract important keywords from query
        query_words = [word for word in query_lower.split() if len(word) > 2]
        
        # Split document into sentences
        sentences = re.split(r'[.!?]+', self.full_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Score each sentence based on keyword matches
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            score = 0
            
            # Count keyword matches
            for word in query_words:
                if word in sentence_lower:
                    score += 1
            
            # Bonus for exact phrase matches
            if any(phrase in sentence_lower for phrase in [query_lower, " ".join(query_words[:2])]):
                score += 2
            
            sentence_scores.append((score, i, sentence))
        
        # Sort by score and get top sentences
        sentence_scores.sort(reverse=True, key=lambda x: x[0])
        
        # Get best matching sentences with context
        relevant_sentences = []
        for score, idx, sentence in sentence_scores[:3]:
            if score > 0:  # Only include sentences with keyword matches
                # Add context (previous and next sentence if available)
                context_sentences = []
                
                # Add previous sentence if relevant
                if idx > 0 and len(relevant_sentences) == 0:
                    context_sentences.append(sentences[idx-1])
                
                context_sentences.append(sentence)
                
                # Add next sentence if relevant
                if idx < len(sentences) - 1:
                    next_sentence = sentences[idx + 1]
                    if any(word in next_sentence.lower() for word in query_words):
                        context_sentences.append(next_sentence)
                
                relevant_sentences.extend(context_sentences)
        
        # If no good matches, return first part of document
        if not relevant_sentences:
            return self.full_text[:1000]
        
        return ". ".join(relevant_sentences[:4]) + "."
    
    def _format_answer(self, query: str, content: str) -> str:
        """Format the final answer based on query type and content"""
        query_lower = query.lower()
        
        # Handle list-type questions
        if any(word in query_lower for word in ['types', 'list', 'applications', 'kinds', 'examples']):
            return self._extract_list_format(content, query_lower)
        
        # Handle definition questions
        if any(word in query_lower for word in ['what is', 'define', 'definition', 'meaning']):
            return self._extract_definition(content, query_lower)
        
        # Handle "how" questions
        if query_lower.startswith('how'):
            return self._extract_process(content)
        
        # Default: return relevant content cleanly
        return self._clean_content(content)
    
    def _extract_list_format(self, content: str, query: str) -> str:
        """Extract and format list items"""
        lines = content.split('\n')
        list_items = []
        
        # Look for numbered or bulleted lists
        for line in lines:
            line = line.strip()
            if re.match(r'^[0-9]+\.', line) or line.startswith('-') or line.startswith('•'):
                list_items.append(line)
            elif ':' in line and any(keyword in query for keyword in ['types', 'applications']):
                if 'include' in line.lower():
                    parts = line.split('include')
                    if len(parts) > 1:
                        items = parts[1].split(',')
                        for item in items:
                            list_items.append(f"- {item.strip()}")
        
        if list_items:
            return '\n'.join(list_items)
        
        # Fallback: look for comma-separated items
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            if ',' in sentence and any(keyword in sentence.lower() for keyword in ['include', 'are']):
                return sentence.strip()
        
        return self._clean_content(content)
    
    def _extract_definition(self, content: str, query: str) -> str:
        """Extract clean definitions"""
        # Look for definition patterns
        sentences = re.split(r'[.!?]+', content)
        
        # Find sentence with "is" or "are" that defines the term
        for sentence in sentences:
            sentence = sentence.strip()
            if ' is ' in sentence.lower() or ' are ' in sentence.lower():
                # Check if it's actually defining something
                if len(sentence) > 20 and not sentence.lower().startswith('there'):
                    return sentence + "."
        
        # Fallback to first substantial sentence
        for sentence in sentences:
            if len(sentence.strip()) > 30:
                return sentence.strip() + "."
        
        return self._clean_content(content)
    
    def _extract_process(self, content: str) -> str:
        """Extract process or method descriptions"""
        sentences = re.split(r'[.!?]+', content)
        relevant = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(word in sentence.lower() for word in ['process', 'method', 'work', 'function', 'operate']):
                relevant.append(sentence)
        
        if relevant:
            return ". ".join(relevant[:2]) + "."
        
        return self._clean_content(content)
    
    def _clean_content(self, content: str) -> str:
        """Clean and format content for final output"""
        # Remove extra whitespace and clean up
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Ensure proper sentence ending
        if content and not content.endswith(('.', '!', '?')):
            content += "."
        
        # Limit length
        if len(content) > 500:
            sentences = re.split(r'[.!?]+', content)
            content = ". ".join(sentences[:3]) + "."
        
        return content
    
    def get_response(self, question: str) -> str:
        """Generate response using improved logic"""
        try:
            # Find relevant content
            relevant_content = self._find_relevant_content(question)
            
            # Format answer based on question type
            answer = self._format_answer(question, relevant_content)
            
            # Final cleanup
            if not answer or len(answer.strip()) < 5:
                return "I couldn't find relevant information to answer your question. Try rephrasing or asking about different aspects of the document."
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I encountered an error while processing your question. Please try again with a different question."