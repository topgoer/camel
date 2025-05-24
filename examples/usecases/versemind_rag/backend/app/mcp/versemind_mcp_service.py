"""
VerseMind MCP Service

This module implements an MCP-compatible service that exposes VerseMind-RAG functionalities
through the Model Context Protocol (MCP).

The service provides tools to interact with VerseMind-RAG's knowledge bases, search capabilities,
and text generation features.
"""

import os
import sys
import json
import traceback
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging

from camel.utils.mcp import MCPServer

# Set up logging
logger = logging.getLogger(__name__)

# Decorator for MCP tools to manage errors
def mcp_tool_with_error_handling(func):
    """Decorator that wraps MCP tool methods to handle errors."""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": error_msg
            }
    return wrapper

@MCPServer(function_names=[
    # Data access
    "get_versemind_data", 
    
    # RAG capabilities
    "list_knowledge_bases",
    "get_knowledge_base_info", 
    "search_knowledge_base",
    "list_available_models",
    
    # Advanced features
    "execute_python_with_versemind_data", 
    "versemind_multi_round"
])
class VersemindMCPService:
    """Service that provides access to VerseMind-RAG functionalities through MCP."""
    
    def __init__(self):
        """Initialize the service with access to VerseMind data and RAG capabilities."""
        self.title = None
        self.reference = None
        
        # Attempt to load current context from global environment if available
        self._load_from_globals()
        
        # Path to document storage
        self.storage_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../storage'))
        self.documents_dir = os.path.join(self.storage_dir, "documents")
        self.indices_dir = os.path.join(self.storage_dir, "indices")
        
        # Available models (would typically be loaded from config)
        self.models = {
            "ollama": ["llama3:8b", "codellama:7b", "mistral:7b"],
            "openai": ["gpt-3.5-turbo", "gpt-4o"],
            "deepseek": ["deepseek-chat", "deepseek-reasoner"]
        }
        
        logger.info("VerseMind MCP Service initialized")
        
    def _load_from_globals(self):
        """Load title and reference from globals if available."""
        try:
            main_module = sys.modules.get('__main__')
            if main_module:
                if hasattr(main_module, 'title'):
                    self.title = main_module.title
                if hasattr(main_module, 'reference'):
                    self.reference = main_module.reference
              # Check if we loaded anything
            if self.title or self.reference:
                logger.info(f"Loaded VerseMind data: title='{self.title}' and reference data of length {len(self.reference) if self.reference else 0}")
            else:
                logger.debug("No VerseMind data loaded from globals")  # Downgraded from warning to debug
        except Exception as e:
            logger.error(f"Error loading from globals: {e}")
    
    def _get_knowledge_bases(self) -> List[Dict[str, Any]]:
        """Get list of available knowledge bases from the documents directory."""
        knowledge_bases = []
        
        try:
            # List knowledge bases from document directory
            if os.path.exists(self.documents_dir):
                for filename in os.listdir(self.documents_dir):
                    if filename.endswith(('.pdf', '.txt', '.md', '.docx')):
                        kb_path = os.path.join(self.documents_dir, filename)
                        kb_info = {
                            "id": filename.split('.')[0],
                            "name": filename,
                            "file_type": filename.split('.')[-1],
                            "size_bytes": os.path.getsize(kb_path),
                            "last_modified": os.path.getmtime(kb_path)
                        }
                        knowledge_bases.append(kb_info)
        except Exception as e:
            logger.error(f"Error getting knowledge bases: {e}")
        
        return knowledge_bases
    
    @mcp_tool_with_error_handling
    async def get_versemind_data(self) -> Dict[str, Any]:
        """Get current VerseMind data (title and reference).
        
        Returns:
            Dictionary containing title and reference data, plus status info.
        """
        # Refresh data in case it changed in the main module
        self._load_from_globals()
        
        # Return the current VerseMind data
        return {
            "success": True,
            "title": self.title,
            "reference": self.reference,
            "debug_info": {
                "available_globals": list(sys.modules.get('__main__', {}).__dict__.keys())
                if '__main__' in sys.modules else []
            }
        }
        
    @mcp_tool_with_error_handling
    async def list_knowledge_bases(self) -> Dict[str, Any]:
        """List all available knowledge bases in VerseMind-RAG.
        
        Returns:
            Dictionary containing list of knowledge bases.
        """
        knowledge_bases = self._get_knowledge_bases()
        
        return {
            "success": True,
            "knowledge_bases": knowledge_bases,
            "count": len(knowledge_bases)
        }
    
    @mcp_tool_with_error_handling
    async def get_knowledge_base_info(self, kb_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific knowledge base.
        
        Args:
            kb_id: The ID of the knowledge base.
            
        Returns:
            Dictionary containing knowledge base details.
        """
        knowledge_bases = self._get_knowledge_bases()
        
        # Find the requested knowledge base
        kb_info = next((kb for kb in knowledge_bases if kb["id"] == kb_id), None)
        
        if not kb_info:
            return {
                "success": False,
                "error": f"Knowledge base with ID '{kb_id}' not found"
            }
          # Add additional information if available
        kb_info["chunks_count"] = 0  # Would be populated from actual chunking data
        kb_info["status"] = "ready"  # Would reflect actual processing status
        
        return {
            "success": True,
            "knowledge_base": kb_info
        }
    
    @mcp_tool_with_error_handling
    async def list_available_models(self) -> Dict[str, Any]:
        """List all available models that can be used with VerseMind-RAG.
        
        Returns:
            Dictionary containing available models grouped by provider.
        """
        return {
            "success": True,
            "models": self.models
        }
    
    @mcp_tool_with_error_handling
    async def search_knowledge_base(self, query: str, kb_id: str = None, 
                                   model: str = None, similarity_threshold: float = 0.5,
                                   max_results: int = 5) -> Dict[str, Any]:
        """Search a knowledge base with the given query.
        
        Args:
            query: The search query
            kb_id: ID of the knowledge base to search (if None, search all)
            model: Model to use for embeddings and ranking
            similarity_threshold: Minimum similarity score for results
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing search results.
        """
        try:
            # This would actually use the VerseMind-RAG search service
            # For this example, we'll simulate a search
            
            # Validate knowledge base exists if specified
            if kb_id:
                knowledge_bases = self._get_knowledge_bases()
                kb = next((kb for kb in knowledge_bases if kb["id"] == kb_id), None)
                if not kb:
                    return {
                        "success": False,
                        "error": f"Knowledge base with ID '{kb_id}' not found"
                    }
            
            # Simulate search results
            import random
            mock_results = []
              # Use existing reference as content if available
            content_source = self.reference or "This is sample content from the knowledge base."
            
            # Generate some mock search results
            for _ in range(min(3, max_results)):
                # Extract a random segment from content as a "search result"
                words = content_source.split()
                if len(words) > 20:
                    start = random.randint(0, len(words) - 20)
                    snippet = " ".join(words[start:start+20])
                else:
                    snippet = content_source
                
                # Create a mock result
                mock_results.append({
                    "text": snippet,
                    "similarity": round(random.uniform(similarity_threshold, 0.99), 2),
                    "source": f"document_{kb_id or 'default'}.txt",
                    "page": random.randint(1, 10) if random.random() > 0.5 else None
                })
            
            # Sort by similarity
            mock_results.sort(key=lambda x: x["similarity"], reverse=True)
            
            return {
                "success": True,
                "query": query,
                "knowledge_base": kb_id or "all",
                "model_used": model or "default",
                "results": mock_results,
                "result_count": len(mock_results)
            }
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return {
                "success": False,
                "error": f"Search failed: {str(e)}"
            }
    
    @mcp_tool_with_error_handling
    async def execute_python_with_versemind_data(self, code: str) -> Dict[str, Any]:
        """Execute Python code with access to title and reference variables.
        
        Args:
            code: Python code to execute.
            
        Returns:
            Dictionary with execution output and status.
        """
        import io
        from contextlib import redirect_stdout, redirect_stderr
        # Refresh data in case it changed
        self._load_from_globals()
        
        # Prepare execution environment with VerseMind data
        exec_globals = {
            'title': self.title,  # 简洁的标题
            'reference': self.reference,  # 完整的参考内容（可能包含结构化数据）
            'prompt': self.title,  # 为向后兼容提供 prompt 变量
            'complete_reference': self.reference,  # 为向后兼容提供 complete_reference 变量
        }
        
        # Capture stdout and stderr
        stdout = io.StringIO()
        stderr = io.StringIO()
        
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exec(code, exec_globals)
              # Return execution result
            return {
                "success": True,
                "output": stdout.getvalue(),
                "errors": stderr.getvalue() if stderr.getvalue() else None
            }
        except Exception as e:
            error_output = f"Error executing code: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_output)
            return {
                "success": False,
                "output": stdout.getvalue() if stdout.getvalue() else None,
                "errors": stderr.getvalue() + "\n" + error_output if stderr.getvalue() else error_output
            }
            
    @mcp_tool_with_error_handling
    async def versemind_multi_round(self, query: str, kb_id: str = None, 
                                   model: str = None, max_rounds: int = 3) -> Dict[str, Any]:
        """Run a multi-round dialog with RAG using the specified model and knowledge base.
        
        Provides advanced conversational access to knowledge base content,
        maintaining context across multiple questions.
        
        Args:
            query: The initial user query
            kb_id: Knowledge base ID to use for RAG context
            model: The model ID to use for the dialog
            max_rounds: Maximum number of conversation rounds to simulate
            
        Returns:
            Dictionary containing results of the dialog.
        """
        # Refresh data in case it changed
        self._load_from_globals()
        
        # If model is not specified, use a default
        if not model:
            model = "llama3:8b"  # A reasonable default model
            
        # First, search the knowledge base for relevant content
        search_result = await self.search_knowledge_base(
            query=query, 
            kb_id=kb_id,
            model=model
        )
        
        # Extract search context
        search_context = ""
        if search_result.get("success") and search_result.get("results"):
            search_context = "Based on the following information:\n\n"
            for i, result in enumerate(search_result.get("results", [])):
                search_context += f"[{i+1}] {result.get('text', '')}\n\n"
        
        # Simulate a multi-round dialog with the model
        # In a real implementation, this would use the actual model specified
        conversation = [
            {
                "role": "user",
                "content": query
            },
            {
                "role": "assistant",
                "content": f"I've found some relevant information about your query. {search_context[:100]}..."
            }
        ]
        
        # Generate follow-up exchanges
        follow_ups = [
            {"user": "Can you elaborate more on this topic?", 
             "assistant": "Certainly! Based on the knowledge base, I can provide these additional details..."},
            {"user": "What are the key points to remember?", 
             "assistant": "The most important aspects to remember are:\n1. First key point\n2. Second key point\n3. Third key point"},
            {"user": "How does this relate to the previous topic?", 
             "assistant": "This connects to our earlier discussion in several ways..."}
        ]
        
        # Add follow-up rounds (limited by max_rounds)
        for idx in range(min(len(follow_ups), max_rounds-1)):
            conversation.append({"role": "user", "content": follow_ups[idx]["user"]})
            conversation.append({"role": "assistant", "content": follow_ups[idx]["assistant"]})
        
        # Prepare summary information
        summary = {
            "topic": self.title or kb_id or "Knowledge base search",
            "query": query,
            "knowledge_base": kb_id or "default",
            "model_used": model,
            "rounds": len(conversation) // 2,
            "key_points": "- Retrieved relevant information from knowledge base\n- Addressed user query with context\n- Provided follow-up responses",
        }
        
        return {
            "success": True,
            "model_used": model,
            "conversation": conversation,
            "summary": summary,
            "search_results": search_result.get("results", [])
        }


# For direct testing
if __name__ == "__main__":
    import asyncio
    
    async def test_service():
        """Test the VerseMind MCP Service."""
        service = VersemindMCPService()
        
        # Set up test data
        service.title = "Test Title"
        service.reference = "This is a test reference document."
        
        # Test get_versemind_data
        result = await service.get_versemind_data()
        print(f"get_versemind_data result: {json.dumps(result, indent=2)}")
          # Test execute_python_with_versemind_data
        test_code = """
print(f"Title: {title}")
print(f"Reference: {reference}")
"""
        result = await service.execute_python_with_versemind_data(test_code)
        print(f"execute_python_with_versemind_data result: {json.dumps(result, indent=2)}")
        
        # Test versemind_multi_round
        result = await service.versemind_multi_round(
            query="What can you tell me about this document?", 
            model="test-model"
        )
        print(f"versemind_multi_round result: {json.dumps(result, indent=2)}")
    
    asyncio.run(test_service())
