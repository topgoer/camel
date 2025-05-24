# Copyright (c) 2025 VerseMind-RAG Contributors
# Licensed under the MIT License

"""
This file contains the refactored _find_index_file and search functions with reduced cognitive complexity.
"""
import os
import json
import datetime
import uuid
from typing import Dict, List, Any, Optional, Tuple

# REFACTORED _find_index_file FUNCTION
# Original cognitive complexity: 32, new target: 15 or less

def _find_index_file_refactored(self, index_id: str) -> Optional[str]:
    """查找指定ID的索引文�?""
    self.logger.debug(f"Searching for index file with index_id='{index_id}'")
    
    # 获取搜索目录
    possible_dirs = self._get_search_directories()
    self.logger.debug(f"Will search in directories: {possible_dirs}")
        
    for dir_path in possible_dirs:
        self.logger.debug(f"Checking directory: {dir_path}")
        
        if not os.path.exists(dir_path):
            self.logger.debug(f"Directory '{dir_path}' does not exist")
            continue
            
        # 在目录中搜索匹配的索引文�?
        matching_file = self._find_matching_index_in_directory(dir_path, index_id)
        if matching_file:
            return matching_file
            
    return None
    
def _get_search_directories(self) -> List[str]:
    """获取索引文件的搜索目录列�?""
    from app.core.config import settings
    vector_db_dir = settings.VECTOR_STORE_PERSIST_DIR if hasattr(settings, 'VECTOR_STORE_PERSIST_DIR') else os.path.join(self.storage_dir, "storage", "vector_db")
    return [self.indices_dir, vector_db_dir]
    
def _find_matching_index_in_directory(self, dir_path: str, index_id: str) -> Optional[str]:
    """在指定目录中查找匹配的索引文�?""
    for filename in os.listdir(dir_path):
        if not filename.endswith(".json"):
            continue
            
        file_path = os.path.join(dir_path, filename)
        index_data = self._safely_read_json_file(file_path)
        
        if not index_data:
            continue
        
        # 检查索引ID是否匹配
        if self._is_index_match(index_data, filename, index_id):
            return file_path
    
    self.logger.debug(f"No index file with index_id='{index_id}' found in {dir_path}")
    return None
    
def _safely_read_json_file(self, file_path: str) -> Optional[Dict[str, Any]]:
    """安全地读取JSON文件，处理可能的异常"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        self.logger.error(f"Could not decode JSON from file: '{os.path.basename(file_path)}'")
    except Exception as e:
        self.logger.error(f"Error reading file '{os.path.basename(file_path)}': {str(e)}")
    return None
    
def _is_index_match(self, index_data: Dict[str, Any], filename: str, index_id: str) -> bool:
    """检查索引数据是否与目标索引ID匹配"""
    # 检查JSON中的索引ID
    internal_index_id = index_data.get("index_id")
    if internal_index_id == index_id:
        self.logger.debug(f"Match found: File '{filename}' contains index_id='{index_id}'")
        return True
        
    # 备用策略：检查文件名是否包含索引ID
    if index_id in filename:
        self.logger.debug(f"Match found by filename: '{filename}' contains index_id='{index_id}'")
        return True
        
    return False


# REFACTORED search FUNCTION
# Original cognitive complexity: 20, new target: 15 or less

def search_refactored(self, index_id_or_collection: str, query: str, top_k: int = 3, 
                    similarity_threshold: float = 0.5, min_chars: int = 100) -> Dict[str, Any]:
    """
    执行语义搜索，支持单个索引或整个集合
    
    参数:
        index_id_or_collection: 索引ID或集合名�?
        query: 查询文本
        top_k: 返回结果数量
        similarity_threshold: 相似度阈�?(降低�?.5以提高召回率)
        min_chars: 最小字符数
    
    返回:
        包含搜索结果的字�?
    """
    self.logger.debug(f"Starting search with index_id_or_collection={index_id_or_collection}, query={query}, top_k={top_k}, similarity_threshold={similarity_threshold}")
    
    original_id_or_collection = index_id_or_collection
    start_time = datetime.datetime.now()
    
    # 初始化搜索环�?
    search_info = self._initialize_search_info(
        index_id_or_collection, query, top_k, similarity_threshold, min_chars
    )
    
    # 执行搜索流程
    search_results, index_files, collection_info = self._execute_search_process(
        index_id_or_collection, original_id_or_collection, query, 
        top_k, similarity_threshold, min_chars, search_info
    )
    
    # 处理文档元数�?
    document_filename, document_id = self._process_document_metadata(
        search_info, search_results
    )
    
    # 生成结果标识符和计时
    search_id, timestamp = self._generate_result_identifiers()
    total_time = datetime.datetime.now() - start_time
    search_info["timing"]["total"] = total_time.total_seconds()
    
    # 构建最终结果对�?
    collection_display_name = self._generate_collection_display_name(
        collection_info, original_id_or_collection
    )
    
    result = self._build_result_object(
        search_id, timestamp, query, original_id_or_collection,
        collection_display_name, collection_info, index_files,
        document_id, document_filename, top_k, similarity_threshold,
        min_chars, search_results, search_info
    )
    
    # 保存结果并记录日�?
    self._save_and_log_results(result, search_results)
    
    return result

def _execute_search_process(self, index_id_or_collection: str, original_id_or_collection: str, 
                           query: str, top_k: int, similarity_threshold: float, min_chars: int, 
                           search_info: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    """执行完整的搜索流程，包括查找索引、准备向量和搜索"""
    # 查找索引文件
    index_files = self._find_and_validate_index_files(index_id_or_collection, search_info)
    
    # 初始化集合信�?
    collection_info = self._initialize_collection_info(original_id_or_collection)
    
    # 准备索引数据
    index_data = self._prepare_index_data(index_files, search_info)
    
    # 准备查询向量
    query_vector = self._prepare_query_vector(search_info, index_data, query)
    
    # 执行向量搜索
    search_results, collection_info = self._execute_vector_search(
        query_vector, index_files, index_data, collection_info, 
        top_k, similarity_threshold, min_chars, search_info
    )
    
    return search_results, index_files, collection_info

def _prepare_index_data(self, index_files: List[str], search_info: Dict[str, Any]) -> Dict[str, Any]:
    """准备索引数据，从第一个索引文件中加载"""
    index_file = index_files[0]
    search_info["status"]["index_file_found"] = True
    search_info["index_file_path"] = index_file
    
    try:
        return self._load_index_data(index_file, search_info)
    except Exception as e:
        error_msg = f"读取索引文件失败: {str(e)}"
        self.logger.error(error_msg)
        search_info["status"]["error"] = error_msg
        raise ValueError(error_msg)

def _execute_vector_search(self, query_vector: List[float], index_files: List[str], 
                          index_data: Dict[str, Any], collection_info: Dict[str, Any], 
                          top_k: int, similarity_threshold: float, min_chars: int,
                          search_info: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """执行向量搜索并记录性能指标"""
    search_start_time = datetime.datetime.now()
    self.logger.debug(f"Performing vector search with {len(query_vector)}-dimensional query vector")
    
    # 执行搜索并获取结�?
    search_results, collection_info = self._perform_vector_search(
        query_vector, index_files, index_data, collection_info, 
        top_k, similarity_threshold, min_chars
    )
    
    # 记录搜索时间和统计信�?
    search_time = datetime.datetime.now() - search_start_time
    search_info["timing"]["vector_search"] = search_time.total_seconds()
    
    # 计算搜索统计信息
    self._calculate_search_stats(search_results, search_info)
    search_info["collection_info"] = collection_info
    
    return search_results, collection_info

def _process_document_metadata(self, search_info: Dict[str, Any], 
                              search_results: List[Dict[str, Any]]) -> Tuple[str, str]:
    """处理文档元数据，确保有有效的文件�?""
    document_filename = search_info.get("document_filename", "")
    document_id = search_info.get("document_id", "")
    self.logger.debug(f"Initial document_filename: {document_filename}, document_id: {document_id}")
    
    # 如果没有文件名但有文档ID，尝试提�?
    if not document_filename and document_id:
        document_filename = self._extract_document_filename_from_sources(
            document_id, search_results, search_info
        )
    
    # 美化文件�?
    document_filename = self._clean_document_filename(document_filename)
    
    # 确保文件名不为None
    if document_filename is None:
        document_filename = ""
        self.logger.warning(f"document_filename is None, setting to empty string")
        
    # 如果仍然没有有效文件名，使用文档ID的一部分
    if not document_filename and document_id:
        document_filename = self._create_fallback_filename(document_id)
    
    self.logger.debug(f"Final document_filename: '{document_filename}'")
    return document_filename, document_id

def _extract_document_filename_from_sources(self, document_id: str, 
                                          search_results: List[Dict[str, Any]],
                                          search_info: Dict[str, Any]) -> str:
    """从各种来源尝试提取文档文件名"""
    # 1. 从文档ID部分提取
    extracted_name = self._extract_filename_from_document_id_parts(document_id)
    if extracted_name:
        return extracted_name
        
    # 2. 从搜索结果中提取
    extracted_filename = self._extract_document_filename_from_results(search_results)
    if extracted_filename:
        search_info["document_filename"] = extracted_filename
        return extracted_filename
        
    return ""

def _create_fallback_filename(self, document_id: str) -> str:
    """从文档ID创建备用文件�?""
    # 使用文档ID的前30个字符，避免过长
    filename = document_id[:30]
    if len(document_id) > 30:
        filename += "..."
    return filename

def _generate_result_identifiers(self) -> Tuple[str, str]:
    """生成搜索结果标识�?""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    search_id = str(uuid.uuid4())[:8]
    return search_id, timestamp

def _build_result_object(self, search_id: str, timestamp: str, query: str,
                        original_id_or_collection: str, collection_display_name: str,
                        collection_info: Dict[str, Any], index_files: List[str],
                        document_id: str, document_filename: str, top_k: int, 
                        similarity_threshold: float, min_chars: int,
                        search_results: List[Dict[str, Any]], 
                        search_info: Dict[str, Any]) -> Dict[str, Any]:
    """构建搜索结果对象"""
    result = {
        "search_id": search_id,
        "timestamp": timestamp,
        "query": query,
        "index_id_or_collection": original_id_or_collection,
        "collection_name": original_id_or_collection,
        "collection_display_name": collection_display_name,
        "document_count": len(collection_info["document_ids"]),
        "document_id": document_id if len(index_files) == 1 else None,
        "document_filename": document_filename if len(index_files) == 1 else collection_display_name,
        "top_k": top_k,
        "similarity_threshold": similarity_threshold,
        "min_chars": min_chars,
        "results": search_results,
        "search_info": search_info
    }
    
    # 添加便于访问的辅助字�?
    result["search_info"]["document_filename"] = document_filename if len(index_files) == 1 else collection_display_name
    result["search_info"]["collection_display_name"] = collection_display_name
    
    return result

def _save_and_log_results(self, result: Dict[str, Any], search_results: List[Dict[str, Any]]) -> None:
    """保存搜索结果并记录日�?""
    # 保存搜索结果
    result_file = self._save_search_results(result, result["search_id"], result["timestamp"])
    result["result_file"] = result_file
    
    # 打印搜索结果摘要
    if search_results:
        similarities = [f"{r['similarity']:.4f}" for r in search_results]
        self.logger.debug(f"Found {len(search_results)} results with similarities: {similarities}")
    else:
        self.logger.debug("No results found matching the criteria")

